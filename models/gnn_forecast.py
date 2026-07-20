"""
models/gnn_forecast.py  —  Engine 1: ward-level AQI Graph Transformer
=================================================================
A spatio-temporal graph *transformer* forecaster for 289 Delhi wards.

Architecture
------------
    dynamic (19 cells)                static (289 wards)
        |                                   |
   [temporal encoder]  GRU over L hours     |
        |  h_cell [C, H]                     |
   gather to wards  h_cell[cell_of_node]     |
        |  [N, H]                            | Linear
        +----------------- concat -----------+
                        |
                 input projection  [N, H]
                        |
        L x  TransformerConv (multi-head, edge-aware)     <-- the "graph transformer"
             edge_dim = (dist_z, bearing_sin, bearing_cos, wind_alignment)
                        |  residual + LayerNorm
                 quantile head  ->  p10 / p50 / p90  (monotone)

Why this shape
--------------
* The spatial mixing is genuine multi-head attention over the wind-aware ward
  graph (TransformerConv), not plain convolution — this is the Graph Transformer
  the project asks for.
* `wind_alignment` = cos(bearing - wind_dir) is injected as a per-timestep edge
  feature, so attention can route messages *downwind* (regional transport) rather
  than isotropically.
* Quantile heads give the p10/p50/p90 band Engine 3 (future) needs, trained with
  pinball loss. Monotonicity (p10<=p50<=p90) is structural (cumulative softplus).

Two modes, one forward
-----------------------
The canonical `forward(x, edge_index, edge_attr)` takes a plain node-feature
matrix — exactly what GNNExplainer / SHAP call. In snapshot mode `x` is the raw,
interpretable `[static | dynamic]` (so explanations are in real feature space);
in temporal mode the trainer swaps the dynamic block for the GRU embedding via
`encode_dynamic()` before calling forward. `--no-temporal` is therefore both a
serving option and the "learns physics, not autocorrelation" ablation.
=================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class WardGraphTransformer(nn.Module):
    def __init__(self, n_static: int, n_dyn: int, hidden: int = 64, heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, edge_dim: int = 4,
                 temporal: bool = True, quantiles=(0.1, 0.5, 0.9),
                 temporal_kind: str = "gru", lookback: int = 24):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.temporal = temporal
        self.temporal_kind = temporal_kind
        self.n_static = n_static
        self.n_dyn = n_dyn
        self.hidden = hidden
        self.quantiles = tuple(quantiles)

        # temporal encoder over the 19 cell sequences (cheap: 19, not 289).
        # `temporal_kind` (improvement 2.5):
        #   gru         — recurrent (default, original)
        #   transformer — self-attention over the lookback window (learned pos-enc)
        #   tcn         — dilated temporal convolutions
        self.gru = None
        if temporal:
            if temporal_kind == "gru":
                self.gru = nn.GRU(n_dyn, hidden, batch_first=True)
            elif temporal_kind == "transformer":
                self.t_in = nn.Linear(n_dyn, hidden)
                self.t_pos = nn.Parameter(torch.randn(1, lookback, hidden) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    hidden, nhead=heads, dim_feedforward=2 * hidden,
                    dropout=dropout, batch_first=True, activation="gelu")
                self.t_enc = nn.TransformerEncoder(layer, num_layers=2)
            elif temporal_kind == "tcn":
                self.t_in = nn.Linear(n_dyn, hidden)
                self.tcn = nn.ModuleList([
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=d, dilation=d)
                    for d in (1, 2, 4)])
                self.tcn_act = nn.GELU()
            else:
                raise ValueError(f"unknown temporal_kind {temporal_kind!r}")
            dyn_in = hidden
        else:
            dyn_in = n_dyn
        self.dyn_in = dyn_in

        self.stat_proj = nn.Linear(n_static, hidden)
        self.dyn_proj = nn.Linear(dyn_in, hidden)
        self.in_proj = nn.Linear(2 * hidden, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                TransformerConv(hidden, hidden // heads, heads=heads,
                                edge_dim=edge_dim, dropout=dropout, beta=True)
            )
            self.norms.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, len(self.quantiles))

    # ---- temporal pre-encoder (call before forward in temporal mode) -----
    def encode_dynamic(self, dyn_seq: torch.Tensor) -> torch.Tensor:
        """[B*C, L, n_dyn] -> [B*C, hidden] sequence summary at the last step."""
        assert self.temporal, "encode_dynamic only valid in temporal mode"
        if self.temporal_kind == "gru":
            _, h = self.gru(dyn_seq)              # h: [1, B*C, hidden]
            return h[-1]
        if self.temporal_kind == "transformer":
            x = self.t_in(dyn_seq) + self.t_pos[:, :dyn_seq.size(1)]
            z = self.t_enc(x)                     # [B*C, L, hidden]
            return z[:, -1]                       # last-step representation
        # tcn
        x = self.t_in(dyn_seq).transpose(1, 2)    # [B*C, hidden, L]
        for conv in self.tcn:
            x = self.tcn_act(conv(x))
        return x[:, :, -1]                        # [B*C, hidden]

    def build_node_input(self, x_static: torch.Tensor, dyn_block: torch.Tensor,
                         cell_of_node: torch.Tensor) -> torch.Tensor:
        """Assemble the [N, n_static + dyn_in] matrix `forward` consumes.

        dyn_block is [C, dyn_in] (GRU embedding, temporal) or [C, n_dyn]
        (snapshot). Gather to wards, concat static.
        """
        dyn_nodes = dyn_block[cell_of_node]            # [N, dyn_in]
        return torch.cat([x_static, dyn_nodes], dim=1)

    # ---- canonical forward (GNNExplainer / SHAP call this) ---------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """x: [N, n_static + dyn_in] -> quantiles [N, Q] (monotone increasing)."""
        stat = self.stat_proj(x[:, :self.n_static])
        dyn = self.dyn_proj(x[:, self.n_static:])
        h = F.relu(self.in_proj(torch.cat([stat, dyn], dim=1)))
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index, edge_attr)
            h = norm(h + self.drop(F.relu(m)))         # residual + norm
        raw = self.head(h)                             # [N, Q]
        # enforce p10 <= p50 <= p90 via cumulative softplus on the deltas
        base = raw[:, :1]
        deltas = F.softplus(raw[:, 1:])
        return torch.cat([base, base + torch.cumsum(deltas, dim=1)], dim=1)

    def median_index(self) -> int:
        return self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """Mean pinball loss over quantiles. pred [M, Q], target [M]."""
    target = target.unsqueeze(1)                       # [M, 1]
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, -1)
    e = target - pred                                  # [M, Q]
    return torch.maximum(q * e, (q - 1) * e).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, Fd, Fs, E, L = 289, 19, 73, 15, 1670, 24
    cell_of_node = torch.randint(0, C, (N,))
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)
    x_static = torch.randn(N, Fs)

    for temporal in (True, False):
        m = WardGraphTransformer(Fs, Fd, temporal=temporal)
        if temporal:
            dyn_seq = torch.randn(C, L, Fd)
            dyn_block = m.encode_dynamic(dyn_seq)      # [C, hidden]
        else:
            dyn_block = torch.randn(C, Fd)             # snapshot
        x = m.build_node_input(x_static, dyn_block, cell_of_node)
        out = m(x, edge_index, edge_attr)
        mono = bool((out[:, 1:] >= out[:, :-1] - 1e-5).all())
        loss = pinball_loss(out[:50], torch.randn(50) * 50 + 200, m.quantiles)
        print(f"temporal={temporal!s:>5} | x {tuple(x.shape)} -> out {tuple(out.shape)} | "
              f"monotone={mono} | params={sum(p.numel() for p in m.parameters()):,} | pinball={loss.item():.2f}")
