"""
advisor/kg/knowledge_graph.py — PART 7: the air-quality knowledge graph.

A typed directed multigraph (NetworkX) linking the domain ontology to the policy
corpus, so retrieval can reason over *relationships*, not just text similarity.

Node types  : source, pollutant, aqi_stage, intervention, weather, policy
Relations   : causes, reduces, recommended_for, restricted_by, effective_under,
              defines, addresses

The ontology edges (source->pollutant, intervention->source, intervention->stage)
are curated domain knowledge; policy nodes are linked in automatically from the
ingested chunks' metadata (aqi_stage, tags). Stored locally as a pickle.

Traversal powers two things:
  * hybrid retrieval (Part 5): from {sources, stage, pollutants} -> related policy
    doc_ids to boost.
  * action generation (Part 10/11): from a source -> interventions that reduce it,
    filtered by the applicable GRAP stage.
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import networkx as nx

from advisor.config import (
    CONFIG, INTERVENTION_FEATURE_MAP, ACTION_TARGET_SOURCE, INTERVENTION_STAGES,
    ACTION_CATALOGUE, GRAP_STAGES,
)

# curated ontology: which pollutants each source emits
SOURCE_POLLUTANTS = {
    "traffic": ["NO2", "CO", "PM2.5"],
    "industrial": ["SO2", "PM2.5", "PM10"],
    "dust": ["PM10", "dust"],
    "biomass_burning": ["PM2.5", "PM10", "CO"],
    "secondary": ["O3", "PM2.5"],
}
# which weather condition amplifies each source
SOURCE_WEATHER = {
    "biomass_burning": ["nw_wind"],
    "secondary": ["stagnation", "high_temperature"],
    "dust": ["high_wind", "low_humidity"],
    "traffic": ["stagnation", "low_boundary_layer"],
    "industrial": ["stagnation"],
}


class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.g = graph if graph is not None else nx.MultiDiGraph()

    # ---- construction ----------------------------------------------------
    def _add(self, node, ntype, **attrs):
        if node not in self.g:
            self.g.add_node(node, ntype=ntype, **attrs)

    def _rel(self, u, v, relation, **attrs):
        self.g.add_edge(u, v, key=relation, relation=relation, **attrs)

    @classmethod
    def build(cls, chunks=None) -> "KnowledgeGraph":
        kg = cls()
        # AQI stages
        for stage, (lo, hi, band) in GRAP_STAGES.items():
            kg._add(stage, "aqi_stage", band=band, aqi_low=lo, aqi_high=hi)
        # sources -> pollutants, sources -> weather
        for src, polls in SOURCE_POLLUTANTS.items():
            kg._add(src, "source")
            for p in polls:
                kg._add(p, "pollutant")
                kg._rel(src, p, "causes")
        for src, conds in SOURCE_WEATHER.items():
            for c in conds:
                kg._add(c, "weather")
                kg._rel(src, c, "effective_under")
        # interventions -> reduce source, recommended_for stage
        for act, feats in INTERVENTION_FEATURE_MAP.items():
            meta = ACTION_CATALOGUE.get(act, {})
            kg._add(act, "intervention", label=meta.get("label", act),
                    feasibility=meta.get("feasibility", 0.5),
                    cost=meta.get("cost", 0.5),
                    time_to_effect_h=meta.get("time_to_effect_h", 24),
                    features=list(feats.keys()))
            src = ACTION_TARGET_SOURCE.get(act)
            if src:
                kg._add(src, "source")
                kg._rel(act, src, "reduces")
            for stage in INTERVENTION_STAGES.get(act, []):
                kg._rel(act, stage, "recommended_for")
        # policies (from ingested chunks' metadata)
        if chunks:
            kg._link_policies(chunks)
        return kg

    def _link_policies(self, chunks):
        docs = {}
        for c in chunks:
            m = c.metadata
            docs.setdefault(c.doc_id, m)
        for doc_id, m in docs.items():
            self._add(doc_id, "policy", title=m.get("title", ""),
                      authority=m.get("authority", ""), doc_type=m.get("document_type", ""),
                      citation=m.get("citation", ""))
            stage = m.get("aqi_stage")
            if stage in self.g and self.g.nodes[stage].get("ntype") == "aqi_stage":
                self._rel(doc_id, stage, "defines")
            # tag-based links to interventions / sources
            tags = m.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            for t in tags:
                for act in INTERVENTION_FEATURE_MAP:
                    if t in act or act.split("_")[0] in t:
                        self._rel(doc_id, act, "addresses")
                for src in SOURCE_POLLUTANTS:
                    if src.split("_")[0] in t:
                        self._rel(doc_id, src, "addresses")

    # ---- persistence -----------------------------------------------------
    def save(self, path=None):
        path = path or CONFIG.kg_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.g, f)
        return path

    @classmethod
    def load(cls, path=None) -> "KnowledgeGraph":
        path = path or CONFIG.kg_path
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    # ---- traversal -------------------------------------------------------
    def out(self, node, relation=None):
        if node not in self.g:
            return []
        res = []
        for _, v, k, data in self.g.out_edges(node, keys=True, data=True):
            if relation is None or data.get("relation") == relation:
                res.append((v, data.get("relation")))
        return res

    def incoming(self, node, relation=None):
        if node not in self.g:
            return []
        res = []
        for u, _, k, data in self.g.in_edges(node, keys=True, data=True):
            if relation is None or data.get("relation") == relation:
                res.append((u, data.get("relation")))
        return res

    def interventions_for_source(self, source: str, stage: str | None = None) -> list[str]:
        acts = [u for u, _ in self.incoming(source, "reduces")]
        if stage:
            acts = [a for a in acts if any(s == stage for s, _ in self.out(a, "recommended_for"))]
        return acts

    _STAGE_ORDER = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}

    def policies_for(self, stage=None, sources=None, pollutants=None) -> list[dict]:
        """Return policy nodes connected to the given stage/sources/pollutants.

        6.2 — the EXACT stage match dominates (weight 3.0) so a wrong-stage doc can
        never outrank the right one. GRAP is cumulative, so lower-stage docs get a
        modest relevance bump; higher-stage docs a small one. Tag-based source/
        intervention 'addresses' hits are SECONDARY and CAPPED, so accumulating
        tags can't overwhelm the stage signal.
        """
        stage_hits: dict[str, float] = {}
        tag_hits: dict[str, float] = {}
        qrank = self._STAGE_ORDER.get(stage)

        # 1. stage relevance (dominant), stage-aware
        for s, r in self._STAGE_ORDER.items():
            if s not in self.g:
                continue
            for u, _ in self.incoming(s, "defines"):
                if stage is None:
                    w = 0.5
                elif s == stage:
                    w = 3.0                       # exact stage
                elif qrank and r < qrank:
                    w = 0.8                       # cumulative lower stage (still applies)
                else:
                    w = 0.3                       # higher stage (context)
                stage_hits[u] = max(stage_hits.get(u, 0.0), w)

        # 2. source / intervention 'addresses' — secondary, CAPPED at 1.0/doc
        def add_tag(doc_id, w):
            tag_hits[doc_id] = min(tag_hits.get(doc_id, 0.0) + w, 1.0)
        for src in (sources or []):
            if src in self.g:
                for u, _ in self.incoming(src, "addresses"):
                    add_tag(u, 0.4)
            for act in self.interventions_for_source(src, stage):
                for u, _ in self.incoming(act, "addresses"):
                    add_tag(u, 0.25)

        hits = {d: stage_hits.get(d, 0.0) + tag_hits.get(d, 0.0)
                for d in set(stage_hits) | set(tag_hits)}
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])
        return [{"doc_id": d, "weight": round(w, 3),
                 "title": self.g.nodes[d].get("title", ""),
                 "authority": self.g.nodes[d].get("authority", "")}
                for d, w in ranked]

    def stats(self) -> dict:
        from collections import Counter
        nt = Counter(d.get("ntype") for _, d in self.g.nodes(data=True))
        et = Counter(d.get("relation") for _, _, d in self.g.edges(data=True))
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "node_types": dict(nt), "relations": dict(et)}


@lru_cache(maxsize=1)
def get_knowledge_graph() -> KnowledgeGraph:
    try:
        return KnowledgeGraph.load()
    except Exception:
        kg = build_graph()
        return kg


def build_graph() -> KnowledgeGraph:
    from advisor.kb.ingest import Ingestor
    try:
        chunks = Ingestor.load_chunks()
    except Exception:
        chunks = Ingestor().ingest()
    kg = KnowledgeGraph.build(chunks)
    kg.save()
    return kg


if __name__ == "__main__":
    kg = build_graph()
    print("KG stats:", kg.stats())
    print("\ninterventions that reduce 'traffic' @ Stage IV:",
          kg.interventions_for_source("traffic", "Stage IV"))
    print("interventions that reduce 'dust' @ Stage III:",
          kg.interventions_for_source("dust", "Stage III"))
    print("\npollutants caused by 'industrial':", [v for v, _ in kg.out("industrial", "causes")])
    print("\npolicies for Stage III + sources[dust,traffic]:")
    for p in kg.policies_for(stage="Stage III", sources=["dust", "traffic"])[:5]:
        print("  ", p)
