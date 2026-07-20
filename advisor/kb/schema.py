"""
advisor/kb/schema.py — PART 2: knowledge-base metadata schema.

One typed, validated schema every document and chunk carries, so retrieval can
filter on it (city, pollutant, AQI stage, authority, date...) and the LLM can
cite it. Controlled vocabularies keep the metadata queryable rather than free-text.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict

# ---- controlled vocabularies ---------------------------------------------
SOURCE_CATEGORIES = {
    "regulation", "action_plan", "guideline", "notification",
    "health_advisory", "research", "standard",
}
DOCUMENT_TYPES = {
    "grap", "caqm_order", "cpcb_guideline", "dpcc_notification",
    "who_guideline", "ncap", "govt_notification", "research_paper", "standard",
}
AQI_STAGES = {"none", "Stage I", "Stage II", "Stage III", "Stage IV", "all"}
POLLUTANTS = {"PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Pb", "dust", "all"}
AUTHORITIES = {
    "CAQM", "CPCB", "DPCC", "MoEFCC", "WHO", "Delhi Government",
    "Supreme Court", "NGT", "Research", "Other",
}


def _norm_list(v, allowed=None, default=None):
    if v is None:
        return list(default) if default else []
    if isinstance(v, str):
        v = [s.strip() for s in v.split(",") if s.strip()]
    v = [str(x).strip() for x in v]
    if allowed:
        v = [x for x in v if x in allowed] or ([default] if default else [])
    return v


@dataclass
class DocumentMetadata:
    title: str
    authority: str = "Other"
    city: str = "Delhi"
    pollutant: list = field(default_factory=lambda: ["all"])
    aqi_stage: str = "all"
    effective_date: str = ""            # ISO yyyy-mm-dd, or "" if not dated
    document_type: str = "govt_notification"
    source_category: str = "guideline"
    tags: list = field(default_factory=list)
    source_url: str = ""
    doc_id: str = ""

    def __post_init__(self):
        self.title = str(self.title)
        self.effective_date = str(self.effective_date) if self.effective_date else ""
        self.pollutant = _norm_list(self.pollutant, POLLUTANTS, "all")
        self.tags = _norm_list(self.tags)
        if self.authority not in AUTHORITIES:
            self.authority = "Other"
        if self.aqi_stage not in AQI_STAGES:
            self.aqi_stage = "all"
        if self.source_category not in SOURCE_CATEGORIES:
            self.source_category = "guideline"
        if self.document_type not in DOCUMENT_TYPES:
            self.document_type = "govt_notification"
        if not self.doc_id:
            self.doc_id = self.make_doc_id(self.title, self.authority)

    @staticmethod
    def make_doc_id(title: str, authority: str) -> str:
        h = hashlib.sha1(f"{authority}::{title}".encode()).hexdigest()[:10]
        return f"doc_{h}"

    def as_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        bits = [self.title, self.authority]
        if self.effective_date:
            bits.append(self.effective_date)
        return " — ".join(b for b in bits if b)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict                       # flattened DocumentMetadata + extras
    citation: str
    section: str = ""

    @staticmethod
    def make_chunk_id(doc_id: str, position: int, text: str) -> str:
        h = hashlib.sha1(text.encode()).hexdigest()[:8]
        return f"{doc_id}::c{position:03d}::{h}"

    def as_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        norm = " ".join(self.text.lower().split())
        return hashlib.sha1(norm.encode()).hexdigest()


if __name__ == "__main__":
    m = DocumentMetadata(title="GRAP Stage III Measures", authority="CAQM",
                         pollutant="PM2.5, PM10", aqi_stage="Stage III",
                         document_type="grap", source_category="action_plan",
                         effective_date="2022-08-05", tags=["construction", "dust"])
    print("doc_id:", m.doc_id, "| citation:", m.citation())
    print("normalised pollutants:", m.pollutant)
    ch = Chunk(Chunk.make_chunk_id(m.doc_id, 0, "hello world"), m.doc_id,
               "Hello World", 0, m.as_dict(), m.citation())
    print("chunk_id:", ch.chunk_id, "| hash:", ch.content_hash()[:12])
