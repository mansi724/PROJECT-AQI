"""
advisor/retrieval/expand.py — PART 5.6: query expansion.

Air-quality policy queries use jargon and acronyms that a short user query rarely
spells out ("severe" ↔ "Stage IV", "traffic" ↔ "NO2 / odd-even / BS-III"). Rule-
based domain expansion appends these related terms so BM25 (lexical) recall rises
without needing an LLM. (LLM-based HyDE is a heavier, key-gated add-on — see the
note in `hybrid.py`.)

The expansion is deterministic and transparent; it only *adds* terms, so it can't
hurt semantic recall (which runs on the original query).
"""
from __future__ import annotations

import re

# term (regex, word-boundary, case-insensitive) -> extra terms to append
EXPANSIONS: dict[str, list[str]] = {
    r"severe|emergency|hazardous": ["Stage III", "Stage IV", "severe", "GRAP"],
    r"very poor": ["Stage II", "Stage III"],
    r"\bpoor\b": ["Stage I", "GRAP"],
    r"traffic|vehicle|vehicular|car": ["NO2", "odd-even", "BS-III", "BS-IV", "diesel", "PUC"],
    r"construction|building|demolition": ["dust", "C&D", "construction", "stone crusher"],
    r"dust|road dust": ["mechanised sweeping", "water sprinkling", "PM10"],
    r"industry|industrial|factory": ["SO2", "brick kiln", "hot-mix", "emissions"],
    r"stubble|crop|paddy|farm": ["biomass", "paddy straw", "bio-decomposer", "fire"],
    r"firecracker|cracker|diwali": ["firecracker", "ban", "DPCC"],
    r"health|advice|advisory|mask|elderly|children|asthma": ["health advisory", "N95", "sensitive groups"],
    r"limit|guideline|standard|safe level": ["WHO", "guideline", "µg/m3", "annual", "24-hour"],
    r"reduce|reduction|target|clean air": ["NCAP", "PM2.5", "reduction", "non-attainment"],
    r"transport|bus|metro": ["public transport", "CNG", "electric", "parking"],
}


def expand_query(query: str, max_extra: int = 12) -> str:
    """Return `query` plus deduplicated domain terms triggered by it."""
    q = query.lower()
    extra: list[str] = []
    seen = set(re.findall(r"[a-z0-9]+", q))
    for pattern, terms in EXPANSIONS.items():
        if re.search(pattern, q):
            for t in terms:
                key = t.lower()
                if key not in seen:
                    extra.append(t); seen.add(key)
    return query + (" " + " ".join(extra[:max_extra]) if extra else "")


if __name__ == "__main__":
    for q in ["what to do when AQI is severe",
              "how to control traffic pollution",
              "WHO safe limit for PM2.5",
              "firecracker rules for Diwali"]:
        print(f"{q!r}\n  -> {expand_query(q)!r}\n")
