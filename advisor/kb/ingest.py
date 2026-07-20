"""
advisor/kb/ingest.py — PART 3: document processing & ingestion.

Turns heterogeneous source documents into clean, de-duplicated, citation-bearing
chunks with validated metadata, ready for embedding (Part 4) and the knowledge
graph (Part 7).

Supported inputs
  * `data/kb/corpus/*.md`     — curated docs with YAML front-matter metadata
  * `data/kb/raw_docs/*`      — user-dropped PDF / DOCX / HTML / TXT
      - metadata is read from an optional sidecar `<name>.meta.json`, else inferred

Pipeline: parse -> clean -> section-aware chunk -> content-hash de-dup ->
stable chunk IDs -> citation string per chunk. Output: `data/kb/chunks.jsonl`.

    python -m advisor.kb.ingest            # ingest everything
    python -m advisor.kb.ingest --stats    # just report counts
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from advisor.config import CONFIG
from advisor.kb.schema import DocumentMetadata, Chunk

# ---- optional parsers (guarded so missing extras never crash ingestion) ---
try:
    import pypdf
except Exception:
    pypdf = None
try:
    import docx2txt
except Exception:
    docx2txt = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# ==========================================================================
# Parsers  -> return (text, metadata_overrides)
# ==========================================================================
def parse_markdown(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8")
    meta = {}
    body = raw
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    meta.setdefault("title", path.stem.replace("_", " ").title())
    return body, meta


def parse_pdf(path: Path) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf not installed")
    reader = pypdf.PdfReader(str(path))
    pages = [(pg.extract_text() or "") for pg in reader.pages]
    text = "\n\n".join(pages).strip()
    if len(text) < 40:                       # scanned PDF -> OCR fallback
        text = _ocr_pdf(path) or text
    return text


def _ocr_pdf(path: Path) -> str:
    """OCR fallback for scanned PDFs. Requires pytesseract + pdf2image + the
    system tesseract/poppler binaries; returns '' (with a note) if unavailable."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        print(f"  [ocr] skipped for {path.name} (pytesseract/pdf2image not available)")
        return ""
    try:
        return "\n\n".join(pytesseract.image_to_string(img)
                           for img in convert_from_path(str(path)))
    except Exception as e:
        print(f"  [ocr] failed for {path.name}: {e}")
        return ""


def parse_docx(path: Path) -> str:
    if docx2txt is None:
        raise RuntimeError("docx2txt not installed")
    return docx2txt.process(str(path)) or ""


def parse_html(path: Path) -> str:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 not installed")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n")


def parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".doc": parse_docx,
            ".html": parse_html, ".htm": parse_html, ".txt": parse_txt}


# ==========================================================================
# Cleaning + chunking
# ==========================================================================
def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop page-number-only lines
    text = "\n".join(ln for ln in text.split("\n") if not re.fullmatch(r"\s*\d+\s*", ln))
    return text.strip()


def _sections(body: str):
    """Yield (heading, block) using markdown headings as section boundaries."""
    lines = body.split("\n")
    heading, buf = "", []
    for ln in lines:
        if re.match(r"^#{1,6}\s+", ln):
            if buf:
                yield heading, "\n".join(buf).strip()
                buf = []
            heading = re.sub(r"^#{1,6}\s+", "", ln).strip()
        else:
            buf.append(ln)
    if buf:
        yield heading, "\n".join(buf).strip()


def chunk_text(body: str, size: int, overlap: int):
    """Section-aware, sentence-respecting chunks. Yields (section, text)."""
    for heading, block in _sections(body):
        block = block.strip()
        if not block:
            continue
        if len(block) <= size:
            yield heading, block
            continue
        sentences = re.split(r"(?<=[.!?])\s+", block)
        cur = ""
        for s in sentences:
            if len(cur) + len(s) + 1 > size and cur:
                yield heading, cur.strip()
                cur = cur[-overlap:] + " " + s if overlap else s
            else:
                cur = (cur + " " + s).strip()
        if cur.strip():
            yield heading, cur.strip()


# ==========================================================================
# Ingestor
# ==========================================================================
class Ingestor:
    def __init__(self, config=CONFIG):
        self.cfg = config
        config.ensure_dirs()

    def _doc_meta(self, overrides: dict) -> DocumentMetadata:
        allowed = {k: overrides[k] for k in DocumentMetadata.__dataclass_fields__
                   if k in overrides}
        return DocumentMetadata(**allowed)

    def _infer_meta(self, path: Path) -> dict:
        name = path.stem.lower()
        meta = {"title": path.stem.replace("_", " ").title(), "source_url": path.name}
        table = {"grap": ("CAQM", "grap", "action_plan"),
                 "caqm": ("CAQM", "caqm_order", "action_plan"),
                 "cpcb": ("CPCB", "cpcb_guideline", "guideline"),
                 "dpcc": ("DPCC", "dpcc_notification", "notification"),
                 "who": ("WHO", "who_guideline", "guideline"),
                 "ncap": ("MoEFCC", "ncap", "action_plan")}
        for key, (auth, dt, sc) in table.items():
            if key in name:
                meta.update(authority=auth, document_type=dt, source_category=sc)
                break
        return meta

    def _iter_sources(self):
        for p in sorted(self.cfg.kb_corpus_dir.glob("*.md")):
            body, meta = parse_markdown(p)
            yield p, body, meta
        for p in sorted(self.cfg.kb_raw_docs_dir.glob("*")):
            if p.suffix.lower() not in _PARSERS:
                continue
            sidecar = p.with_suffix(p.suffix + ".meta.json")
            meta = json.loads(sidecar.read_text()) if sidecar.exists() else self._infer_meta(p)
            try:
                text = _PARSERS[p.suffix.lower()](p)
            except Exception as e:
                print(f"  [skip] {p.name}: {e}")
                continue
            yield p, text, meta

    def ingest(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen_hashes: set[str] = set()
        n_docs = 0
        for path, body, meta_over in self._iter_sources():
            meta = self._doc_meta(meta_over)
            body = clean_text(body)
            n_docs += 1
            pos = 0
            for section, text in chunk_text(body, self.cfg.chunk_size, self.cfg.chunk_overlap):
                if len(text) < 40:
                    continue
                cid = Chunk.make_chunk_id(meta.doc_id, pos, text)
                cmeta = meta.as_dict()
                cmeta["section"] = section
                cmeta["source_file"] = path.name
                cmeta["char_start"] = body.find(text)   # 5.9: deep-link anchor into the source
                ch = Chunk(chunk_id=cid, doc_id=meta.doc_id, text=text, position=pos,
                           metadata=cmeta, citation=meta.citation(), section=section)
                h = ch.content_hash()
                if h in seen_hashes:                 # cross-document de-dup
                    continue
                seen_hashes.add(h)
                chunks.append(ch)
                pos += 1
        print(f"ingested {n_docs} documents -> {len(chunks)} unique chunks")
        return chunks

    def write(self, chunks: list[Chunk], path: Path | None = None) -> Path:
        path = path or (self.cfg.kb_dir / "chunks.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch.as_dict(), ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def load_chunks(path: Path | None = None) -> list[Chunk]:
        path = path or (CONFIG.kb_dir / "chunks.jsonl")
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out.append(Chunk(**d))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    ing = Ingestor()
    chunks = ing.ingest()
    if not args.stats:
        p = ing.write(chunks)
        print("wrote", p)
    # quick report
    from collections import Counter
    by_auth = Counter(c.metadata.get("authority") for c in chunks)
    by_stage = Counter(c.metadata.get("aqi_stage") for c in chunks)
    print("by authority:", dict(by_auth))
    print("by aqi_stage:", dict(by_stage))
    if chunks:
        print("\nsample chunk:")
        c = chunks[0]
        print(" id:", c.chunk_id, "| section:", c.section, "| cite:", c.citation)
        print(" text:", c.text[:160], "...")


if __name__ == "__main__":
    main()
