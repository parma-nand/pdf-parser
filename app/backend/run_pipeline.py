"""
Resume Processing Pipeline
===========================
Run:  python run_pipeline.py path/to/resume.pdf
      python run_pipeline.py path/to/resume.pdf --embed
      python run_pipeline.py path/to/resume.pdf --embed --chroma

Steps
-----
1  section_divide          extract text, split into sections
2  normalize               clean encoding, bullets, whitespace
3  information_extraction  parse structured fields
4  tokenizer_chunking      chunk text for embedding
5  schema                  build UI schema + embed schema (Pydantic)
6  embedding               generate embeddings (optional, --embed flag)
"""

import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from section_divide.extractor          import extract_and_divide
from normalize.normalizer              import normalize_sections
from information_extraction.extractor  import extract_information
from tokenizer_chunking.chunker        import tokenize_and_chunk
from schema.resume_schema              import build_ui_schema, build_embed_schema

OUTPUT_DIRS = {
    "section_divide":         BASE_DIR / "section_divide",
    "normalize":              BASE_DIR / "normalize",
    "information_extraction": BASE_DIR / "information_extraction",
    "tokenizer_chunking":     BASE_DIR / "tokenizer_chunking",
    "schema":                 BASE_DIR / "schema",
    "embedding":              BASE_DIR / "embedding",
}


def _clear_output_json():
    for key, folder in OUTPUT_DIRS.items():
        folder.mkdir(parents=True, exist_ok=True)
        if key not in ("schema", "embedding"):
            for f in folder.glob("*.json"):
                f.unlink()


def run(pdf_path: str, do_embed: bool = False, do_chroma: bool = False):
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    stem = pdf_path.stem

    print(f"\n{'='*60}")
    print(f"  Resume Pipeline  ->  {pdf_path.name}")
    print(f"{'='*60}\n")

    _clear_output_json()

    # Step 1
    print("Step 1 | Extracting + Section Divide ...")
    sections = extract_and_divide(pdf_path)
    (OUTPUT_DIRS["section_divide"] / "sections.json").write_text(
        json.dumps(sections, indent=2, ensure_ascii=False))
    print(f"   Saved -> section_divide/sections.json  ({sections['section_count']} sections)\n")

    # Step 2
    print("Step 2 | Normalizing ...")
    normalized = normalize_sections(sections)
    (OUTPUT_DIRS["normalize"] / "normalized.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False))
    print(f"   Saved -> normalize/normalized.json  ({normalized['total_words']} words)\n")

    # Step 3
    print("Step 3 | Extracting Structured Information ...")
    extracted = extract_information(normalized)
    (OUTPUT_DIRS["information_extraction"] / "extracted.json").write_text(
        json.dumps(extracted, indent=2, ensure_ascii=False))
    name = extracted.get("personal_info", {}).get("name", "unknown")
    print(f"   Saved -> information_extraction/extracted.json  (candidate: {name})\n")

    # Step 4
    print("Step 4 | Tokenizing + Chunking ...")
    chunks = tokenize_and_chunk(normalized, extracted, stem)
    (OUTPUT_DIRS["tokenizer_chunking"] / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"   Saved -> tokenizer_chunking/chunks.json  ({len(chunks['chunks'])} chunks)\n")

    # Step 5
    print("Step 5 | Building Structured Schemas ...")
    ui_schema    = build_ui_schema(extracted)
    embed_schema = build_embed_schema(chunks, resume_id=ui_schema.resume_id)
    (OUTPUT_DIRS["schema"] / "ui_schema.json").write_text(
        ui_schema.model_dump_json(indent=2))
    (OUTPUT_DIRS["schema"] / "embed_schema.json").write_text(
        embed_schema.model_dump_json(indent=2))
    print(f"   Saved -> schema/ui_schema.json")
    print(f"   Saved -> schema/embed_schema.json  ({embed_schema.total_chunks} chunks)\n")

    # Step 6 (optional)
    if do_embed:
        print("Step 6 | Generating Embeddings ...")
        try:
            from embedding.embedder import Embedder
            emb          = Embedder(model="local")
            embed_schema = emb.encode(embed_schema)
            emb.save_json(embed_schema, OUTPUT_DIRS["embedding"] / "embeddings.json")
            print(f"   Saved -> embedding/embeddings.json\n")
            if do_chroma:
                print("   Upserting into ChromaDB ...")
                emb.upsert_chroma(embed_schema)
                print()
        except ImportError as e:
            print(f"   Skipped (install sentence-transformers): {e}\n")

    print(f"{'='*60}")
    print(f"  Done. Outputs in:")
    for key, folder in OUTPUT_DIRS.items():
        jsons = list(folder.glob("*.json"))
        if jsons:
            for f in jsons:
                print(f"    {f.relative_to(BASE_DIR)}")
    print(f"{'='*60}\n")

    return ui_schema, embed_schema


# REPLACE with:
if __name__ == "__main__":
    PDF_PATH = r"C:\Users\realm\Desktop\AI_ML_Project\pdf-parser\TestFile\Test FIles\Resume Test File 1.pdf"
    run(PDF_PATH, do_embed=True, do_chroma=False)