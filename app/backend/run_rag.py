"""
run_rag.py
===========
Entry point for the RAG system.

Usage
-----
# Step 1: run the pipeline first to generate embed_schema.json
    python run_pipeline.py "path/to/resume.pdf"

# Step 2: run RAG (add your key to config/.env first)
    python run_rag.py

# Or with a custom PDF path:
    python run_rag.py --pdf "path/to/resume.pdf"
    python run_rag.py --question "What are the candidate's skills?"
    python run_rag.py --match "We need a Python + AWS + Kafka engineer"
    python run_rag.py --summary
    python run_rag.py --interview "Senior Backend Engineer"
    python run_rag.py --search "machine learning experience"
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from rag.resume_rag import ResumeRAG
from run_pipeline   import run as run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Resume RAG — powered by GPT-4o-mini")
    parser.add_argument("--pdf",       help="Run pipeline on this PDF first, then start RAG")
    parser.add_argument("--question",  help="Ask a question about the resume")
    parser.add_argument("--match",     help="Match resume against a job description string")
    parser.add_argument("--summary",   action="store_true", help="Generate professional summary")
    parser.add_argument("--interview", help="Generate interview questions for a role")
    parser.add_argument("--search",    help="Raw semantic search query")
    parser.add_argument("--rebuild",   action="store_true", default=True,
                        help="Rebuild ChromaDB from embed_schema.json (default: True)")
    args = parser.parse_args()

    # If PDF path given, run pipeline first
    if args.pdf:
        print(f"Running pipeline on: {args.pdf}")
        run_pipeline(args.pdf)

    # Load RAG
    rag = ResumeRAG(k=5, rebuild_store=args.rebuild)
    rag.load(embed_schema_path=BASE_DIR / "schema" / "embed_schema.json")

    # Execute requested operation
    if args.question:
        rag.ask(args.question)

    elif args.match:
        rag.match_skills(args.match)

    elif args.summary:
        rag.generate_summary()

    elif args.interview:
        rag.interview_questions(args.interview)

    elif args.search:
        results = rag.search(args.search)
        print(f"\nSearch results for: '{args.search}'")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] section={r['section']}  score={r['score']}")
            print(f"    {r['text']}")

    else:
        # Interactive mode — loop until user types 'exit'
        print("\n" + "="*60)
        print("  Resume RAG — Interactive Mode")
        print("  Commands: ask / match / summary / interview / search / exit")
        print("="*60)

        while True:
            try:
                cmd = input("\nCommand > ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if cmd in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            elif cmd == "ask":
                q = input("Question > ").strip()
                if q:
                    rag.ask(q)

            elif cmd == "match":
                print("Paste job description (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                if lines:
                    rag.match_skills("\n".join(lines))

            elif cmd == "summary":
                rag.generate_summary()

            elif cmd == "interview":
                role = input("Role > ").strip()
                if role:
                    rag.interview_questions(role)

            elif cmd == "search":
                q = input("Search query > ").strip()
                if q:
                    results = rag.search(q)
                    for i, r in enumerate(results, 1):
                        print(f"\n[{i}] section={r['section']}  score={r['score']}")
                        print(f"    {r['text']}")

            else:
                print("Unknown command. Try: ask / match / summary / interview / search / exit")


if __name__ == "__main__":
    main()