"""Main entry point for TraditionalRAG.

Supports:
- Interactive Q&A chat in terminal: `python main.py`
- Single query execution via CLI: `python main.py --query "What is TCS NQT?"`
- Rebuilding the vector store: `python main.py --rebuild`
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from src.search import RAGSearch


def run_single_query(rag: RAGSearch, query: str, top_k: int = 3):
    """Execute a single query and display formatted answer and citations."""
    print(f"\n🔎 Query: {query}")
    print("=" * 60)
    result = rag.generate_answer(query, top_k=top_k)
    print(f"\n💡 Answer:\n{result['answer']}")

    if result.get("citations"):
        print("\n📚 Sources & Citations:")
        for cit in result["citations"]:
            print(f"  • {cit}")

    print(f"\n⏱️  Response Time: {result['latency_sec']} seconds")
    print("=" * 60)


def run_interactive_mode(rag: RAGSearch, top_k: int = 3):
    """Run an interactive conversational loop in the terminal."""
    print("\n" + "=" * 60)
    print("🤖 Welcome to the Traditional RAG Interactive Console!")
    print("Ask any question based on your indexed documents.")
    print("Type 'exit', 'quit', or press Ctrl+C to terminate.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("❓ Enter your question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye!")
                break

            result = rag.generate_answer(query, top_k=top_k)
            print(f"\n💡 Answer:\n{result['answer']}")

            if result.get("citations"):
                print("\n📚 Sources & Citations:")
                for cit in result["citations"]:
                    print(f"  • {cit}")

            print(f"\n⏱️  Latency: {result['latency_sec']}s\n" + "-" * 60 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Session ended. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="TraditionalRAG: Real-time Retrieval-Augmented Generation with FAISS & Groq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Run a single search query and exit.",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to include in context (default: 3).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the FAISS vector store from data/ instead of loading existing cache.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to raw documents directory (default: 'data').",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default="vector_stores/faiss",
        help="Path to FAISS persistence directory (default: 'vector_stores/faiss').",
    )

    args = parser.parse_args()

    print("[START] Initializing Traditional RAG Engine...")
    rag = RAGSearch(
        persist_dir=args.store_dir,
        data_dir=args.data_dir,
        force_rebuild=args.rebuild,
    )

    if args.query:
        run_single_query(rag, args.query, top_k=args.top_k)
    else:
        run_interactive_mode(rag, top_k=args.top_k)


if __name__ == "__main__":
    main()
