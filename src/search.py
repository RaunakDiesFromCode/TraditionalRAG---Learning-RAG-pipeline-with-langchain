"""RAG Search and Generation Engine for TraditionalRAG.

Combines FAISS vector retrieval with Groq-hosted LLMs (e.g. Qwen, Llama)
to deliver context-grounded answers with numbered bibliographic citations.
Reuses existing vector stores from disk to execute queries in seconds.
"""

from typing import List, Dict, Any, Optional
import os
import time
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv()

# Robust import fallbacks for running either as 'python -m src.search' or 'python src/search.py'
try:
    from src.vectorstore import FaissVectorStore
    from src.data_loader import load_all_documents
except ModuleNotFoundError:
    from vectorstore import FaissVectorStore
    from data_loader import load_all_documents

# LangChain Groq Chat Model integration
from langchain_groq import ChatGroq


class RAGSearch:
    """End-to-End Retrieval-Augmented Generation Engine."""

    def __init__(
        self,
        persist_dir: str = "vector_stores/faiss",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "qwen/qwen3.8-27b",
        data_dir: str = "data",
        auto_build: bool = True,
        force_rebuild: bool = False,
    ):
        """Initialize the RAG search pipeline.

        Like a real RAG system:
        - If the FAISS vector store exists on disk, it loads instantly without rebuilding.
        - If it does not exist and auto_build is True, it automatically ingests documents
          from data_dir and creates the index.

        Args:
            persist_dir: Path to directory containing faiss.index and metadata.pkl.
            embedding_model: HuggingFace model for sentence embeddings.
            llm_model: Target Groq-hosted LLM model name.
            data_dir: Path to directory with raw documents if building is required.
            auto_build: Automatically build the vector store if not found on disk.
            force_rebuild: Force a clean rebuild even if an index already exists.
        """
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.llm_model = llm_model

        # 1. Initialize Vector Store (auto_load=True loads existing index)
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            auto_load=True,
        )

        # 2. Check if vector store needs to be built or rebuilt
        if force_rebuild or not self.vectorstore.is_built():
            if auto_build:
                print(f"[INFO] Index not ready or rebuild requested. Ingesting documents from '{data_dir}'...")
                docs = load_all_documents(data_dir)
                if docs:
                    self.vectorstore.build_from_documents(docs, force_rebuild=force_rebuild)
                else:
                    print(f"[WARNING] No documents found in '{data_dir}' to build index.")
            else:
                print(f"[WARNING] Vector store at '{persist_dir}' is not built and auto_build is False.")
        else:
            print(f"[INFO] Real RAG mode: Reusing existing vector store from '{persist_dir}'. Rebuild skipped!")

        # 3. Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            print("[WARNING] GROQ_API_KEY is missing or unset in .env file. "
                  "Semantic retrieval will work, but LLM generation will be unavailable.")
            self.llm = None
        else:
            try:
                self.llm = ChatGroq(
                    api_key=groq_api_key,
                    model=llm_model,
                    temperature=0.1,
                    max_tokens=1024,
                )
                print(f"[INFO] Initialized Groq LLM: {llm_model}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize Groq LLM: {e}")
                self.llm = None

    def retrieve(self, query: str, top_k: int = 5, min_score: float = -1.0) -> List[Dict[str, Any]]:
        """Retrieve the top_k most relevant chunks for a user query.

        Args:
            query: The user query string.
            top_k: Maximum number of chunks to retrieve.
            min_score: Minimum cosine similarity threshold (range -1.0 to 1.0).

        Returns:
            List of retrieved chunk dictionaries.
        """
        raw_results = self.vectorstore.query(query, top_k=top_k)
        filtered = [r for r in raw_results if r["score"] >= min_score]
        return filtered

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> Dict[str, Any]:
        """Execute end-to-end RAG: Retrieve matching context and generate a grounded answer with citations.

        Args:
            query: Question or search query.
            top_k: Number of context chunks to retrieve.
            min_score: Score threshold for retrieval.

        Returns:
            Dictionary containing:
                - 'query': Original user question
                - 'answer': Generated answer string
                - 'citations': List of formatted citation strings
                - 'sources': Raw retrieved chunk dictionaries
                - 'latency_sec': End-to-end processing duration
        """
        start_time = time.time()
        results = self.retrieve(query, top_k=top_k, min_score=min_score)

        if not results:
            return {
                "query": query,
                "answer": "No relevant documents found in the vector store matching your query.",
                "citations": [],
                "sources": [],
                "latency_sec": round(time.time() - start_time, 3),
            }

        # Build context string with explicit provenance separators
        context_blocks = []
        for i, r in enumerate(results):
            src_tag = f"[{i+1}] {r['source']} (Page {r['page']}, Similarity: {r['score']:.3f})"
            context_blocks.append(f"--- Context {src_tag} ---\n{r['content']}")

        context = "\n\n".join(context_blocks)

        # Check LLM availability
        if self.llm is None:
            answer = ("GROQ_API_KEY is not configured. Context was retrieved successfully, "
                      "but answer synthesis requires a valid GROQ_API_KEY in your .env file.")
        else:
            # Format prompt safely without risky .format() calls on context text
            prompt = (
                "You are an expert, precise assistant. Answer the user's question based ONLY on the "
                "provided context below. If the context does not contain enough information to answer, "
                "state clearly that the answer is not available in the provided documents. "
                "Do not hallucinate facts.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            try:
                response = self.llm.invoke([prompt])
                answer = response.content
            except Exception as e:
                answer = f"[ERROR during LLM generation]: {e}"

        # Generate numbered citations
        citations = [
            f"[{i+1}] {r['source']} (Page {r['page']}, Cosine Similarity: {r['score']:.4f})"
            for i, r in enumerate(results)
        ]

        latency = round(time.time() - start_time, 3)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "sources": results,
            "latency_sec": latency,
        }

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """High-level helper returning a single formatted string containing the answer and citations.

        Args:
            query: The user query string.
            top_k: Number of retrieved chunks to use.

        Returns:
            A formatted answer string with appended citations and latency.
        """
        res = self.generate_answer(query, top_k=top_k)
        output = f"{res['answer']}\n\n"
        if res["citations"]:
            output += "Sources & Citations:\n" + "\n".join(res["citations"]) + "\n"
        output += f"(Response generated in {res['latency_sec']}s)"
        return output


# Example usage demonstration
if __name__ == "__main__":
    # Create RAG search engine (uses existing vector_stores/faiss without rebuilding)
    rag = RAGSearch()

    test_query = "What is Machine Learning Basics?"
    print(f"\n[DEMO] Query: {test_query}\n")

    result = rag.generate_answer(test_query, top_k=3)
    print("Answer:\n", result["answer"])
    print("\nCitations:")
    for cit in result["citations"]:
        print(" ", cit)
    print(f"\nLatency: {result['latency_sec']}s")
