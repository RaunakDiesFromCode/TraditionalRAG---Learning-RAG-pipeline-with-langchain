"""TraditionalRAG Package.

A modular, production-ready Retrieval-Augmented Generation (RAG) system:
- Multi-format document loader (PDF, TXT, CSV, Excel, Word, JSON)
- Recursive text chunking and SentenceTransformer embedding pipeline
- Persistent FAISS vector store with instant disk loading
- Context-grounded LLM synthesis with Groq and source citations
"""

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

__all__ = [
    "load_all_documents",
    "EmbeddingPipeline",
    "FaissVectorStore",
    "RAGSearch",
]
