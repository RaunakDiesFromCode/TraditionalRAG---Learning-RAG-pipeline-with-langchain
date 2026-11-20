"""Embedding and Text Chunking Pipeline for TraditionalRAG.

Splits LangChain Document objects into overlapping chunks and generates dense
vector representations using HuggingFace SentenceTransformer models (default: all-MiniLM-L6-v2).
Embeddings are L2-normalized so dot products equal cosine similarities.
"""

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Robust import fallback for running either as 'python -m src.embedding' or 'python src/embedding.py'
try:
    from src.data_loader import load_all_documents
except ModuleNotFoundError:
    from data_loader import load_all_documents


class EmbeddingPipeline:
    """Manages document chunking and vector embedding generation."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        normalize: bool = True,
    ):
        """Initialize the chunker and SentenceTransformer embedding model.

        Args:
            model_name: HuggingFace model identifier for sentence embeddings.
            chunk_size: Target character length per chunk.
            chunk_overlap: Number of overlapping characters between adjacent chunks.
            normalize: Whether to L2-normalize vectors (recommended for cosine similarity).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize = normalize
        self.model_name = model_name

        print(f"[INFO] Loading SentenceTransformer embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_embedding_dimension()
        print(f"[INFO] Model loaded successfully. Dimension: {self.embedding_dim}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Split a list of LangChain Document objects into overlapping chunks.

        Preserves original document metadata (source_file, page, etc.) and tags
        each chunk with its chunk_id and character length.

        Args:
            documents: List of input Document objects.

        Returns:
            List of chunked Document objects.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        # Annotate chunk index and content length into metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"
            chunk.metadata["char_count"] = len(chunk.page_content)

        print(f"[INFO] Split {len(documents)} source documents into {len(chunks)} text chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Generate dense vector embeddings for a list of document chunks.

        Args:
            chunks: List of chunked LangChain Document objects.

        Returns:
            2D NumPy array of shape (num_chunks, embedding_dim) as float32.
        """
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        embeddings_arr = np.array(embeddings, dtype=np.float32)
        print(f"[INFO] Embeddings generated successfully. Shape: {embeddings_arr.shape}")
        return embeddings_arr

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding vector for a single search query string.

        Args:
            query: User search query or question.

        Returns:
            2D NumPy array of shape (1, embedding_dim) as float32.
        """
        query_emb = self.model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.array(query_emb, dtype=np.float32)


# Self-test demo when executed directly
if __name__ == "__main__":
    docs = load_all_documents("data")
    if docs:
        emb_pipe = EmbeddingPipeline()
        sample_chunks = emb_pipe.chunk_documents(docs[:3])
        sample_embeddings = emb_pipe.embed_chunks(sample_chunks)
        print(f"[DEMO] Generated {len(sample_embeddings)} chunk embeddings.")
        q_emb = emb_pipe.embed_query("What is the TCS NQT exam pattern?")
        print(f"[DEMO] Query embedding shape: {q_emb.shape}")
    else:
        print("[DEMO] No documents found in 'data/' to test embedding.")
