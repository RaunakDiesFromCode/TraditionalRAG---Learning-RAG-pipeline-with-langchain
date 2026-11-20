"""Persistent Vector Store Module for TraditionalRAG using FAISS.

Manages indexing, disk persistence, and fast similarity search over document embeddings.
Uses FAISS IndexFlatIP (Inner Product) on L2-normalized vectors so search scores
directly equal cosine similarity (-1.0 to 1.0).

Automatically checks for an existing saved index on disk so the vector store
loads in milliseconds instead of re-embedding documents on every execution.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import pickle
import faiss
import numpy as np

# Robust import fallbacks for running either as 'python -m src.vectorstore' or 'python src/vectorstore.py'
try:
    from src.embedding import EmbeddingPipeline
except ModuleNotFoundError:
    from embedding import EmbeddingPipeline


class FaissVectorStore:
    """Manages dense document vectors and metadata using a persistent FAISS index."""

    def __init__(
        self,
        persist_dir: str = "vector_stores/faiss",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        auto_load: bool = True,
    ):
        """Initialize the vector store and load the existing index if available.

        Args:
            persist_dir: Directory where faiss.index and metadata.pkl are saved.
            embedding_model: Model used for text embeddings.
            chunk_size: Target characters per text chunk.
            chunk_overlap: Overlapping characters between adjacent chunks.
            auto_load: If True, automatically load the existing index from disk if present.
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

        # Lazily instantiated embedding pipeline to avoid redundant model loads
        self._embedding_pipeline: Optional[EmbeddingPipeline] = None

        # Ensure target persistence directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        # Check if a previously built index already exists on disk
        if auto_load and self._index_files_exist():
            self.load()
        else:
            print(f"[INFO] No existing index found at '{self.persist_dir}'. Call build_from_documents() to create one.")

    @property
    def embedding_pipeline(self) -> EmbeddingPipeline:
        """Lazily initialize and return the shared EmbeddingPipeline."""
        if self._embedding_pipeline is None:
            self._embedding_pipeline = EmbeddingPipeline(
                model_name=self.embedding_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                normalize=True,
            )
        return self._embedding_pipeline

    def _index_files_exist(self) -> bool:
        """Check whether both the FAISS index file and metadata pickle exist on disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        return os.path.isfile(faiss_path) and os.path.isfile(meta_path)

    def is_built(self) -> bool:
        """Return True if the index is loaded in memory and holds at least one vector."""
        return self.index is not None and self.index.ntotal > 0 and len(self.metadata) > 0

    def build_from_documents(self, documents: List[Any], force_rebuild: bool = False) -> bool:
        """Build the vector store from raw documents and persist it to disk.

        Like a real RAG system: If an index is already loaded on disk and force_rebuild
        is False, this method skips rebuilding entirely to save compute and time.

        Args:
            documents: List of raw LangChain Document objects.
            force_rebuild: If True, forces re-chunking and re-embedding even if an index exists.

        Returns:
            True if a new index was built; False if skipped because index already exists.
        """
        if self.is_built() and not force_rebuild:
            print(f"[INFO] Vector store already loaded ({self.index.ntotal} vectors). Skipping build. "
                  f"Pass force_rebuild=True if you wish to overwrite.")
            return False

        if not documents:
            print("[WARNING] No documents provided to build the vector store.")
            return False

        print(f"[INFO] Building vector store from {len(documents)} source documents...")

        # 1. Chunk documents while preserving all metadata
        chunks = self.embedding_pipeline.chunk_documents(documents)
        if not chunks:
            print("[WARNING] Chunking yielded 0 chunks. Cannot build vector store.")
            return False

        # 2. Compute normalized vector embeddings
        embeddings = self.embedding_pipeline.embed_chunks(chunks)

        # 3. Preserve full metadata + raw text for every chunk
        metadatas: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            meta = dict(chunk.metadata)
            meta["text"] = chunk.page_content
            meta["chunk_index"] = i
            metadatas.append(meta)

        # 4. Initialize FAISS Inner Product index (on L2-normalized vectors, IP == Cosine Similarity)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

        # 5. Add embeddings and metadata
        self.add_embeddings(embeddings, metadatas)

        # 6. Persist to disk
        self.save()
        print(f"[INFO] Successfully built and saved vector store to '{self.persist_dir}'.")
        return True

    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add embedding vectors and their corresponding metadata to the FAISS index.

        Args:
            embeddings: 2D NumPy array of shape (N, dim) with dtype float32.
            metadatas: List of N metadata dictionaries matching the vectors.
        """
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_f32)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings_f32.shape[0]} vectors to FAISS index. Total: {self.index.ntotal}")

    def save(self):
        """Persist the in-memory FAISS index and metadata to disk."""
        if not self.is_built():
            print("[WARNING] Cannot save: Index is empty or uninitialized.")
            return

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index ({self.index.ntotal} vectors) and metadata to '{self.persist_dir}'.")

    def load(self) -> bool:
        """Load the FAISS index and metadata pickle from disk.

        Returns:
            True if loading succeeded, False otherwise.
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not self._index_files_exist():
            print(f"[WARNING] Missing index files at '{self.persist_dir}'.")
            return False

        try:
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded FAISS vector store from '{self.persist_dir}' ({self.index.ntotal} vectors).")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load index from '{self.persist_dir}': {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for the top_k most similar vectors.

        Args:
            query_embedding: 2D NumPy array of shape (1, dim) with dtype float32.
            top_k: Maximum number of closest vectors to retrieve.

        Returns:
            List of result dictionaries containing rank, score, content, source, page, and metadata.
        """
        if not self.is_built():
            print("[WARNING] Vector store is empty. No documents to search.")
            return []

        # Ensure top_k does not exceed index size
        k = min(top_k, self.index.ntotal)
        query_f32 = np.ascontiguousarray(query_embedding, dtype=np.float32)

        distances, indices = self.index.search(query_f32, k)

        results: List[Dict[str, Any]] = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            content = meta.get("text", "")
            source = meta.get("source_file", meta.get("source", "unknown"))
            page = meta.get("page", 0)

            results.append({
                "rank": i + 1,
                "index": int(idx),
                "score": float(dist),  # With normalized vectors and IndexFlatIP, this is cosine similarity
                "content": content,
                "source": source,
                "page": page,
                "metadata": meta,
            })

        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store with natural language text.

        Args:
            query_text: Natural language query or question.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            Ranked list of result dictionaries with content, scores, and provenance.
        """
        print(f"[INFO] Querying FAISS vector store for: '{query_text}' (top_k={top_k})")
        query_emb = self.embedding_pipeline.embed_query(query_text)
        return self.search(query_emb, top_k=top_k)


# Example usage demonstrating instant loading without rebuilding
if __name__ == "__main__":
    try:
        from src.data_loader import load_all_documents
    except ModuleNotFoundError:
        from data_loader import load_all_documents

    # Initialize store (automatically loads from vector_stores/faiss if already saved)
    store = FaissVectorStore("vector_stores/faiss")

    if not store.is_built():
        print("[DEMO] Index not found on disk. Building from raw documents...")
        docs = load_all_documents("data")
        store.build_from_documents(docs)
    else:
        print("[DEMO] Vector store is ALREADY BUILT! Skipping re-embedding and running query directly.")

    results = store.query("What is Machine Learning?", top_k=3)
    print(f"\n[DEMO] Retrieved {len(results)} chunks:")
    for res in results:
        print(f"  - Rank {res['rank']} | Score: {res['score']:.4f} | Source: {res['source']} (p. {res['page']})")
        print(f"    Preview: {res['content'][:120]}...\n")
