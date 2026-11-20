# TraditionalRAG: Modular Retrieval-Augmented Generation Pipeline

A production-grade, modular **Retrieval-Augmented Generation (RAG)** system implemented in Python using **LangChain**, **FAISS**, **ChromaDB**, **SentenceTransformers**, and **Groq Cloud LLMs**.

Designed with clean separation of concerns, persistent vector indexing (no redundant rebuilding on every execution), multi-format document ingestion, and verifiable bibliographic source citations.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Key Features](#key-features)
- [Project Directory Structure](#project-directory-structure)
- [Prerequisites](#prerequisites)
- [Installation & Environment Setup](#installation--environment-setup)
- [How to Run](#how-to-run)
  - [1. Interactive CLI Console](#1-interactive-cli-console)
  - [2. Single Query via CLI](#2-single-query-via-cli)
  - [3. Rebuilding the Vector Store](#3-rebuilding-the-vector-store)
  - [4. Running Core Modules Directly](#4-running-core-modules-directly)
  - [5. Interactive Jupyter Notebooks](#5-interactive-jupyter-notebooks)
- [Core Components](#core-components)
  - [Data Loader (`src/data_loader.py`)](#data-loader-srcdata_loaderpy)
  - [Embedding Pipeline (`src/embedding.py`)](#embedding-pipeline-srcembeddingpy)
  - [Persistent Vector Store (`src/vectorstore.py`)](#persistent-vector-store-srcvectorstorepy)
  - [RAG Search Engine (`src/search.py`)](#rag-search-engine-srcsearchpy)
- [Vector Store Persistence](#vector-store-persistence)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Overview & Architecture

TraditionalRAG transforms unorganized documents stored in the `data/` folder into context-grounded answers. Instead of rebuilding the vector index every time a query is executed, the system automatically checks for a pre-built index in `vector_stores/faiss` and loads it in milliseconds.

```
 ┌────────────────────────────────────────────────────────┐
 │                      RAW DATA                          │
 │  data/pdf/*.pdf, data/text_files/*.txt, CSV, etc.      │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │             1. DATA LOADER (src/data_loader.py)        │
 │  Multi-format loaders + standard metadata enrichment   │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │           2. CHUNKING & EMBEDDING (src/embedding.py)   │
 │  RecursiveCharacterTextSplitter (1000 char / 200 ovlp) │
 │  SentenceTransformer (all-MiniLM-L6-v2, 384-dim, norm) │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │         3. PERSISTENT VECTOR STORE (vector_stores/)    │
 │  FAISS IndexFlatIP (vector_stores/faiss)               │
 │  ChromaDB SQLite + HNSW (vector_stores/chroma)         │
 │  * Loads from disk in < 0.05s on subsequent runs *     │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │              4. SEARCH ENGINE (src/search.py)          │
 │  Semantic retrieval + Cosine similarity ranking        │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │       5. GENERATION WITH GROQ LLM (ChatGroq)           │
 │  Context-grounded prompt (Qwen 3.8 / Llama 3)          │
 │  Numbered citations: [1] source.pdf (Page X, Score Y)  │
 └────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Multi-Format Ingestion**: Ingests PDF (`PyPDFLoader`), TXT (`TextLoader`), CSV (`CSVLoader`), Excel (`.xlsx`), Word (`.docx`), and JSON.
- **Smart Vector Persistence**: Separates raw data (`data/`) from generated indexes (`vector_stores/`). Checks disk cache first and skips rebuilding unless explicitly commanded (`--rebuild`).
- **Normalized Cosine Similarity**: Uses L2-normalized embeddings with FAISS `IndexFlatIP` (Inner Product) so search scores directly reflect true cosine similarity (`-1.0` to `1.0`).
- **Full Provenance Preservation**: Chunks retain original metadata (`source_file`, `page`, `file_type`, `file_path`), allowing the LLM to output verifiable citations.
- **Dual-Mode Import Resolution**: All `src/` modules can be run either as packages (`python -m src.vectorstore`) or directly as standalone scripts (`python src/vectorstore.py`).
- **Flexible Interfaces**: Choose between interactive terminal chat, single-query CLI, standalone modules, or interactive Jupyter notebooks.

---

## Project Directory Structure

```text
TraditionalRAG/
├── .env.example               # Example environment variable template
├── .gitignore                 # Ignores .venv, .env, and vector_stores/
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Flat dependency list
├── main.py                    # Unified CLI & interactive console entry point
│
├── data/                      # USER DATA DIRECTORY (Git-ignored: add your own files here)
│   ├── pdf/                   # Drop your PDF files here
│   └── text_files/            # Drop your TXT, CSV, DOCX, XLSX, or JSON files here
│
├── vector_stores/             # COMPILED VECTOR INDEXES (Git-ignored)
│   ├── faiss/                 # FAISS binary index (faiss.index + metadata.pkl)
│   └── chroma/                # ChromaDB SQLite database & HNSW segments
│
├── src/                       # MODULAR APPLICATION CODE
│   ├── __init__.py            # Package root & public exports
│   ├── data_loader.py         # Multi-format document ingestion
│   ├── embedding.py           # Recursive chunking & SentenceTransformer embeddings
│   ├── vectorstore.py         # Persistent FAISS vector store manager
│   └── search.py              # Retrieval & Groq LLM answer generation
│
└── notebook/                  # INTERACTIVE JUPYTER NOTEBOOKS
    ├── document.ipynb         # Document & directory loading walkthrough
    └── pdf_loader.ipynb       # End-to-end PDF → ChromaDB → Groq RAG pipeline
```

---

## Prerequisites

- **Python**: Version `3.10` or higher (tested with `3.11` and `3.14`).
- **Groq API Key**: Free API key from [Groq Console](https://console.groq.com) for fast LLM inference.

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone <repository_url>
cd TraditionalRAG
```

### 2. Create and Activate a Virtual Environment

Using standard Python:
```bash
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# or: .venv\Scripts\activate  # Windows
```

Or using `uv` (faster):
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 5. Add Your Own Data
TraditionalRAG is **completely data-agnostic** and operates on your own documents. The `data/` directory is intentionally included in `.gitignore` so your personal notes, proprietary PDFs, and sensitive records are never accidentally committed to version control.

1. Ensure the `data/` directory exists (or create subfolders as desired):
   ```bash
   mkdir -p data/pdf data/text_files
   ```
2. Copy your own documents into `data/` (supports `.pdf`, `.txt`, `.csv`, `.docx`, `.xlsx`, and `.json`).
3. Build the initial FAISS vector index:
   ```bash
   python main.py --rebuild
   ```
   *(On subsequent runs, the index loads instantly from disk without re-embedding!)*

---

## How to Run

### 1. Interactive CLI Console
Run the interactive terminal console where you can ask questions continuously:
```bash
python main.py
```
Example interaction:
```text
🤖 Welcome to the Traditional RAG Interactive Console!
Ask any question based on your indexed documents.
Type 'exit', 'quit', or press Ctrl+C to terminate.
============================================================

❓ Enter your question: What is Python Programming?

💡 Answer:
Python is a high-level, interpreted programming language created by Guido van Rossum
and first released in 1991. It is known for its simplicity, readability, extensive
standard library, cross-platform compatibility, and strong community support.

📚 Sources & Citations:
  • [1] python_intro.txt (Page 0, Cosine Similarity: 0.8384)

⏱️ Latency: 0.42s
```

### 2. Single Query via CLI
Pass a single question directly via the `--query` flag:
```bash
python main.py --query "What is the TCS NQT exam pattern?" --top-k 3
```

### 3. Rebuilding the Vector Store
To refresh the FAISS vector store after adding new documents to `data/`:
```bash
python main.py --rebuild --query "What are the 5 sections in TCS NQT?"
```

### 4. Running Core Modules Directly
Each module in `src/` can be run independently for debugging or verification:

```bash
# Test document loading from data/
python -m src.data_loader

# Test text chunking and embedding generation
python -m src.embedding

# Test FAISS index loading and vector query
python -m src.vectorstore

# Test end-to-end RAG search and answer synthesis
python -m src.search
```

*(You can also run them directly as scripts, e.g. `python src/search.py`, thanks to built-in import fallbacks.)*

### 5. Interactive Jupyter Notebooks
Launch Jupyter or open the notebooks directly in VS Code / Cursor:

- [notebook/document.ipynb](file:///Users/raunakmanna/Documents/Programming/RAG/TraditionalRAG/notebook/document.ipynb): Covers document structure, manual `Document` instantiation, `TextLoader`, and `DirectoryLoader`.
- [notebook/pdf_loader.ipynb](file:///Users/raunakmanna/Documents/Programming/RAG/TraditionalRAG/notebook/pdf_loader.ipynb): Complete end-to-end RAG tutorial using `PyPDFLoader`, `ChromaDB`, `RAGRetriever`, and `ChatGroq`, with advanced features including streaming, citations, conversational history, and automated summaries.

---

## Core Components

### Data Loader (`src/data_loader.py`)
- Traverses the `data/` directory recursively.
- Auto-detects and loads `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, and `.json`.
- Enriches every page/document with `source_file`, `file_type`, and `file_path` metadata.

### Embedding Pipeline (`src/embedding.py`)
- Employs `RecursiveCharacterTextSplitter` with configurable chunk size (default `1000`) and overlap (default `200`).
- Generates 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
- Embeddings are L2-normalized to ensure inner product calculations reflect exact cosine similarities.

### Persistent Vector Store (`src/vectorstore.py`)
- Manages an in-memory `faiss.IndexFlatIP` persisted to `vector_stores/faiss/faiss.index`.
- Stores chunk content and metadata in `vector_stores/faiss/metadata.pkl`.
- **Real RAG Behavior**: Automatically detects if the index exists on disk. If found, it loads instantly and skips re-embedding.

### RAG Search Engine (`src/search.py`)
- Coordinates the vector store with the Groq chat model (`qwen/qwen3.8-27b`).
- Retrieves top matching chunks, filters by score, and formats context safely.
- Appends numbered citations linking answers back to the original source file and page number.

---

## Vector Store Persistence

Vector stores are isolated in the `vector_stores/` folder to separate raw source documents from compiled binary indexes:

| Folder | Technology | Files | Description |
|---|---|---|---|
| `vector_stores/faiss/` | FAISS | `faiss.index`, `metadata.pkl` | Fast flat inner product index for CLI & `src/` modules. |
| `vector_stores/chroma/` | ChromaDB | `chroma.sqlite3`, segment files | SQLite + HNSW vector index used in `notebook/pdf_loader.ipynb`. |

Both directories are included in [.gitignore](file:///.gitignore) to keep version control lightweight and prevent committing binary index files.

---

## Troubleshooting & FAQs

#### Q: `ModuleNotFoundError: No module named 'src'`
**A**: Ensure you are running commands from the project root (`TraditionalRAG/`). Use `python -m src.module_name` or run `python main.py`. Built-in `sys.path` fallbacks have also been added to all `src/` scripts to support direct execution.

#### Q: `Request to POST /openai/v1/chat/completions on api.groq.com not allowed`
**A**: Check that your `GROQ_API_KEY` in `.env` is valid and active. Also ensure your terminal environment has network access to `api.groq.com`.

#### Q: Why did similarity scores appear very low in ChromaDB?
**A**: ChromaDB uses squared $L_2$ distance by default (`space: "l2"`), where distance $\approx 2(1 - \cos(\theta))$. For normalized vectors, a cosine similarity of $0.5$ results in an $L_2$ distance of $\approx 1.0$. The FAISS store in `src/` uses `IndexFlatIP` on normalized vectors, directly returning standard cosine similarity scores between `0.0` and `1.0`.

#### Q: Where are the sample documents? Why is `data/` empty or ignored?
**A**: TraditionalRAG is designed to be completely data-agnostic. The `data/` directory is deliberately gitignored so your personal notes, company PDFs, and proprietary data are never pushed to a remote repository. Simply drop your own `.pdf`, `.txt`, `.csv`, `.docx`, `.xlsx`, or `.json` files into `data/` (or subdirectories like `data/pdf/`), then run `python main.py --rebuild` to index them.
