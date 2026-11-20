"""Data Loader Module for TraditionalRAG.

Discovers and loads documents from the local filesystem across multiple formats:
PDF, plain text (TXT), CSV, Excel (XLSX), Word (DOCX), and JSON.
Standardizes every document into a LangChain Document with rich provenance metadata.
"""

from pathlib import Path
from typing import List, Any
import os

# Standard LangChain document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader

# Optional loaders with graceful fallback if optional dependencies are not installed
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Docx2txtLoader = None

try:
    from langchain_community.document_loaders.excel import UnstructuredExcelLoader
except ImportError:
    UnstructuredExcelLoader = None


def load_all_documents(data_dir: str = "data") -> List[Any]:
    """Load all supported documents from the target directory recursively.

    Supported file types:
    - PDF (.pdf) via PyPDFLoader (per-page extraction)
    - Plain text (.txt) via TextLoader (UTF-8)
    - CSV (.csv) via CSVLoader
    - Excel (.xlsx) via UnstructuredExcelLoader (if unstructured is installed)
    - Word (.docx) via Docx2txtLoader (if docx2txt is installed)
    - JSON (.json) via JSONLoader

    Args:
        data_dir: Path to directory containing raw documents (relative or absolute).

    Returns:
        List of LangChain Document objects with unified metadata attributes.
    """
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Scanning for documents in: {data_path}")

    if not data_path.exists():
        print(f"[WARNING] Data directory does not exist: {data_path}")
        return []

    documents: List[Any] = []

    # 1. PDF Files (.pdf)
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            # Standardize metadata across all pages
            for doc in loaded:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
                doc.metadata["file_path"] = str(pdf_file)
            print(f"  ✓ Loaded {len(loaded)} pages from: {pdf_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ [ERROR] Failed to load PDF '{pdf_file.name}': {e}")

    # 2. Text Files (.txt)
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"[INFO] Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_file"] = txt_file.name
                doc.metadata["file_type"] = "txt"
                doc.metadata["file_path"] = str(txt_file)
            print(f"  ✓ Loaded text file: {txt_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"  ✗ [ERROR] Failed to load TXT '{txt_file.name}': {e}")

    # 3. CSV Files (.csv)
    csv_files = list(data_path.glob("**/*.csv"))
    if csv_files:
        print(f"[INFO] Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            try:
                loader = CSVLoader(str(csv_file))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source_file"] = csv_file.name
                    doc.metadata["file_type"] = "csv"
                    doc.metadata["file_path"] = str(csv_file)
                print(f"  ✓ Loaded {len(loaded)} records from CSV: {csv_file.name}")
                documents.extend(loaded)
            except Exception as e:
                print(f"  ✗ [ERROR] Failed to load CSV '{csv_file.name}': {e}")

    # 4. Excel Files (.xlsx)
    xlsx_files = list(data_path.glob("**/*.xlsx"))
    if xlsx_files:
        print(f"[INFO] Found {len(xlsx_files)} Excel files")
        if UnstructuredExcelLoader is None:
            print("  ! [WARNING] UnstructuredExcelLoader not available. Skipping .xlsx files.")
        else:
            for xlsx_file in xlsx_files:
                try:
                    loader = UnstructuredExcelLoader(str(xlsx_file))
                    loaded = loader.load()
                    for doc in loaded:
                        doc.metadata["source_file"] = xlsx_file.name
                        doc.metadata["file_type"] = "xlsx"
                    print(f"  ✓ Loaded Excel file: {xlsx_file.name}")
                    documents.extend(loaded)
                except Exception as e:
                    print(f"  ✗ [ERROR] Failed to load Excel '{xlsx_file.name}': {e}")

    # 5. Word Documents (.docx)
    docx_files = list(data_path.glob("**/*.docx"))
    if docx_files:
        print(f"[INFO] Found {len(docx_files)} Word files")
        if Docx2txtLoader is None:
            print("  ! [WARNING] Docx2txtLoader not available. Skipping .docx files.")
        else:
            for docx_file in docx_files:
                try:
                    loader = Docx2txtLoader(str(docx_file))
                    loaded = loader.load()
                    for doc in loaded:
                        doc.metadata["source_file"] = docx_file.name
                        doc.metadata["file_type"] = "docx"
                    print(f"  ✓ Loaded Word file: {docx_file.name}")
                    documents.extend(loaded)
                except Exception as e:
                    print(f"  ✗ [ERROR] Failed to load Word '{docx_file.name}': {e}")

    # 6. JSON Files (.json)
    json_files = list(data_path.glob("**/*.json"))
    if json_files:
        print(f"[INFO] Found {len(json_files)} JSON files")
        for json_file in json_files:
            try:
                loader = JSONLoader(str(json_file), jq_schema=".", text_content=False)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source_file"] = json_file.name
                    doc.metadata["file_type"] = "json"
                print(f"  ✓ Loaded JSON file: {json_file.name}")
                documents.extend(loaded)
            except Exception as e:
                print(f"  ✗ [ERROR] Failed to load JSON '{json_file.name}': {e}")

    print(f"[INFO] Total documents loaded across all formats: {len(documents)}")
    return documents


# Quick self-test when executed directly
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"\n[DEMO] Total loaded: {len(docs)}")
    if docs:
        print(f"[DEMO] Sample document metadata: {docs[0].metadata}")
        print(f"[DEMO] Sample document content preview: {docs[0].page_content[:150]}...")
