import logging
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

os.makedirs("logs", exist_ok=True)
os.makedirs("data_store", exist_ok=True)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf": "PDF", ".txt": "Texto"}

def _load_pdf(path):
    try:
        pages = PyPDFLoader(path).load()
        return pages
    except Exception as e:
        logger.error(f"Erro PDF '{path}': {e}")
        return []

def _load_txt(path):
    try:
        docs = TextLoader(path, encoding="utf-8").load()
        return docs
    except Exception as e:
        logger.error(f"Erro TXT '{path}': {e}")
        return []

def load_documents(data_dir="data/"):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Pasta '{data_dir}' nao encontrada.")
    all_docs = []
    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            docs = _load_pdf(str(file_path))
        elif ext == ".txt":
            docs = _load_txt(str(file_path))
        else:
            continue
        for doc in docs:
            doc.metadata.setdefault("source", str(file_path))
        all_docs.extend(docs)
    print(f"  Documentos carregados: {len(all_docs)} paginas")
    return all_docs