import logging
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunks_validos = []
    for i, chunk in enumerate(chunks):
        texto = chunk.page_content.strip()
        if len(texto) < 50:
            continue
        chunk.page_content = texto
        chunk.metadata["chunk_index"] = i
        chunks_validos.append(chunk)
    print(f"  Chunks gerados: {len(chunks_validos)}")
    return chunks_validos