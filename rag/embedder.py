import logging
import os
from sentence_transformers import SentenceTransformer
import chromadb

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

CHROMA_PATH = r"C:\Users\lebro\chroma_db"
COLLECTION_NAME = "jarvis_materiais"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        os.makedirs(CHROMA_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def indexar(self, chunks, batch_size=64):
        if not chunks:
            return 0
        print(f"  Indexando {len(chunks)} chunks...")
        textos, embeddings, ids, metadatas = [], [], [], []
        for i in range(0, len(chunks), batch_size):
            lote = chunks[i:i+batch_size]
            textos_lote = [c.page_content for c in lote]
            embs_lote = self.model.encode(textos_lote, convert_to_numpy=True).tolist()
            for j, (chunk, emb) in enumerate(zip(lote, embs_lote)):
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", 0)
                c_index = chunk.metadata.get("chunk_index", i+j)
                uid = f"doc__{i+j}"
                textos.append(chunk.page_content)
                embeddings.append(emb)
                ids.append(uid)
                metadatas.append({
                    "source": str(source),
                    "page": str(page),
                    "chunk_index": str(c_index),
                })
        self.collection.upsert(
            documents=textos,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        total = self.collection.count()
        print(f"  Indexacao concluida. Total no banco: {total}")
        return len(chunks)

    def contar_documentos(self):
        return self.collection.count()

    def gerar_embedding(self, texto):
        return self.model.encode([texto], convert_to_numpy=True).tolist()[0]