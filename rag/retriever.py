"""
rag/retriever.py
----------------
Busca os chunks mais relevantes para uma query usando similaridade vetorial.
"""

import logging
import os
import sys
import warnings

# Suprime mensagens de carregamento do HuggingFace e sentence_transformers
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Silencia loggers externos
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
CHROMA_PATH     = r"C:\Users\lebro\chroma_db"
COLLECTION_NAME = "jarvis_materiais"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classe Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Busca semantica nos chunks indexados no ChromaDB.
    """

    def __init__(self):
        import io
        stderr_backup = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        finally:
            sys.stderr = stderr_backup  # restaura stderr
        
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                f"Colecao '{COLLECTION_NAME}' nao encontrada. "
                "Execute setup.py primeiro."
            )
        total = self.collection.count()
        if total == 0:
            raise RuntimeError("Indice RAG vazio. Execute setup.py.")
        logger.info(f"Retriever inicializado | {total} chunks")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Busca os top_k chunks mais relevantes para a query.

        Parametros
        ----------
        query : str - pergunta de busca
        top_k : int - numero de resultados (padrao 3)

        Retorna
        -------
        list[dict] com: texto, fonte, pagina, score
        """
        if not query.strip():
            return []

        top_k = max(1, min(top_k, 10))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        resultados_raw = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documentos = resultados_raw.get("documents",  [[]])[0]
        metadatas  = resultados_raw.get("metadatas",  [[]])[0]
        distancias = resultados_raw.get("distances",  [[]])[0]

        resultados = []
        for texto, meta, distancia in zip(documentos, metadatas, distancias):
            score = round(1 - (distancia / 2), 4)
            resultados.append({
                "texto":  texto,
                "fonte":  meta.get("source", "Desconhecido"),
                "pagina": meta.get("page",   ""),
                "score":  score,
            })

        logger.info(f"retrieve() | query='{query[:60]}' | resultados={len(resultados)}")
        return resultados

    def contar_documentos(self) -> int:
        """Retorna o numero total de chunks indexados."""
        return self.collection.count()

    def retrieve_formatado(self, query: str, top_k: int = 3) -> str:
        """Retorna os resultados ja formatados como string."""
        resultados = self.retrieve(query, top_k)

        if not resultados:
            return f"Nenhum resultado encontrado para: '{query}'"

        from pathlib import Path
        linhas = [f"Resultados para: '{query}'\n"]
        for i, r in enumerate(resultados, 1):
            nome = Path(r["fonte"]).name
            pag  = f"p.{r['pagina']}" if r["pagina"] else ""
            ref  = f"{nome} {pag}".strip()
            linhas.append(
                f"[{i}] Score: {r['score']:.2f} | {ref}\n"
                f"    {r['texto'][:200]}{'...' if len(r['texto']) > 200 else ''}"
            )

        return "\n\n".join(linhas)


# ---------------------------------------------------------------------------
# Teste rapido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    try:
        retriever = Retriever()
        queries = [
            "O que e teste?",
            "explique o conceito de caixa preta",
            "O que e validação?",
        ]
        for q in queries:
            print(f"\n{'='*55}")
            print(retriever.retrieve_formatado(q, top_k=2))
    except RuntimeError as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)