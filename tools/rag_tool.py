"""
tools/rag_tool.py
-----------------
Ferramenta de busca nos materiais de estudo (RAG) do JARVIS.

Esta ferramenta é o ponto de entrada do pipeline RAG para o agente.
Ela conecta o agente com o retriever, formata o resultado e retorna
um texto estruturado pronto para a LLM usar na geração da resposta.

Pipeline interno:
    query do usuário
        → gera embedding da query
        → busca chunks mais similares no ChromaDB
        → retorna trechos relevantes + metadados
        → formata resposta para a LLM

Funções exportadas (usadas pelo agente):
    buscar_material_rag(query, top_k)  → trechos relevantes dos documentos
"""

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Adiciona raiz do projeto ao path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Configuração de logs
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import lazy do retriever (evita carregar modelo na importação do módulo)
# ---------------------------------------------------------------------------
_retriever = None


def _get_retriever():
    """
    Carrega o retriever sob demanda (lazy loading).
    Assim o modelo de embeddings só é carregado quando realmente necessário.
    """
    global _retriever
    if _retriever is None:
        try:
            from rag.retriever import Retriever
            _retriever = Retriever()
            logger.info("Retriever carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar retriever: {e}")
            raise RuntimeError(
                f"Não foi possível inicializar o retriever RAG: {e}\n"
                "Execute setup.py primeiro para indexar os documentos."
            )
    return _retriever


# ---------------------------------------------------------------------------
# Função principal (chamada pelo agente)
# ---------------------------------------------------------------------------

def buscar_material_rag(query: str, top_k: int = 3) -> dict:
    """
    Busca trechos relevantes nos materiais de estudo indexados.

    Parâmetros
    ----------
    query : str
        Pergunta ou termo de busca do usuário.
        Exemplos: "explique KNN", "o que é regressão logística?",
                  "como funciona o BM25?"

    top_k : int
        Número de trechos mais relevantes a retornar. Padrão: 3.

    Retorna
    -------
    dict com:
        query       : pergunta original
        total       : quantidade de trechos encontrados
        trechos     : lista de trechos com texto e metadados
        contexto    : texto formatado para a LLM usar na resposta
        mensagem    : resumo do resultado para log/debug
    """
    query = query.strip()

    if not query:
        return {"erro": "A query de busca não pode estar vazia."}

    if top_k < 1 or top_k > 10:
        top_k = 3

    try:
        retriever = _get_retriever()
        resultados = retriever.retrieve(query, top_k=top_k)
    except RuntimeError as e:
        return {"erro": str(e)}
    except Exception as e:
        logger.error(f"Erro na busca RAG: {e}")
        return {"erro": f"Falha na busca: {e}"}

    if not resultados:
        mensagem = f"Nenhum trecho relevante encontrado para: '{query}'"
        logger.info(json.dumps({
            "ferramenta": "buscar_material_rag",
            "entrada": {"query": query, "top_k": top_k},
            "saida": mensagem,
        }, ensure_ascii=False))
        return {
            "query": query,
            "total": 0,
            "trechos": [],
            "contexto": mensagem,
            "mensagem": mensagem,
        }

    # Formata os trechos para a LLM
    linhas_contexto = []
    for i, trecho in enumerate(resultados, 1):
        texto   = trecho.get("texto", "")
        fonte   = trecho.get("fonte", "Desconhecido")
        pagina  = trecho.get("pagina", "")
        score   = trecho.get("score", None)

        # Nome curto do arquivo
        nome_arquivo = Path(fonte).name if fonte else "Desconhecido"
        ref = f"{nome_arquivo}" + (f", p.{pagina}" if pagina else "")

        linhas_contexto.append(
            f"[Trecho {i} — Fonte: {ref}]\n{texto}"
        )

    contexto = "\n\n---\n\n".join(linhas_contexto)

    contexto_completo = (
        f"Foram encontrados {len(resultados)} trecho(s) relevante(s) "
        f"nos materiais de estudo para a consulta '{query}':\n\n"
        f"{contexto}\n\n"
        "Use estes trechos para responder ao estudante de forma clara e didática."
    )

    mensagem = (
        f"RAG: {len(resultados)} trecho(s) recuperado(s) "
        f"para '{query[:50]}{'...' if len(query) > 50 else ''}'"
    )

    logger.info(json.dumps({
        "ferramenta": "buscar_material_rag",
        "entrada": {"query": query, "top_k": top_k},
        "saida": mensagem,
    }, ensure_ascii=False))

    return {
        "query":    query,
        "total":    len(resultados),
        "trechos":  resultados,
        "contexto": contexto_completo,
        "mensagem": mensagem,
    }


# ---------------------------------------------------------------------------
# Função auxiliar: verifica se o índice RAG está pronto
# ---------------------------------------------------------------------------

def verificar_indice() -> dict:
    """
    Verifica se o índice vetorial foi criado e tem documentos.

    Retorna
    -------
    dict com status e quantidade de documentos indexados.
    """
    try:
        retriever = _get_retriever()
        total = retriever.contar_documentos()
        if total > 0:
            return {
                "status": "ok",
                "total_chunks": total,
                "mensagem": f"✅ Índice RAG pronto com {total} chunks indexados.",
            }
        else:
            return {
                "status": "vazio",
                "total_chunks": 0,
                "mensagem": "⚠️ Índice RAG está vazio. Execute setup.py para indexar os documentos.",
            }
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": f"❌ Índice RAG não disponível: {e}",
        }


# ---------------------------------------------------------------------------
# Teste rápido (rodar diretamente: python tools/rag_tool.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Verificando índice RAG ===")
    status = verificar_indice()
    print(status["mensagem"])

    if status["status"] == "ok":
        print("\n=== Teste: buscar_material_rag ===")
        queries_teste = [
            "O que é KNN?",
            "explique embeddings",
            "como funciona BM25",
        ]
        for q in queries_teste:
            print(f"\n🔍 Query: {q}")
            resultado = buscar_material_rag(q, top_k=2)
            if "erro" in resultado:
                print(f"  ❌ {resultado['erro']}")
            else:
                print(f"  {resultado['mensagem']}")
                for i, t in enumerate(resultado["trechos"], 1):
                    fonte = Path(t.get("fonte", "?")).name
                    texto = t.get("texto", "")[:120].replace("\n", " ")
                    print(f"  [{i}] {fonte}: {texto}...")
    else:
        print("\n⚠️  Execute setup.py primeiro para indexar os documentos.")
        print("    python setup.py")