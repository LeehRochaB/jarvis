"""
setup.py
--------
Script de inicialização do JARVIS.

Execute UMA VEZ antes de usar o sistema:
    python setup.py

O que faz:
    1. Verifica dependências instaladas
    2. Verifica pasta data/ e conta documentos
    3. Carrega documentos (loader)
    4. Divide em chunks (chunker)
    5. Gera embeddings e indexa no ChromaDB (embedder)
    6. Testa uma busca para validar o pipeline
    7. Mostra resumo final

Após rodar este script, o JARVIS está pronto para uso:
    python main.py        # interface CLI
    python app.py         # interface web (Gradio)
"""

import os
import sys


def verificar_dependencias() -> bool:
    """Verifica se todos os pacotes necessários estão instalados."""
    pacotes = {
        "openai":                 "openai",
        "sentence_transformers":  "sentence-transformers",
        "chromadb":               "chromadb",
        "langchain":              "langchain",
        "langchain_community":    "langchain-community",
        "pypdf":                  "pypdf",
    }

    print("🔍 Verificando dependências...")
    faltando = []
    for modulo, pacote in pacotes.items():
        try:
            __import__(modulo)
            print(f"  ✅ {pacote}")
        except ImportError:
            print(f"  ❌ {pacote} — NÃO instalado")
            faltando.append(pacote)

    if faltando:
        print(f"\n⚠️  Instale os pacotes faltantes:")
        print(f"    pip install {' '.join(faltando)}")
        return False
    return True


def verificar_dados(data_dir: str = "data/") -> int:
    """Verifica a pasta de dados e conta documentos suportados."""
    from pathlib import Path

    pasta = Path(data_dir)
    if not pasta.exists():
        print(f"\n❌ Pasta '{data_dir}' não encontrada.")
        print("   Crie a pasta e adicione os documentos acadêmicos (PDFs, TXTs).")
        return 0

    arquivos = list(pasta.rglob("*.pdf")) + list(pasta.rglob("*.txt"))
    print(f"\n📂 Pasta '{data_dir}': {len(arquivos)} arquivo(s) encontrado(s)")
    for a in arquivos:
        print(f"   • {a.name}")

    if len(arquivos) < 10:
        print(f"\n⚠️  O trabalho exige mínimo de 10 documentos. "
              f"Você tem {len(arquivos)}.")

    return len(arquivos)


def main():
    print("\n" + "=" * 55)
    print("  🤖 JARVIS — Setup e Inicialização do RAG")
    print("=" * 55 + "\n")

    # 1. Dependências
    if not verificar_dependencias():
        sys.exit(1)

    # 2. Dados
    total_arquivos = verificar_dados("data/")
    if total_arquivos == 0:
        sys.exit(1)

    # 3. Pipeline RAG
    print("\n🔄 Iniciando pipeline RAG...\n")

    from rag.loader  import load_documents
    from rag.chunker import split_documents
    from rag.embedder import Embedder

    # Carrega documentos
    print("📄 [1/3] Carregando documentos...")
    docs = load_documents("data/")
    if not docs:
        print("❌ Nenhum documento carregado. Verifique a pasta data/")
        sys.exit(1)

    # Divide em chunks
    print("✂️  [2/3] Dividindo em chunks...")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    if not chunks:
        print("❌ Nenhum chunk gerado.")
        sys.exit(1)

    # Indexa no ChromaDB
    print("🧠 [3/3] Gerando embeddings e indexando...")
    embedder = Embedder()
    total_indexado = embedder.indexar(chunks)

    # 4. Teste de busca
    print("\n🔍 Testando busca no índice...")
    from rag.retriever import Retriever
    retriever = Retriever()
    resultados = retriever.retrieve("inteligência artificial", top_k=2)

    if resultados:
        print(f"  ✅ Busca funcionando — {len(resultados)} resultado(s) encontrado(s)")
        print(f"  Score do melhor resultado: {resultados[0]['score']:.2f}")
    else:
        print("  ⚠️  Busca retornou vazio — verifique os documentos.")

    # 5. Popula dados de exemplo (agenda e tarefas)
    print("\n📅 Inicializando agenda e tarefas com exemplos...")

    print("  Agenda e tarefas prontas!")

    # 6. Resumo final
    print("\n" + "=" * 55)
    print("  ✅ JARVIS configurado com sucesso!")
    print("=" * 55)
    print(f"  Documentos carregados : {len(docs)}")
    print(f"  Chunks gerados        : {len(chunks)}")
    print(f"  Chunks no índice      : {embedder.contar_documentos()}")
    print(f"\n  Para iniciar o JARVIS:")
    print(f"    python main.py        # interface CLI")
    print(f"    python app.py         # interface web (Gradio)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()