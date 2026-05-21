
import os
import sys
import warnings
import logging

# Suprime todos os warnings externos
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Redireciona stderr permanentemente para suprimir warnings do HF
import io
_stderr_original = sys.stderr
sys.stderr = open(os.devnull, 'w')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.makedirs("logs", exist_ok=True)
os.makedirs("data_store", exist_ok=True)

from agent.jarvis import Jarvis

def main():
    jarvis = Jarvis()
    print("\n" + "="*50)
    print("  JARVIS ACADEMICO — Assistente Inteligente")
    print("  Powered by Gemma 12B (FACOM/UFMS)")
    print("="*50)
    print("  Comandos: /sair, /limpar, /hist")
    print("="*50 + "\n")

    while True:
        try:
            entrada = input("Voce: ").strip()
            if not entrada:
                continue
            if entrada.lower() == "/sair":
                print("JARVIS: Ate logo! Bons estudos!")
                break
            elif entrada.lower() == "/limpar":
                jarvis.limpar_historico()
            elif entrada.lower() == "/hist":
                jarvis.mostrar_historico()
            else:
                print("JARVIS: ", end="", flush=True)
                resposta = jarvis.chat(entrada)
                print(resposta + "\n")
        except KeyboardInterrupt:
            print("\nJARVIS: Encerrando...")
            break
        except Exception as e:
            print(f"[ERRO] {e}\n")

if __name__ == "__main__":
    main()