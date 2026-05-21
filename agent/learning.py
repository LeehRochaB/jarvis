"""
agent/learning.py
-----------------
Módulo de aprendizado ativo do JARVIS.

Implementa as 2 funcionalidades de aprendizado obrigatórias:

1. Active Recall (interativa) — obrigatório ser interativa
   O JARVIS gera uma pergunta sobre o material, o aluno responde,
   e o JARVIS avalia e dá feedback personalizado.

2. Geração de Exercícios
   O JARVIS gera questões de múltipla escolha ou dissertativas
   com base nos materiais de estudo indexados.

Ambas usam o pipeline RAG para basear as perguntas no conteúdo
real dos documentos do estudante.

Uso:
    from agent.learning import LearningModule
    lm = LearningModule()

    # Active Recall
    r = lm.gerar_pergunta_active_recall("KNN")
    print(r["pergunta"])

    # Exercícios
    r = lm.gerar_exercicios("embeddings", quantidade=3)
    print(r["exercicios"])
"""

import json
import logging
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://llm.liaufms.org/v1/gemma-3-12b-it",
    api_key="Cxt2ftLF7d3mHS2JdiFqB-eSDAQeZvFATPXPs02lV9A",
)
MODEL = "google/gemma-3-12b-it"


def _chamar_llm(prompt: str, max_tokens: int = 800) -> str:
    """Chama o Gemma 12B com um prompt simples e retorna o texto."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erro na LLM (learning): {e}")
        return f"[ERRO] Falha na comunicação com o modelo: {e}"


def _buscar_contexto_rag(topico: str) -> str:
    """Busca contexto RAG sobre o tópico para embasar as perguntas."""
    try:
        from tools.rag_tool import buscar_material_rag
        resultado = buscar_material_rag(topico, top_k=3)
        if resultado.get("total", 0) > 0:
            # Retorna só o texto dos trechos, sem o cabeçalho
            trechos = resultado.get("trechos", [])
            return "\n\n".join(t["texto"] for t in trechos)
    except Exception as e:
        logger.warning(f"RAG indisponível para learning: {e}")
    return ""


# ---------------------------------------------------------------------------
# Classe LearningModule
# ---------------------------------------------------------------------------

class LearningModule:
    """
    Módulo de aprendizado ativo do JARVIS.
    Implementa Active Recall e Geração de Exercícios.
    """

    def __init__(self):
        logger.info("LearningModule inicializado.")

    # ── 1. ACTIVE RECALL ──────────────────────────────────────────────────

    def gerar_pergunta_active_recall(self, topico: str) -> dict:
        """
        Gera uma pergunta de active recall sobre o tópico.

        O JARVIS busca o conteúdo do material e gera uma pergunta
        que testa compreensão real (não apenas memorização).

        Parâmetros
        ----------
        topico : str — Tópico de estudo (ex: "KNN", "embeddings")

        Retorna
        -------
        dict com:
            pergunta : str — pergunta gerada
            gabarito : str — resposta esperada (para avaliação interna)
            topico   : str — tópico informado
            contexto : str — trecho do material usado
        """
        topico = topico.strip()
        contexto = _buscar_contexto_rag(topico)

        if contexto:
            prompt = f"""Voce e um professor universitario.
Com base no seguinte conteudo sobre "{topico}":

{contexto}

Gere UMA pergunta de active recall que teste a COMPREENSAO do estudante.
(não apenas memorização). A pergunta deve ser clara e objetiva.

Responda EXATAMENTE neste formato JSON (sem mais nada):
{{
  "pergunta": "A pergunta aqui",
  "gabarito": "A resposta esperada aqui (2-4 frases)"
}}"""
        else:
            prompt = f"""Voce e um professor universitario.
Gere UMA pergunta de active recall sobre o tópico "{topico}" que teste
a compreensão do estudante.

Responda EXATAMENTE neste formato JSON (sem mais nada):
{{
  "pergunta": "A pergunta aqui",
  "gabarito": "A resposta esperada aqui (2-4 frases)"
}}"""

        resposta_llm = _chamar_llm(prompt, max_tokens=400)

        # Tenta extrair JSON
        try:
            import re
            match = re.search(r"\{.*\}", resposta_llm, re.DOTALL)
            if match:
                dados = json.loads(match.group())
                pergunta = dados.get("pergunta", resposta_llm)
                gabarito = dados.get("gabarito", "")
            else:
                pergunta = resposta_llm
                gabarito = ""
        except (json.JSONDecodeError, AttributeError):
            pergunta = resposta_llm
            gabarito = ""

        logger.info(json.dumps({
            "modulo": "active_recall",
            "acao": "gerar_pergunta",
            "topico": topico,
            "tem_contexto_rag": bool(contexto),
        }, ensure_ascii=False))

        return {
            "pergunta": pergunta,
            "gabarito": gabarito,
            "topico":   topico,
            "contexto": contexto[:300] if contexto else "",
        }

    def avaliar_resposta(
        self,
        pergunta: str,
        resposta_aluno: str,
        gabarito: str = "",
    ) -> dict:
        """
        Avalia a resposta do aluno e fornece feedback construtivo.

        Parâmetros
        ----------
        pergunta      : str — pergunta que foi feita
        resposta_aluno: str — resposta dada pelo aluno
        gabarito      : str — resposta esperada (pode ser vazia)

        Retorna
        -------
        dict com:
            avaliacao   : str  — feedback completo do JARVIS
            acertou     : bool — True se a resposta foi correta
            pontuacao   : str  — "correta" | "parcial" | "incorreta"
        """
        gabarito_str = f"\nResposta esperada: {gabarito}" if gabarito else ""

        prompt = f"""Voce e um professor avaliando a resposta de um estudante.

Pergunta: {pergunta}
Resposta do aluno: {resposta_aluno}{gabarito_str}

Avalie a resposta do aluno de forma construtiva:
1. Diga se está CORRETA, PARCIALMENTE CORRETA ou INCORRETA
2. Explique o que está certo (se houver)
3. Corrija o que está errado (se houver)
4. Dê uma dica para fixar o conceito

Seja encorajador e didático. Responda em português."""

        avaliacao = _chamar_llm(prompt, max_tokens=600)

        # Determina pontuação
        avaliacao_lower = avaliacao.lower()
        if "incorreta" in avaliacao_lower or "errada" in avaliacao_lower:
            pontuacao = "incorreta"
            acertou = False
        elif "parcialmente" in avaliacao_lower or "parcial" in avaliacao_lower:
            pontuacao = "parcial"
            acertou = False
        else:
            pontuacao = "correta"
            acertou = True

        logger.info(json.dumps({
            "modulo": "active_recall",
            "acao": "avaliar_resposta",
            "pontuacao": pontuacao,
        }, ensure_ascii=False))

        return {
            "avaliacao": avaliacao,
            "acertou":   acertou,
            "pontuacao": pontuacao,
        }

    def sessao_active_recall_cli(self, topico: str) -> None:
        """
        Conduz uma sessão interativa de active recall no terminal.
        Usada pelo main.py ou diretamente.

        Parâmetros
        ----------
        topico : str — tópico que o aluno quer revisar
        """
        print(f"\n🧠 Active Recall — Tópico: {topico}")
        print("─" * 45)

        resultado = self.gerar_pergunta_active_recall(topico)
        pergunta  = resultado["pergunta"]
        gabarito  = resultado["gabarito"]

        print(f"\n📌 Pergunta:\n{pergunta}")
        print("\n(Pense bem antes de responder...)")

        try:
            resposta_aluno = input("\nSua resposta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada.")
            return

        if not resposta_aluno:
            print("JARVIS: Nenhuma resposta fornecida.")
            return

        print("\n⏳ Avaliando sua resposta...")
        avaliacao = self.avaliar_resposta(pergunta, resposta_aluno, gabarito)

        print(f"\n📊 Avaliação:\n{avaliacao['avaliacao']}")

        emoji = "✅" if avaliacao["acertou"] else \
                "🔶" if avaliacao["pontuacao"] == "parcial" else "❌"
        print(f"\n{emoji} Resultado: {avaliacao['pontuacao'].upper()}")
        print("─" * 45)

    # ── 2. GERAÇÃO DE EXERCÍCIOS ──────────────────────────────────────────

    def gerar_exercicios(self, topico: str, quantidade: int = 3) -> dict:
        topico = topico.strip()
        contexto = _buscar_contexto_rag(topico)
        contexto_str = f"\nBaseie-se neste conteudo:\n{contexto}\n" if contexto else ""

        # Gera em lotes de 5 para garantir a quantidade
        LOTE = 5
        todos_exercicios = []
        num_lotes = (quantidade + LOTE - 1) // LOTE  # arredonda para cima

        for lote in range(num_lotes):
            inicio = lote * LOTE + 1
            fim = min(inicio + LOTE - 1, quantidade)
            qtd_lote = fim - inicio + 1

            prompt = f"""Voce e um professor universitario.
            {contexto_str}
            Crie EXATAMENTE {qtd_lote} exercicio(s) sobre "{topico}", numerados de {inicio} a {fim}.
            Apenas enunciados, sem respostas.

            Exercicio {inicio}: [enunciado]
            {"..." if qtd_lote > 1 else ""}
            Exercicio {fim}: [enunciado]

            Responda em portugues."""

            resultado = _chamar_llm(prompt, max_tokens=2048)
            todos_exercicios.append(resultado)

        exercicios_final = "\n\n".join(todos_exercicios)
        exercicios_final += "\n\nDeseja ver as respostas? Digite 'sim' para ver o gabarito."

        return {
            "exercicios": exercicios_final,
            "topico": topico,
            "quantidade": quantidade,
        }

    
    def gerar_exercicios_com_gabarito(self, topico: str, quantidade: int = 3) -> dict:
        """Gera exercicios com gabarito sobre um topico."""
        topico = topico.strip()
        contexto = _buscar_contexto_rag(topico)
        contexto_str = f"\nBaseie-se neste conteudo:\n{contexto}\n" if contexto else ""

        prompt = f"""Voce e um professor universitario.
        {contexto_str}
        Crie {quantidade} exercicio(s) sobre "{topico}" com gabarito completo.

        Formate assim:

        Exercicio 1: [enunciado]
        Resposta: [gabarito detalhado]

        Exercicio 2: [enunciado]
        Resposta: [gabarito detalhado]

        Responda em portugues."""

        exercicios = _chamar_llm(prompt, max_tokens=4096)
        return {"exercicios": exercicios, "topico": topico, "quantidade": quantidade}

# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lm = LearningModule()

    print("\n=== Teste: Geração de Exercícios ===")
    resultado = lm.gerar_exercicios("KNN", quantidade=2)
    print(resultado["exercicios"])

    print("\n=== Teste: Sessão Active Recall ===")
    lm.sessao_active_recall_cli("embeddings")