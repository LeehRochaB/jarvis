"""
tools/pdf_reader.py
-------------------
Modulo de leitura e interpretacao de PDFs enviados pelo usuario.

O usuario envia um PDF e diz o que quer extrair.
A LLM interpreta o conteudo conforme a instrucao do usuario.

Exemplos de uso:
  - "Anexei meu cronograma, adiciona as aulas na agenda"
  - "Esse PDF tem minhas notas, cadastra as disciplinas"
  - "Extraia os exercicios desse PDF"
  - "Resume o conteudo desse documento"
  - "Quais sao os requisitos listados nesse PDF?"

Funcoes exportadas:
    ler_pdf(caminho)
    processar_pdf_com_instrucao(caminho, instrucao)
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://llm.liaufms.org/v1/gemma-3-12b-it",
    api_key="Cxt2ftLF7d3mHS2JdiFqB-eSDAQeZvFATPXPs02lV9A",
)
MODEL = "google/gemma-3-12b-it"


# ---------------------------------------------------------------------------
# Extracao de texto
# ---------------------------------------------------------------------------

def ler_pdf(caminho: str) -> str:
    """
    Extrai o texto de um arquivo PDF.

    Parametros
    ----------
    caminho : str - Caminho completo para o arquivo PDF

    Retorna
    -------
    str - Texto extraido do PDF ou mensagem de erro
    """
    path = Path(caminho)

    if not path.exists():
        return f"[ERRO] Arquivo nao encontrado: {caminho}"

    if path.suffix.lower() != ".pdf":
        return f"[ERRO] O arquivo nao e um PDF: {caminho}"

    try:
        reader = PdfReader(str(path))
        texto  = ""
        for i, page in enumerate(reader.pages):
            conteudo = page.extract_text() or ""
            if conteudo.strip():
                texto += f"\n--- Pagina {i+1} ---\n{conteudo}"

        if not texto.strip():
            return (
                "[AVISO] O PDF nao possui texto extraivel. "
                "Pode ser um arquivo escaneado (imagem). "
                "Tente converter para PDF com texto antes de enviar."
            )

        logger.info(
            f"PDF lido: {path.name} | "
            f"{len(reader.pages)} pagina(s) | "
            f"{len(texto)} caracteres extraidos"
        )
        return texto.strip()

    except Exception as e:
        logger.error(f"Erro ao ler PDF '{caminho}': {e}")
        return f"[ERRO] Falha ao ler o PDF: {e}"


# ---------------------------------------------------------------------------
# Processamento com instrucao livre do usuario
# ---------------------------------------------------------------------------

def processar_pdf_com_instrucao(caminho: str, instrucao: str) -> dict:
    """
    Le um PDF e executa qualquer instrucao do usuario sobre o conteudo.

    O usuario pode pedir qualquer coisa:
      - Extrair cronograma e adicionar na agenda
      - Extrair notas e cadastrar disciplinas
      - Resumir o conteudo
      - Listar exercicios
      - Extrair requisitos
      - Responder perguntas sobre o documento
      - Qualquer outra interpretacao

    A LLM interpreta o conteudo conforme a instrucao.

    Parametros
    ----------
    caminho   : str - Caminho completo para o arquivo PDF
    instrucao : str - O que o usuario quer fazer com o PDF
                      Ex: "adiciona as aulas na agenda"
                          "cadastra as disciplinas e notas"
                          "resume o conteudo"
                          "extraia os exercicios"

    Retorna
    -------
    dict com:
        status   : "ok" ou "erro"
        arquivo  : nome do arquivo
        resposta : resposta da LLM conforme a instrucao
        acoes    : lista de acoes estruturadas (quando aplicavel)
        mensagem : texto final para mostrar ao usuario
    """
    # 1. Le o PDF
    texto = ler_pdf(caminho)
    nome_arquivo = Path(caminho).name

    if texto.startswith("[ERRO]"):
        return {
            "status":   "erro",
            "arquivo":  nome_arquivo,
            "mensagem": texto,
        }

    if texto.startswith("[AVISO]"):
        return {
            "status":   "aviso",
            "arquivo":  nome_arquivo,
            "mensagem": texto,
        }

    # 2. Limita o texto para nao exceder o contexto da LLM
    texto_limitado = texto[:4000]
    if len(texto) > 4000:
        texto_limitado += f"\n\n[... texto truncado. Total: {len(texto)} caracteres ...]"

    # 3. Detecta se a instrucao pede acao estruturada (agenda/notas)
    instrucao_lower = instrucao.lower()

    precisa_agenda = any(p in instrucao_lower for p in [
        "agenda", "cronograma", "aula", "horario", "adiciona", "cadastra na agenda"
    ])
    precisa_notas = any(p in instrucao_lower for p in [
        "nota", "media", "disciplina", "materia", "boletim", "cadastra nota",
        "registra nota", "cadastra disciplina"
    ])

    # 4. Monta prompt conforme o tipo de instrucao
    if precisa_agenda:
        prompt = f"""Voce e um assistente academico. O usuario enviou um PDF e quer:
"{instrucao}"

Conteudo do PDF ({nome_arquivo}):
{texto_limitado}

Extraia os eventos/aulas do documento e retorne APENAS um JSON valido:
{{
  "tipo": "cronograma",
  "resumo": "descricao do que foi encontrado",
  "eventos": [
    {{
      "dia": "YYYY-MM-DD",
      "hora": "HH:MM",
      "evento": "descricao",
      "tipo": "aula|prova|entrega|reuniao|outro"
    }}
  ]
}}

Use o ano atual se nao especificado. Converta datas para ISO (YYYY-MM-DD)."""

    elif precisa_notas:
        prompt = f"""Voce e um assistente academico. O usuario enviou um PDF e quer:
"{instrucao}"

Conteudo do PDF ({nome_arquivo}):
{texto_limitado}

Extraia as disciplinas e notas do documento e retorne APENAS um JSON valido:
{{
  "tipo": "notas",
  "resumo": "descricao do que foi encontrado",
  "disciplinas": [
    {{
      "nome": "nome da disciplina",
      "formula": "media_simples|ponderada|maior_nota",
      "nota_minima": 6.0,
      "avaliacoes": [
        {{"nome": "P1", "nota": 7.5}},
        {{"nome": "P2", "nota": null}}
      ]
    }}
  ]
}}

nota null = avaliacao ainda nao realizada."""

    else:
        # Instrucao livre — a LLM responde em texto natural
        prompt = f"""Voce e um assistente academico. O usuario enviou um PDF e quer:
"{instrucao}"

Conteudo do PDF ({nome_arquivo}):
{texto_limitado}

Responda a solicitacao do usuario de forma clara e organizada em portugues.
Nao use asteriscos (**) nem markdown na resposta. Use texto simples."""

    # 5. Chama a LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        resposta_llm = response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "status":   "erro",
            "arquivo":  nome_arquivo,
            "mensagem": f"Erro ao processar com a LLM: {e}",
        }

    # 6. Processa resposta conforme o tipo
    acoes = []

    if precisa_agenda or precisa_notas:
        # Tenta extrair JSON estruturado
        try:
            match = re.search(r"\{.*\}", resposta_llm, re.DOTALL)
            if match:
                dados = json.loads(match.group())

                if precisa_agenda and "eventos" in dados:
                    eventos = dados.get("eventos", [])
                    resumo  = dados.get("resumo", "")
                    acoes   = [{"tipo": "agenda", "eventos": eventos}]
                    mensagem = (
                        f"PDF lido: {nome_arquivo}\n"
                        f"{resumo}\n"
                        f"Encontrei {len(eventos)} evento(s).\n\n"
                        f"Deseja que eu adicione esses eventos na sua agenda? (sim/nao)"
                    )
                    logger.info(json.dumps({
                        "ferramenta": "processar_pdf_com_instrucao",
                        "arquivo": nome_arquivo,
                        "tipo": "cronograma",
                        "eventos": len(eventos),
                    }, ensure_ascii=False))
                    return {
                        "status":   "ok",
                        "tipo":     "cronograma",
                        "arquivo":  nome_arquivo,
                        "acoes":    acoes,
                        "mensagem": mensagem,
                        "dados":    dados,
                    }

                elif precisa_notas and "disciplinas" in dados:
                    disciplinas = dados.get("disciplinas", [])
                    resumo      = dados.get("resumo", "")
                    acoes       = [{"tipo": "notas", "disciplinas": disciplinas}]
                    nomes       = [d.get("nome", "") for d in disciplinas]
                    mensagem = (
                        f"PDF lido: {nome_arquivo}\n"
                        f"{resumo}\n"
                        f"Encontrei {len(disciplinas)} disciplina(s):\n"
                        + "\n".join(f"  - {n}" for n in nomes)
                        + "\n\nDeseja que eu cadastre essas disciplinas e notas? (sim/nao)"
                    )
                    logger.info(json.dumps({
                        "ferramenta": "processar_pdf_com_instrucao",
                        "arquivo": nome_arquivo,
                        "tipo": "notas",
                        "disciplinas": len(disciplinas),
                    }, ensure_ascii=False))
                    return {
                        "status":      "ok",
                        "tipo":        "notas",
                        "arquivo":     nome_arquivo,
                        "acoes":       acoes,
                        "mensagem":    mensagem,
                        "dados":       dados,
                    }
        except (json.JSONDecodeError, AttributeError):
            pass

    # Resposta livre (resumo, exercicios, perguntas, etc.)
    mensagem = f"PDF lido: {nome_arquivo}\n\n{resposta_llm}"
    logger.info(json.dumps({
        "ferramenta": "processar_pdf_com_instrucao",
        "arquivo": nome_arquivo,
        "instrucao": instrucao[:100],
    }, ensure_ascii=False))

    return {
        "status":   "ok",
        "tipo":     "geral",
        "arquivo":  nome_arquivo,
        "acoes":    [],
        "mensagem": mensagem,
    }


def confirmar_importacao_agenda(eventos: list) -> dict:
    """Adiciona eventos extraidos do PDF na agenda."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.agenda import adicionar_evento

    adicionados, erros = 0, []
    for e in eventos:
        try:
            r = adicionar_evento(
                dia    = e.get("dia", ""),
                hora   = e.get("hora", "00:00"),
                evento = e.get("evento", ""),
                tipo   = e.get("tipo", "outro"),
            )
            if r.get("status") == "ok":
                adicionados += 1
            else:
                erros.append(e.get("evento", ""))
        except Exception as ex:
            erros.append(str(ex))

    mensagem = f"Importacao concluida: {adicionados} evento(s) adicionado(s) na agenda."
    if erros:
        mensagem += f"\nNao foi possivel adicionar: {', '.join(erros[:3])}"
    return {"status": "ok", "adicionados": adicionados, "mensagem": mensagem}


def confirmar_importacao_notas(disciplinas: list) -> dict:
    """Cadastra disciplinas e notas extraidas do PDF."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.notas import cadastrar_disciplina, registrar_nota

    cadastradas, notas_reg, erros = 0, 0, []
    for disc in disciplinas:
        try:
            nomes = [a["nome"] for a in disc.get("avaliacoes", [])]
            r = cadastrar_disciplina(
                nome        = disc["nome"],
                avaliacoes  = nomes,
                formula     = disc.get("formula", "media_simples"),
                nota_minima = disc.get("nota_minima", 6.0),
            )
            if r.get("status") == "ok":
                cadastradas += 1
                for aval in disc.get("avaliacoes", []):
                    if aval.get("nota") is not None:
                        registrar_nota(disc["nome"], aval["nome"], aval["nota"])
                        notas_reg += 1
        except Exception as ex:
            erros.append(str(ex))

    mensagem = (
        f"Importacao concluida:\n"
        f"  {cadastradas} disciplina(s) cadastrada(s)\n"
        f"  {notas_reg} nota(s) registrada(s)"
    )
    if erros:
        mensagem += f"\nErros: {', '.join(erros[:3])}"
    return {"status": "ok", "cadastradas": cadastradas, "mensagem": mensagem}