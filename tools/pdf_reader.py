"""
tools/pdf_reader.py
-------------------
Modulo de leitura e interpretacao de PDFs enviados pelo usuario.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from openai import OpenAI

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
                "Pode ser um arquivo escaneado (imagem)."
            )
        logger.info(f"PDF lido: {path.name} | {len(reader.pages)} pagina(s) | {len(texto)} chars")
        return texto.strip()
    except Exception as e:
        logger.error(f"Erro ao ler PDF '{caminho}': {e}")
        return f"[ERRO] Falha ao ler o PDF: {e}"


# ---------------------------------------------------------------------------
# Filtro de periodo
# ---------------------------------------------------------------------------

def _extrair_periodo(texto: str, instrucao: str) -> str:
    """Extrai apenas o trecho do periodo mencionado na instrucao."""
    # Detecta periodo mencionado (ex: 2025.1, 2024.2, 2026.1)
    match = re.search(r"20\d\d\.[12]", instrucao)
    if not match:
        return texto  # sem filtro de periodo

    periodo = match.group()  # ex: "2025.1"

    # Procura o inicio do periodo no texto (varias formas possiveis)
    padroes_inicio = [
        f"PERÍODO {periodo}",
        f"PERIODO {periodo}",
        f"período {periodo}",
        periodo,
    ]
    idx_inicio = -1
    for padrao in padroes_inicio:
        idx = texto.find(padrao)
        if idx != -1:
            idx_inicio = idx
            break

    if idx_inicio == -1:
        return texto  # periodo nao encontrado, retorna tudo

    # Encontra o proximo periodo para saber onde termina
    resto = texto[idx_inicio + len(periodo) + 5:]
    proximo = re.search(r"PERÍODO 20\d\d\.[12]|PERIODO 20\d\d\.[12]", resto)
    if proximo:
        idx_fim = idx_inicio + len(periodo) + 5 + proximo.start()
    else:
        # Sem proximo periodo — pega ate componentes curriculares ou fim
        fim_alt = resto.find("COMPONENTES CURRICULARES")
        if fim_alt != -1:
            idx_fim = idx_inicio + len(periodo) + 5 + fim_alt
        else:
            idx_fim = idx_inicio + 4000

    trecho = texto[idx_inicio:idx_fim].strip()
    logger.info(f"Filtrado periodo {periodo}: {len(trecho)} chars extraidos")
    return trecho


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------

def processar_pdf_com_instrucao(caminho: str, instrucao: str) -> dict:
    texto = ler_pdf(caminho)
    nome_arquivo = Path(caminho).name

    if texto.startswith("[ERRO]"):
        return {"status": "erro", "arquivo": nome_arquivo, "mensagem": texto}
    if texto.startswith("[AVISO]"):
        return {"status": "aviso", "arquivo": nome_arquivo, "mensagem": texto}

    instrucao_lower = instrucao.lower()

    precisa_agenda = any(p in instrucao_lower for p in [
        "agenda", "cronograma", "aula", "horario", "adiciona", "cadastra na agenda"
    ])
    precisa_notas = any(p in instrucao_lower for p in [
        "nota", "media", "disciplina", "materia", "boletim", "cadastra nota",
        "registra nota", "cadastra disciplina"
    ])

    # Filtra texto por periodo se necessario
    if precisa_notas:
        texto_filtrado = _extrair_periodo(texto, instrucao)
        texto_limitado = texto_filtrado[:5000]
        if len(texto_filtrado) > 5000:
            texto_limitado += f"\n\n[... truncado. Total: {len(texto_filtrado)} chars ...]"
    else:
        texto_limitado = texto[:4000]
        if len(texto) > 4000:
            texto_limitado += f"\n\n[... truncado. Total: {len(texto)} chars ...]"

    # Detecta periodo na instrucao para incluir no prompt
    periodo_match = re.search(r"20\d\d\.[12]", instrucao)
    periodo_str = f"do periodo {periodo_match.group()}" if periodo_match else ""

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

Conteudo do PDF ({nome_arquivo}) — trecho {periodo_str}:
{texto_limitado}

INSTRUCOES CRITICAS:
1. Extraia APENAS as disciplinas {periodo_str} presentes no trecho acima.
2. NAO inclua disciplinas de outros periodos.
3. Para disciplinas com situacao MATRICULADO, a nota deve ser null.
4. Use apenas os dados do trecho fornecido.

Retorne APENAS um JSON valido, sem texto antes ou depois:
{{
  "tipo": "notas",
  "resumo": "descricao resumida do que foi encontrado",
  "disciplinas": [
    {{
      "nome": "NOME DA DISCIPLINA",
      "formula": "media_simples",
      "nota_minima": 6.0,
      "avaliacoes": [
        {{"nome": "Nota", "nota": 7.5}}
      ]
    }}
  ]
}}"""

    else:
        prompt = f"""Voce e um assistente academico. O usuario enviou um PDF e quer:
"{instrucao}"

Conteudo do PDF ({nome_arquivo}):
{texto_limitado}

Responda a solicitacao do usuario de forma clara e organizada em portugues.
Nao use asteriscos (**) nem markdown na resposta. Use texto simples."""

    # Chama a LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        resposta_llm = response.choices[0].message.content.strip()
    except Exception as e:
        return {"status": "erro", "arquivo": nome_arquivo, "mensagem": f"Erro ao processar com a LLM: {e}"}

    acoes = []

    if precisa_agenda or precisa_notas:
        try:
            match = re.search(r"\{.*\}", resposta_llm, re.DOTALL)
            if match:
                dados = json.loads(match.group())

                if precisa_agenda and "eventos" in dados:
                    eventos  = dados.get("eventos", [])
                    resumo   = dados.get("resumo", "")
                    acoes    = [{"tipo": "agenda", "eventos": eventos}]
                    mensagem = (
                        f"PDF lido: {nome_arquivo}\n"
                        f"{resumo}\n"
                        f"Encontrei {len(eventos)} evento(s).\n\n"
                        f"Deseja que eu adicione esses eventos na sua agenda? (sim/nao)"
                    )
                    return {
                        "status": "ok", "tipo": "cronograma",
                        "arquivo": nome_arquivo, "acoes": acoes,
                        "mensagem": mensagem, "dados": dados,
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
                    return {
                        "status": "ok", "tipo": "notas",
                        "arquivo": nome_arquivo, "acoes": acoes,
                        "mensagem": mensagem, "dados": dados,
                    }
        except (json.JSONDecodeError, AttributeError):
            pass

    mensagem = f"PDF lido: {nome_arquivo}\n\n{resposta_llm}"
    return {
        "status": "ok", "tipo": "geral",
        "arquivo": nome_arquivo, "acoes": [],
        "mensagem": mensagem,
    }


# ---------------------------------------------------------------------------
# Confirmacao de importacao
# ---------------------------------------------------------------------------

def confirmar_importacao_agenda(eventos: list) -> dict:
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