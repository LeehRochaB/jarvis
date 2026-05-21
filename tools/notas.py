"""
tools/notas.py
--------------
Ferramenta de gerenciamento de notas academicas do JARVIS.

Permite cadastrar disciplinas com formulas de calculo flexiveis,
registrar notas e calcular medias automaticamente.

Formulas suportadas:
    media_simples   - soma / quantidade (todas com peso igual)
    ponderada       - cada avaliacao tem um peso definido pelo aluno
    maior_nota      - descarta a menor nota antes de calcular a media
    soma_direta     - soma todas as notas (ex: pontos acumulados)
    personalizada   - o aluno descreve e a LLM interpreta

Estrutura do JSON:
{
  "Verificacao e Validacao": {
    "formula": "media_simples",
    "pesos": {},
    "nota_minima": 6.0,
    "avaliacoes": [
      {"nome": "P1", "nota": 7.5, "peso": 1.0},
      {"nome": "P2", "nota": null, "peso": 1.0}
    ],
    "descricao_formula": ""
  }
}

Funcoes exportadas (usadas pelo agente):
    cadastrar_disciplina(nome, avaliacoes, formula, pesos, nota_minima, descricao_formula)
    registrar_nota(disciplina, avaliacao, nota)
    consultar_notas(disciplina)
    calcular_media(disciplina)
    nota_necessaria(disciplina, avaliacao_faltante)
    listar_disciplinas()
    remover_disciplina(nome)
"""

import json
import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
NOTAS_PATH = Path(r"C:\Users\lebro\data_store\notas.json")

os.makedirs("logs", exist_ok=True)
os.makedirs(str(NOTAS_PATH.parent), exist_ok=True)
logger = logging.getLogger(__name__)

FORMULAS_VALIDAS = {"media_simples", "ponderada", "maior_nota", "soma_direta", "personalizada"}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _carregar_notas() -> dict:
    """Le o arquivo JSON de notas. Cria se nao existir."""
    if not NOTAS_PATH.exists():
        _salvar_notas({})
        return {}
    try:
        with open(NOTAS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao ler notas.json: {e}")
        return {}


def _salvar_notas(dados: dict) -> None:
    """Persiste o dicionario de notas no arquivo JSON."""
    os.makedirs(NOTAS_PATH.parent, exist_ok=True)
    with open(NOTAS_PATH, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def _calcular_media_interna(disciplina_data: dict) -> float | None:
    """
    Calcula a media da disciplina com base na formula configurada.
    Retorna None se houver avaliacoes sem nota.
    """
    avaliacoes = disciplina_data.get("avaliacoes", [])
    formula    = disciplina_data.get("formula", "media_simples")
    pesos      = disciplina_data.get("pesos", {})

    # Filtra avaliacoes com nota lancada
    com_nota    = [a for a in avaliacoes if a.get("nota") is not None]
    sem_nota    = [a for a in avaliacoes if a.get("nota") is None]
    total_avals = len(avaliacoes)

    if not com_nota:
        return None

    notas = [a["nota"] for a in com_nota]

    if formula == "media_simples":
        return round(sum(notas) / len(notas), 2)

    elif formula == "ponderada":
        soma_pesos  = 0
        soma_pontos = 0
        for a in com_nota:
            peso = pesos.get(a["nome"], a.get("peso", 1.0))
            soma_pontos += a["nota"] * peso
            soma_pesos  += peso
        if soma_pesos == 0:
            return None
        return round(soma_pontos / soma_pesos, 2)

    elif formula == "maior_nota":
        # Descarta a menor nota se tiver todas as avaliacoes
        if sem_nota:
            return round(sum(notas) / len(notas), 2)  # provisorio
        notas_ordenadas = sorted(notas)
        notas_validas   = notas_ordenadas[1:] if len(notas_ordenadas) > 1 else notas_ordenadas
        return round(sum(notas_validas) / len(notas_validas), 2)

    elif formula == "soma_direta":
        return round(sum(notas), 2)

    elif formula == "personalizada":
        # Para formula personalizada, usa media simples como calculo base
        # A descricao fica registrada para referencia
        return round(sum(notas) / len(notas), 2)

    return None


def _formatar_situacao(media: float | None, nota_minima: float) -> str:
    """Retorna a situacao do aluno com base na media."""
    if media is None:
        return "Em andamento"
    if media >= nota_minima:
        return f"Aprovado (media: {media:.1f})"
    return f"Reprovado (media: {media:.1f})"


# ---------------------------------------------------------------------------
# Funcoes principais
# ---------------------------------------------------------------------------

def cadastrar_disciplina(
    nome: str,
    avaliacoes: list,
    formula: str = "media_simples",
    pesos: dict = None,
    nota_minima: float = 6.0,
    descricao_formula: str = "",
) -> dict:
    """
    Cadastra uma nova disciplina com sua formula de calculo.

    Parametros
    ----------
    nome              : str   - Nome da disciplina
    avaliacoes        : list  - Lista de nomes das avaliacoes
                                Ex: ["P1", "P2", "Trabalho"]
                                Ou lista de dicts: [{"nome": "P1", "peso": 0.4}]
    formula           : str   - media_simples | ponderada | maior_nota |
                                soma_direta | personalizada
    pesos             : dict  - Pesos por avaliacao (para formula ponderada)
                                Ex: {"P1": 0.4, "P2": 0.4, "Trabalho": 0.2}
    nota_minima       : float - Nota minima para aprovacao. Padrao: 6.0
    descricao_formula : str   - Descricao em texto da formula (para personalizada)
                                Ex: "Media de P1 e P2, descartando a menor"

    Retorna
    -------
    dict com status e dados da disciplina cadastrada.
    """
    nome = nome.strip()
    if not nome:
        return {"erro": "Nome da disciplina nao pode estar vazio."}

    if formula not in FORMULAS_VALIDAS:
        formula = "media_simples"

    if pesos is None:
        pesos = {}

    # Normaliza lista de avaliacoes
    avals_normalizadas = []
    for a in avaliacoes:
        if isinstance(a, str):
            peso = pesos.get(a, 1.0)
            avals_normalizadas.append({"nome": a, "nota": None, "peso": peso})
        elif isinstance(a, dict):
            nome_aval = a.get("nome", "")
            peso      = a.get("peso", pesos.get(nome_aval, 1.0))
            avals_normalizadas.append({"nome": nome_aval, "nota": None, "peso": peso})

    dados = _carregar_notas()

    dados[nome] = {
        "formula":            formula,
        "pesos":              pesos,
        "nota_minima":        nota_minima,
        "avaliacoes":         avals_normalizadas,
        "descricao_formula":  descricao_formula,
    }

    _salvar_notas(dados)

    # Monta mensagem descritiva
    aval_nomes  = [a["nome"] for a in avals_normalizadas]
    formula_str = descricao_formula if descricao_formula else formula

    mensagem = (
        f"Disciplina cadastrada: {nome}\n"
        f"  Formula: {formula_str}\n"
        f"  Avaliacoes: {', '.join(aval_nomes)}\n"
        f"  Nota minima: {nota_minima}"
    )
    if formula == "ponderada" and pesos:
        pesos_str = ", ".join(f"{k}: {v*100:.0f}%" for k, v in pesos.items())
        mensagem += f"\n  Pesos: {pesos_str}"

    logger.info(json.dumps({
        "ferramenta": "cadastrar_disciplina",
        "entrada": {"nome": nome, "formula": formula},
        "saida": mensagem,
    }, ensure_ascii=False))

    return {
        "status":     "ok",
        "disciplina": nome,
        "dados":      dados[nome],
        "mensagem":   mensagem,
    }


def registrar_nota(
    disciplina: str,
    avaliacao: str,
    nota: float,
) -> dict:
    """
    Registra ou atualiza a nota de uma avaliacao especifica.

    Parametros
    ----------
    disciplina : str   - Nome da disciplina
    avaliacao  : str   - Nome da avaliacao (ex: "P1", "Trabalho")
    nota       : float - Nota obtida

    Retorna
    -------
    dict com status, nova media parcial e situacao atual.
    """
    dados = _carregar_notas()

    if disciplina not in dados:
        return {
            "erro": f"Disciplina '{disciplina}' nao cadastrada. "
                    "Use cadastrar_disciplina primeiro."
        }

    if nota < 0 or nota > 10:
        return {"erro": f"Nota invalida: {nota}. Deve estar entre 0 e 10."}

    disc = dados[disciplina]
    avals = disc.get("avaliacoes", [])

    # Procura a avaliacao pelo nome (case-insensitive)
    encontrada = False
    for a in avals:
        if a["nome"].lower() == avaliacao.lower():
            a["nota"] = nota
            encontrada = True
            break

    if not encontrada:
        return {
            "erro": f"Avaliacao '{avaliacao}' nao encontrada em '{disciplina}'. "
                    f"Avaliacoes disponiveis: {', '.join(a['nome'] for a in avals)}"
        }

    _salvar_notas(dados)

    media     = _calcular_media_interna(disc)
    situacao  = _formatar_situacao(media, disc.get("nota_minima", 6.0))
    sem_nota  = [a["nome"] for a in avals if a.get("nota") is None]

    media_str = f"{media:.1f}" if media is not None else "parcial"
    mensagem  = (
        f"Nota registrada: {disciplina} - {avaliacao}: {nota:.1f}\n"
        f"Media atual: {media_str}\n"
        f"Situacao: {situacao}"
    )
    if sem_nota:
        mensagem += f"\nAvaliacoes pendentes: {', '.join(sem_nota)}"

    logger.info(json.dumps({
        "ferramenta": "registrar_nota",
        "entrada": {"disciplina": disciplina, "avaliacao": avaliacao, "nota": nota},
        "saida": mensagem,
    }, ensure_ascii=False))

    return {
        "status":    "ok",
        "media":     media,
        "situacao":  situacao,
        "mensagem":  mensagem,
    }


def consultar_notas(disciplina: str = None) -> dict:
    """
    Consulta as notas de uma disciplina ou de todas.

    Parametros
    ----------
    disciplina : str, opcional - Nome da disciplina. Se None, retorna todas.

    Retorna
    -------
    dict com notas, medias e situacao de cada disciplina.
    """
    dados = _carregar_notas()

    if not dados:
        return {
            "mensagem": "Nenhuma disciplina cadastrada ainda. "
                        "Use cadastrar_disciplina para comecar."
        }

    if disciplina:
        if disciplina not in dados:
            return {"erro": f"Disciplina '{disciplina}' nao encontrada."}
        disciplinas_consultar = {disciplina: dados[disciplina]}
    else:
        disciplinas_consultar = dados

    linhas   = []
    resumo   = []

    for nome_disc, disc in disciplinas_consultar.items():
        avals      = disc.get("avaliacoes", [])
        nota_min   = disc.get("nota_minima", 6.0)
        formula    = disc.get("descricao_formula") or disc.get("formula", "media_simples")
        media      = _calcular_media_interna(disc)
        situacao   = _formatar_situacao(media, nota_min)

        linhas.append(f"\n{nome_disc} (formula: {formula}, minimo: {nota_min})")
        linhas.append("-" * 40)

        for a in avals:
            nota_str = f"{a['nota']:.1f}" if a.get("nota") is not None else "---"
            peso_str = f" (peso: {a.get('peso', 1.0)})" if disc.get("formula") == "ponderada" else ""
            linhas.append(f"  {a['nome']:<20} {nota_str}{peso_str}")

        media_str = f"{media:.1f}" if media is not None else "N/A"
        linhas.append(f"  Media: {media_str} | {situacao}")

        resumo.append({
            "disciplina": nome_disc,
            "media":      media,
            "situacao":   situacao,
        })

    mensagem = "\n".join(linhas)

    return {
        "disciplinas": resumo,
        "mensagem":    mensagem,
    }


def calcular_media(disciplina: str) -> dict:
    """
    Calcula e exibe a media atual de uma disciplina.

    Parametros
    ----------
    disciplina : str - Nome da disciplina

    Retorna
    -------
    dict com media, situacao e detalhes do calculo.
    """
    dados = _carregar_notas()

    if disciplina not in dados:
        return {"erro": f"Disciplina '{disciplina}' nao cadastrada."}

    disc      = dados[disciplina]
    media     = _calcular_media_interna(disc)
    nota_min  = disc.get("nota_minima", 6.0)
    situacao  = _formatar_situacao(media, nota_min)
    formula   = disc.get("descricao_formula") or disc.get("formula", "media_simples")

    if media is not None:
        status_emoji = "Aprovado" if media >= nota_min else "Reprovado"
        mensagem = (
            f"{disciplina}\n"
            f"  Formula: {formula}\n"
            f"  Media atual: {media:.1f}\n"
            f"  Nota minima: {nota_min}\n"
            f"  Situacao: {status_emoji}"
        )
    else:
        mensagem = (
            f"{disciplina}\n"
            f"  Ainda ha avaliacoes sem nota lancada.\n"
            f"  Media parcial indisponivel."
        )

    return {
        "disciplina": disciplina,
        "media":      media,
        "situacao":   situacao,
        "mensagem":   mensagem,
    }


def nota_necessaria(disciplina: str, avaliacao_faltante: str) -> dict:
    """
    Calcula a nota necessaria em uma avaliacao futura para ser aprovado.

    Parametros
    ----------
    disciplina         : str - Nome da disciplina
    avaliacao_faltante : str - Nome da avaliacao que ainda nao foi realizada

    Retorna
    -------
    dict com a nota necessaria e analise da situacao.
    """
    dados = _carregar_notas()

    if disciplina not in dados:
        return {"erro": f"Disciplina '{disciplina}' nao cadastrada."}

    disc      = dados[disciplina]
    avals     = disc.get("avaliacoes", [])
    nota_min  = disc.get("nota_minima", 6.0)
    formula   = disc.get("formula", "media_simples")
    pesos     = disc.get("pesos", {})

    # Valida avaliacao faltante
    nomes_avals = [a["nome"].lower() for a in avals]
    if avaliacao_faltante.lower() not in nomes_avals:
        return {
            "erro": f"Avaliacao '{avaliacao_faltante}' nao encontrada. "
                    f"Disponíveis: {', '.join(a['nome'] for a in avals)}"
        }

    # Notas ja lancadas (exceto a faltante)
    com_nota = [a for a in avals if a.get("nota") is not None
                and a["nome"].lower() != avaliacao_faltante.lower()]
    total    = len(avals)

    if formula == "media_simples":
        soma_atual    = sum(a["nota"] for a in com_nota)
        nota_necessaria_val = (nota_min * total) - soma_atual
        nota_necessaria_val = round(nota_necessaria_val, 2)

    elif formula == "ponderada":
        soma_pesos_total  = sum(pesos.get(a["nome"], a.get("peso", 1.0)) for a in avals)
        soma_pontos_atual = sum(
            a["nota"] * pesos.get(a["nome"], a.get("peso", 1.0))
            for a in com_nota
        )
        peso_faltante     = pesos.get(avaliacao_faltante, 1.0)
        if peso_faltante == 0:
            return {"erro": "Peso da avaliacao e zero — nota nao afeta a media."}
        nota_necessaria_val = round(
            (nota_min * soma_pesos_total - soma_pontos_atual) / peso_faltante, 2
        )

    elif formula in ("maior_nota", "soma_direta", "personalizada"):
        soma_atual          = sum(a["nota"] for a in com_nota)
        nota_necessaria_val = round((nota_min * total) - soma_atual, 2)

    else:
        nota_necessaria_val = nota_min

    # Analise da situacao
    if nota_necessaria_val <= 0:
        analise = f"Voce ja garantiu a aprovacao mesmo sem fazer {avaliacao_faltante}!"
    elif nota_necessaria_val > 10:
        analise = (
            f"Infelizmente nao e possivel ser aprovado mesmo com nota maxima em {avaliacao_faltante}. "
            f"Verifique se ha recuperacao disponivel."
        )
    else:
        analise = f"Voce precisa tirar pelo menos {nota_necessaria_val:.1f} em {avaliacao_faltante} para ser aprovado."

    mensagem = (
        f"{disciplina}\n"
        f"  Nota necessaria em {avaliacao_faltante}: {max(nota_necessaria_val, 0):.1f}\n"
        f"  {analise}"
    )

    logger.info(json.dumps({
        "ferramenta": "nota_necessaria",
        "entrada": {"disciplina": disciplina, "avaliacao_faltante": avaliacao_faltante},
        "saida": mensagem,
    }, ensure_ascii=False))

    return {
        "nota_necessaria": max(nota_necessaria_val, 0),
        "possivel":        nota_necessaria_val <= 10,
        "analise":         analise,
        "mensagem":        mensagem,
    }


def listar_disciplinas() -> dict:
    """
    Lista todas as disciplinas cadastradas com resumo de situacao.

    Retorna
    -------
    dict com lista de disciplinas e situacao de cada uma.
    """
    dados = _carregar_notas()

    if not dados:
        return {
            "total":     0,
            "mensagem":  "Nenhuma disciplina cadastrada ainda.",
        }

    linhas = [f"Disciplinas cadastradas ({len(dados)}):"]
    resumo = []

    for nome_disc, disc in dados.items():
        media    = _calcular_media_interna(disc)
        nota_min = disc.get("nota_minima", 6.0)
        formula  = disc.get("descricao_formula") or disc.get("formula", "media_simples")
        avals    = disc.get("avaliacoes", [])
        pendentes = sum(1 for a in avals if a.get("nota") is None)

        if media is not None:
            status = "Aprovado" if media >= nota_min else "Reprovado"
            media_str = f"{media:.1f}"
        else:
            status    = "Em andamento"
            media_str = "N/A"

        linhas.append(
            f"  {nome_disc:<30} Media: {media_str:<6} {status}"
            + (f" ({pendentes} avals pendentes)" if pendentes else "")
        )
        resumo.append({
            "disciplina": nome_disc,
            "formula":    formula,
            "media":      media,
            "status":     status,
            "pendentes":  pendentes,
        })

    return {
        "total":       len(dados),
        "disciplinas": resumo,
        "mensagem":    "\n".join(linhas),
    }


def remover_disciplina(nome: str) -> dict:
    """
    Remove uma disciplina e todas as suas notas.

    Parametros
    ----------
    nome : str - Nome da disciplina a remover

    Retorna
    -------
    dict com status da operacao.
    """
    dados = _carregar_notas()

    if nome not in dados:
        return {"erro": f"Disciplina '{nome}' nao encontrada."}

    del dados[nome]
    _salvar_notas(dados)

    mensagem = f"Disciplina '{nome}' removida com sucesso."
    logger.info(json.dumps({
        "ferramenta": "remover_disciplina",
        "entrada": {"nome": nome},
        "saida": mensagem,
    }, ensure_ascii=False))

    return {"status": "ok", "mensagem": mensagem}


# ---------------------------------------------------------------------------
# Teste rapido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    print("=== Teste: Cadastrar disciplinas ===")
    r = cadastrar_disciplina(
        nome="Verificacao e Validacao",
        avaliacoes=["P1", "P2", "Trabalho"],
        formula="ponderada",
        pesos={"P1": 0.3, "P2": 0.3, "Trabalho": 0.4},
        nota_minima=6.0,
        descricao_formula="P1 e P2 valem 30% cada, Trabalho vale 40%",
    )
    print(r["mensagem"])

    r = cadastrar_disciplina(
        nome="Inteligencia Artificial",
        avaliacoes=["Prova", "Trabalho Pratico"],
        formula="ponderada",
        pesos={"Prova": 0.4, "Trabalho Pratico": 0.6},
        nota_minima=6.0,
    )
    print(r["mensagem"])

    print("\n=== Teste: Registrar notas ===")
    registrar_nota("Verificacao e Validacao", "P1", 7.5)
    registrar_nota("Verificacao e Validacao", "Trabalho", 8.0)
    r = registrar_nota("Inteligencia Artificial", "Prova", 5.5)
    print(r["mensagem"])

    print("\n=== Teste: Consultar notas ===")
    r = consultar_notas()
    print(r["mensagem"])

    print("\n=== Teste: Nota necessaria ===")
    r = nota_necessaria("Verificacao e Validacao", "P2")
    print(r["mensagem"])

    r = nota_necessaria("Inteligencia Artificial", "Trabalho Pratico")
    print(r["mensagem"])

    print("\n=== Teste: Listar disciplinas ===")
    r = listar_disciplinas()
    print(r["mensagem"])