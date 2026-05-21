import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

TASKS_PATH = Path("data_store/tasks.json")

def _carregar_tasks():
    os.makedirs(TASKS_PATH.parent, exist_ok=True)
    if not TASKS_PATH.exists():
        _salvar([])
        return []
    try:
        with open(TASKS_PATH, encoding="utf-8") as f:
            dados = json.load(f)
            return dados if isinstance(dados, list) else []
    except json.JSONDecodeError:
        return []

def _salvar(tasks):
    os.makedirs(TASKS_PATH.parent, exist_ok=True)
    with open(TASKS_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

def _normalizar_data(dia):
    if not dia:
        return None
    dia = str(dia).strip().lower()
    if dia in ("hoje", "today"):
        return date.today().isoformat()
    if dia in ("amanha", "amanha", "tomorrow"):
        return (date.today() + timedelta(days=1)).isoformat()
    try:
        return datetime.strptime(dia, "%d/%m/%Y").date().isoformat()
    except ValueError:
        try:
            return datetime.strptime(dia, "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None

def listar_tarefas(filtro=None):
    """
    filtro: None = todas, "pendentes", "concluidas", "atrasadas", "hoje"
    """
    tasks = _carregar_tasks()
    hoje = date.today().isoformat()

    if filtro == "pendentes":
        tasks = [t for t in tasks if not t.get("concluida")]
    elif filtro == "concluidas":
        tasks = [t for t in tasks if t.get("concluida")]
    elif filtro == "atrasadas":
        tasks = [
            t for t in tasks
            if not t.get("concluida") and t.get("data_entrega") and t["data_entrega"] < hoje
        ]
    elif filtro == "hoje":
        tasks = [
            t for t in tasks
            if not t.get("concluida") and t.get("data_entrega") == hoje
        ]

    return tasks

def adicionar_tarefa(descricao, data_entrega=None, prioridade="normal"):
    """
    prioridade: "baixa", "normal", "alta"
    data_entrega: "DD/MM/YYYY", "amanha", "hoje", ou None
    """
    tasks = _carregar_tasks()
    data_iso = _normalizar_data(data_entrega)

    nova = {
        "id": len(tasks) + 1,
        "descricao": descricao.strip(),
        "concluida": False,
        "prioridade": prioridade if prioridade in ("baixa", "normal", "alta") else "normal",
        "data_entrega": data_iso,
        "criada_em": date.today().isoformat()
    }
    tasks.append(nova)
    _salvar(tasks)

    msg = f"Tarefa adicionada: '{descricao}'"
    if data_iso:
        msg += f" - entrega: {data_iso}"
    return {"status": "ok", "tarefa": nova, "mensagem": msg}

def concluir_tarefa(indice):
    tasks = _carregar_tasks()
    if 0 <= indice < len(tasks):
        tasks[indice]["concluida"] = True
        tasks[indice]["concluida_em"] = date.today().isoformat()
        _salvar(tasks)
        return {"status": "ok", "mensagem": f"Tarefa '{tasks[indice]['descricao']}' concluida!"}
    return {"erro": "Indice invalido"}

def atualizar_data_entrega(indice, nova_data):
    tasks = _carregar_tasks()
    if 0 <= indice < len(tasks):
        data_iso = _normalizar_data(nova_data)
        if not data_iso:
            return {"erro": f"Data invalida: '{nova_data}'"}
        tasks[indice]["data_entrega"] = data_iso
        _salvar(tasks)
        return {"status": "ok", "mensagem": f"Data de entrega atualizada para {data_iso}"}
    return {"erro": "Indice invalido"}

def remover_tarefa(indice):
    tasks = _carregar_tasks()
    if 0 <= indice < len(tasks):
        removida = tasks.pop(indice)
        _salvar(tasks)
        return {"status": "ok", "mensagem": f"Tarefa '{removida['descricao']}' removida"}
    return {"erro": "Indice invalido"}

def tarefas_proximas(dias=7):
    """Retorna tarefas com entrega nos proximos X dias"""
    tasks = _carregar_tasks()
    hoje = date.today()
    limite = (hoje + timedelta(days=dias)).isoformat()
    hoje_iso = hoje.isoformat()

    proximas = [
        t for t in tasks
        if not t.get("concluida")
        and t.get("data_entrega")
        and hoje_iso <= t["data_entrega"] <= limite
    ]
    proximas.sort(key=lambda t: t["data_entrega"])
    return proximas