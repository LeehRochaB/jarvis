"""
tools/tasks.py
--------------
Gerenciamento de tarefas usando SQLite.
"""

import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

_proj_db = Path(__file__).resolve().parent.parent / "data_store" / "jarvis.db"

if "OneDrive" in str(_proj_db):
    DB_PATH = Path(tempfile.gettempdir()) / "jarvis.db"
    print(f"[JARVIS] OneDrive detectado — banco em: {DB_PATH}")
else:
    DB_PATH = _proj_db
    print(f"[JARVIS] Banco em: {DB_PATH}")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _conectar():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn

def _inicializar():
    try:
        with _conectar() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tarefas (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    descricao    TEXT    NOT NULL,
                    concluida    INTEGER NOT NULL DEFAULT 0,
                    prioridade   TEXT    NOT NULL DEFAULT 'normal',
                    data_entrega TEXT,
                    horario      TEXT    DEFAULT '23:59',
                    criada_em    TEXT    NOT NULL,
                    concluida_em TEXT
                )
            """)
            conn.commit()
            # Adiciona coluna horario se nao existir (migracao)
            try:
                conn.execute("ALTER TABLE tarefas ADD COLUMN horario TEXT DEFAULT '23:59'")
                conn.commit()
            except Exception:
                pass  # Coluna ja existe
    except Exception as e:
        print(f"[JARVIS] Erro ao inicializar banco: {e}")

_inicializar()

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

def _normalizar_horario(horario):
    if not horario:
        return "23:59"
    horario = str(horario).strip()
    # Aceita HH:MM ou HHhMM ou HH:MM:SS
    try:
        if "h" in horario.lower():
            horario = horario.lower().replace("h", ":")
        partes = horario.split(":")
        h = int(partes[0])
        m = int(partes[1]) if len(partes) > 1 else 0
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    except Exception:
        pass
    return "23:59"

def _row_to_dict(row):
    return {
        "id":           row["id"],
        "descricao":    row["descricao"],
        "concluida":    bool(row["concluida"]),
        "prioridade":   row["prioridade"],
        "data_entrega": row["data_entrega"],
        "horario":      row["horario"] if row["horario"] else "23:59",
        "criada_em":    row["criada_em"],
        "concluida_em": row["concluida_em"],
    }

def listar_tarefas(filtro=None):
    hoje = date.today().isoformat()
    with _conectar() as conn:
        if filtro == "pendentes":
            rows = conn.execute("SELECT * FROM tarefas WHERE concluida=0 ORDER BY data_entrega, horario, id").fetchall()
        elif filtro == "concluidas":
            rows = conn.execute("SELECT * FROM tarefas WHERE concluida=1 ORDER BY id").fetchall()
        elif filtro == "atrasadas":
            rows = conn.execute(
                "SELECT * FROM tarefas WHERE concluida=0 AND data_entrega IS NOT NULL AND data_entrega < ? ORDER BY data_entrega, horario",
                (hoje,)
            ).fetchall()
        elif filtro == "hoje":
            rows = conn.execute(
                "SELECT * FROM tarefas WHERE concluida=0 AND data_entrega=? ORDER BY horario, id",
                (hoje,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tarefas ORDER BY data_entrega, horario, id").fetchall()
    return [_row_to_dict(r) for r in rows]

def adicionar_tarefa(descricao, data_entrega=None, horario=None, prioridade="normal"):
    data_iso  = _normalizar_data(data_entrega)
    hora_fmt  = _normalizar_horario(horario)
    prio = prioridade if prioridade in ("baixa", "normal", "alta") else "normal"
    with _conectar() as conn:
        cur = conn.execute(
            "INSERT INTO tarefas (descricao, concluida, prioridade, data_entrega, horario, criada_em) VALUES (?,0,?,?,?,?)",
            (descricao.strip(), prio, data_iso, hora_fmt, date.today().isoformat())
        )
        conn.commit()
        row = conn.execute("SELECT * FROM tarefas WHERE id=?", (cur.lastrowid,)).fetchone()
    tarefa = _row_to_dict(row)
    msg = f"Tarefa adicionada: '{descricao}'"
    if data_iso:
        p = data_iso.split("-")
        msg += f" - entrega: {p[2]}/{p[1]}/{p[0]} as {hora_fmt}"
    return {"status": "ok", "tarefa": tarefa, "mensagem": msg}

def concluir_tarefa(indice):
    tasks = listar_tarefas()
    if 0 <= indice < len(tasks):
        t = tasks[indice]
        with _conectar() as conn:
            conn.execute("UPDATE tarefas SET concluida=1, concluida_em=? WHERE id=?",
                         (date.today().isoformat(), t["id"]))
            conn.commit()
        return {"status": "ok", "mensagem": f"Tarefa '{t['descricao']}' concluida!"}
    return {"erro": "Indice invalido"}

def concluir_tarefa_por_nome(nome: str):
    nome_lower = nome.lower().strip()
    hoje = date.today().isoformat()
    tasks = listar_tarefas()
    for t in tasks:
        if t["descricao"].lower().strip() == nome_lower and not t["concluida"]:
            with _conectar() as conn:
                conn.execute("UPDATE tarefas SET concluida=1, concluida_em=? WHERE id=?", (hoje, t["id"]))
                conn.commit()
            return {"status": "ok", "mensagem": f"Tarefa '{t['descricao']}' concluida!"}
    for t in tasks:
        if nome_lower in t["descricao"].lower() and not t["concluida"]:
            with _conectar() as conn:
                conn.execute("UPDATE tarefas SET concluida=1, concluida_em=? WHERE id=?", (hoje, t["id"]))
                conn.commit()
            return {"status": "ok", "mensagem": f"Tarefa '{t['descricao']}' concluida!"}
    return {"erro": f"Tarefa '{nome}' nao encontrada ou ja concluida"}

def atualizar_data_entrega(indice, nova_data, novo_horario=None):
    tasks = listar_tarefas()
    if 0 <= indice < len(tasks):
        data_iso = _normalizar_data(nova_data)
        if not data_iso:
            return {"erro": f"Data invalida: '{nova_data}'"}
        hora_fmt = _normalizar_horario(novo_horario)
        t = tasks[indice]
        with _conectar() as conn:
            conn.execute("UPDATE tarefas SET data_entrega=?, horario=? WHERE id=?", (data_iso, hora_fmt, t["id"]))
            conn.commit()
        return {"status": "ok", "mensagem": f"Data de entrega atualizada para {data_iso} as {hora_fmt}"}
    return {"erro": "Indice invalido"}

def remover_tarefa(indice):
    tasks = listar_tarefas()
    if 0 <= indice < len(tasks):
        t = tasks[indice]
        with _conectar() as conn:
            conn.execute("DELETE FROM tarefas WHERE id=?", (t["id"],))
            conn.commit()
        return {"status": "ok", "mensagem": f"Tarefa '{t['descricao']}' removida"}
    return {"erro": "Indice invalido"}

def remover_tarefa_por_nome(nome: str):
    nome_lower = nome.lower().strip()
    tasks = listar_tarefas()
    for t in tasks:
        if t["descricao"].lower().strip() == nome_lower:
            with _conectar() as conn:
                conn.execute("DELETE FROM tarefas WHERE id=?", (t["id"],))
                conn.commit()
            return {"status": "ok", "mensagem": f"Tarefa '{t['descricao']}' removida"}
    for t in tasks:
        if nome_lower in t["descricao"].lower():
            with _conectar() as conn:
                conn.execute("DELETE FROM tarefas WHERE id=?", (t["id"],))
                conn.commit()
            return {"status": "ok", "mensagem": f"Tarefa '{t['descricao']}' removida"}
    return {"erro": f"Tarefa '{nome}' nao encontrada"}

def tarefas_proximas(dias=7):
    hoje = date.today()
    limite = (hoje + timedelta(days=dias)).isoformat()
    hoje_iso = hoje.isoformat()
    with _conectar() as conn:
        rows = conn.execute(
            "SELECT * FROM tarefas WHERE concluida=0 AND data_entrega BETWEEN ? AND ? ORDER BY data_entrega, horario",
            (hoje_iso, limite)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]