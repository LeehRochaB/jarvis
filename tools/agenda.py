import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

os.makedirs("logs", exist_ok=True)
os.makedirs("data_store", exist_ok=True)
logger = logging.getLogger(__name__)

AGENDA_PATH = Path(r"C:\Users\lebro\data_store\agenda.json")
TIPOS_VALIDOS = {"aula", "prova", "entrega", "reuniao", "outro"}

def _carregar_agenda():
    os.makedirs(AGENDA_PATH.parent, exist_ok=True)
    if not AGENDA_PATH.exists():
        _salvar_agenda([])
        return []
    try:
        with open(AGENDA_PATH, encoding="utf-8") as f:
            dados = json.load(f)
            return dados if isinstance(dados, list) else []
    except json.JSONDecodeError:
        return []

def _salvar_agenda(eventos):
    os.makedirs(AGENDA_PATH.parent, exist_ok=True)
    eventos_ordenados = sorted(eventos, key=lambda e: (e.get("dia",""), e.get("hora","")))
    with open(AGENDA_PATH, "w", encoding="utf-8") as f:
        json.dump(eventos_ordenados, f, ensure_ascii=False, indent=2)

def _normalizar_data(dia):
    if dia is None or str(dia).strip().lower() in ("hoje", "today"):
        return date.today().isoformat()
    dia = str(dia).strip().lower()
    if dia in ("amanha", "amanhã", "tomorrow"):
        return (date.today() + timedelta(days=1)).isoformat()
    if dia == "ontem":
        return (date.today() - timedelta(days=1)).isoformat()
    try:
        return datetime.strptime(dia, "%d/%m/%Y").date().isoformat()
    except ValueError:
        return dia

def _formatar_evento(evento):
    hora  = evento.get("hora", "-")
    nome  = evento.get("evento", "Sem descricao")
    tipo  = evento.get("tipo", "outro")
    emoji = {"aula":"📚","prova":"📝","entrega":"📤","reuniao":"👥","outro":"📌"}.get(tipo,"📌")
    return f"{emoji} {hora} - {nome} [{tipo}]"

def consultar_agenda(dia=None):
    dia_iso = _normalizar_data(dia)
    try:
        data_obj = date.fromisoformat(dia_iso)
    except ValueError:
        return {"erro": f"Data invalida: '{dia}'."}
    dias_semana = ["Segunda","Terca","Quarta","Quinta","Sexta","Sabado","Domingo"]
    nome_dia = dias_semana[data_obj.weekday()]
    agenda = _carregar_agenda()
    eventos_dia = [e for e in agenda if e.get("dia") == dia_iso]
    if eventos_dia:
        linhas = [_formatar_evento(e) for e in eventos_dia]
        mensagem = f"Agenda {nome_dia} {data_obj.strftime('%d/%m/%Y')}:\n" + "\n".join(linhas)
    else:
        mensagem = f"Agenda {nome_dia} {data_obj.strftime('%d/%m/%Y')}: Nenhum evento."
    return {"dia": dia_iso, "dia_semana": nome_dia, "total": len(eventos_dia),
            "eventos": eventos_dia, "mensagem": mensagem}

def consultar_semana(dia_base=None):
    dia_iso = _normalizar_data(dia_base)
    try:
        inicio = date.fromisoformat(dia_iso)
    except ValueError:
        return {"erro": f"Data invalida."}
    fim = inicio + timedelta(days=6)
    agenda = _carregar_agenda()
    semana = {}
    for i in range(7):
        d = (inicio + timedelta(days=i)).isoformat()
        eventos = [e for e in agenda if e.get("dia") == d]
        if eventos:
            semana[d] = eventos
    if semana:
        linhas = []
        for d, eventos in sorted(semana.items()):
            data_obj = date.fromisoformat(d)
            dias = ["Seg","Ter","Qua","Qui","Sex","Sab","Dom"]
            linhas.append(f"\n{dias[data_obj.weekday()]} {data_obj.strftime('%d/%m')}:")
            linhas += [f"  {_formatar_evento(e)}" for e in eventos]
        mensagem = f"Semana {inicio.strftime('%d/%m')} a {fim.strftime('%d/%m')}:" + "".join(linhas)
    else:
        mensagem = f"Nenhum evento de {inicio.strftime('%d/%m')} a {fim.strftime('%d/%m')}."
    return {"periodo": {"inicio": dia_iso, "fim": fim.isoformat()},
            "total": sum(len(v) for v in semana.values()),
            "por_dia": semana, "mensagem": mensagem}

def listar_proximas_provas(limite_dias=30):
    hoje = date.today()
    limite = hoje + timedelta(days=limite_dias)
    agenda = _carregar_agenda()
    provas = [e for e in agenda if e.get("tipo") == "prova"
              and hoje.isoformat() <= e.get("dia","") <= limite.isoformat()]
    if provas:
        linhas = [f"{e.get('dia')} - {_formatar_evento(e)}" for e in provas]
        mensagem = f"Provas nos proximos {limite_dias} dias:\n" + "\n".join(linhas)
    else:
        mensagem = f"Nenhuma prova nos proximos {limite_dias} dias."
    return {"total": len(provas), "provas": provas, "mensagem": mensagem}

def adicionar_evento(dia, hora, evento, tipo="outro"):
    dia_iso = _normalizar_data(dia)
    try:
        date.fromisoformat(dia_iso)
    except ValueError:
        return {"erro": f"Data invalida: '{dia}'."}
    if tipo not in TIPOS_VALIDOS:
        tipo = "outro"
    novo = {"dia": dia_iso, "hora": hora, "evento": evento.strip(), "tipo": tipo}
    agenda = _carregar_agenda()
    agenda.append(novo)
    _salvar_agenda(agenda)
    return {"status": "ok", "evento": novo, "mensagem": f"Evento adicionado: {_formatar_evento(novo)}"}