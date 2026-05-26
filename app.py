"""
app.py - Interface web do JARVIS usando Flask.
HTML em templates/index.html separado do Python.
"""

import os
import sys
import warnings
import logging
import sqlite3
import re
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "data_store" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from agent.jarvis import Jarvis
from tools.agenda import consultar_agenda
from tools.tasks import listar_tarefas, remover_tarefa, concluir_tarefa, DB_PATH

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
jarvis = Jarvis()

print("JARVIS Web iniciado!")
print(f"  Banco:   {DB_PATH}")
print(f"  Uploads: {UPLOAD_FOLDER}")

# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '')
    try:
        resp = jarvis.chat(msg)
    except Exception as e:
        resp = f"Erro: {e}"
    return jsonify({'response': resp})

@app.route('/limpar', methods=['POST'])
def limpar():
    jarvis.limpar_historico()
    return jsonify({'ok': True})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'status': 'erro', 'mensagem': 'Nenhum arquivo enviado.'})
    file      = request.files['file']
    instrucao = request.form.get('instrucao', 'resume o conteudo')
    if not file.filename.endswith('.pdf'):
        return jsonify({'status': 'erro', 'mensagem': 'Apenas arquivos PDF sao aceitos.'})
    caminho = str(UPLOAD_FOLDER / secure_filename(file.filename))
    file.save(caminho)
    try:
        from tools.pdf_reader import processar_pdf_com_instrucao
        resultado = processar_pdf_com_instrucao(caminho, instrucao)
        return jsonify({
            'status':              resultado.get('status', 'ok'),
            'mensagem':            resultado.get('mensagem', ''),
            'precisa_confirmacao': bool(resultado.get('acoes')),
            'tipo':                resultado.get('tipo', 'geral'),
            'dados':               resultado.get('dados', {}),
        })
    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': f'Erro ao processar PDF: {e}'})
    finally:
        try: os.remove(caminho)
        except: pass

@app.route('/agenda')
def agenda():
    try:
        r = consultar_agenda()
        if not isinstance(r, dict):
            r = {'mensagem': str(r), 'eventos': []}
        r.setdefault('eventos', [])
        try:
            for t in listar_tarefas(filtro="hoje"):
                r['eventos'].append({
                    'hora':   t.get("horario", "23:59"),
                    'evento': f"[TAREFA] {t['descricao']}",
                    'tipo':   'tarefa'
                })
            r['eventos'].sort(key=lambda e: e.get('hora', '00:00'))
        except Exception:
            pass
        if not r.get('eventos'):
            r['mensagem'] = r.get('mensagem', 'Nenhum evento hoje.')
        return jsonify(r)
    except Exception as e:
        return jsonify({'mensagem': f'Erro: {e}', 'eventos': []})

@app.route('/api/tarefas')
def api_tarefas():
    try:
        return jsonify({'tasks': listar_tarefas()})
    except Exception as e:
        return jsonify({'tasks': [], 'erro': str(e)})

@app.route('/api/tarefas/concluir', methods=['POST'])
def api_concluir():
    try: return jsonify(concluir_tarefa(request.json.get('indice', -1)))
    except Exception as e: return jsonify({'erro': str(e)})

@app.route('/api/tarefas/remover', methods=['POST'])
def api_remover():
    try: return jsonify(remover_tarefa(request.json.get('indice', -1)))
    except Exception as e: return jsonify({'erro': str(e)})

@app.route('/api/tarefas/apagar-pendentes', methods=['POST'])
def api_apagar_pendentes():
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("DELETE FROM tarefas WHERE concluida=0")
            conn.commit()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'erro': str(e)})

@app.route('/api/tarefas/apagar-todas', methods=['POST'])
def api_apagar_todas():
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("DELETE FROM tarefas")
            conn.commit()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'erro': str(e)})

@app.route('/api/confirmar_agenda', methods=['POST'])
def api_confirmar_agenda():
    try:
        from tools.pdf_reader import confirmar_importacao_agenda
        return jsonify(confirmar_importacao_agenda(request.json.get('eventos', [])))
    except Exception as e:
        return jsonify({'erro': str(e), 'mensagem': f'Erro: {e}'})

@app.route('/api/confirmar_notas', methods=['POST'])
def api_confirmar_notas():
    try:
        from tools.pdf_reader import confirmar_importacao_notas
        return jsonify(confirmar_importacao_notas(request.json.get('disciplinas', [])))
    except Exception as e:
        return jsonify({'erro': str(e), 'mensagem': f'Erro: {e}'})

@app.route('/notas')
def notas():
    try:
        from tools.notas import listar_disciplinas
        r = listar_disciplinas()
        msg = r.get('mensagem', 'Nenhuma nota cadastrada.') if isinstance(r, dict) else str(r)
        return jsonify({'mensagem': msg})
    except Exception as e:
        return jsonify({'mensagem': f'Erro: {e}'})

@app.route('/active_recall', methods=['POST'])
def active_recall():
    try:
        from agent.learning import LearningModule
        r = LearningModule().gerar_pergunta_active_recall(request.json.get('topico', ''))
        return jsonify({'pergunta': r.get('pergunta', 'Erro ao gerar pergunta.')})
    except Exception as e:
        return jsonify({'pergunta': f'Erro: {e}'})

@app.route('/avaliar_recall', methods=['POST'])
def avaliar_recall():
    try:
        from agent.learning import LearningModule
        r = LearningModule().avaliar_resposta(
            request.json.get('pergunta', ''),
            request.json.get('resposta', '')
        )
        avaliacao = r.get('avaliacao', 'Erro na avaliacao.')
        avaliacao = avaliacao.replace("**","").replace("__","")
        avaliacao = re.sub(r"^#{1,6}\s+","", avaliacao, flags=re.MULTILINE)
        return jsonify({'avaliacao': avaliacao.strip()})
    except Exception as e:
        return jsonify({'avaliacao': f'Erro: {e}'})

@app.route('/gerar_exercicios', methods=['POST'])
def gerar_exercicios():
    try:
        from agent.learning import LearningModule
        r = LearningModule().gerar_exercicios(
            request.json.get('topico', ''),
            int(request.json.get('quantidade', 3))
        )
        exercicios = r.get('exercicios','').replace(
            "\n\nDeseja ver as respostas? Digite 'sim' para ver o gabarito.", ""
        ).strip()
        return jsonify({'exercicios': exercicios})
    except Exception as e:
        return jsonify({'exercicios': f'Erro: {e}'})

@app.route('/gerar_exercicios_gabarito', methods=['POST'])
def gerar_exercicios_gabarito():
    try:
        from agent.learning import LearningModule
        r = LearningModule().gerar_exercicios_com_gabarito(
            request.json.get('topico', ''),
            int(request.json.get('quantidade', 3))
        )
        return jsonify({'exercicios': r.get('exercicios','')})
    except Exception as e:
        return jsonify({'exercicios': f'Erro: {e}'})

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\nIniciando JARVIS Web...")
    print("Acesse em: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)