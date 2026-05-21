"""
app.py
------
Interface web do JARVIS usando Flask.

Funcionalidades:
  - Chat principal com o JARVIS
  - Upload de PDF para processamento
  - Painel lateral com agenda e tarefas
  - Active Recall e Exercicios

Uso:
    python app.py
    Acessa em: http://localhost:5000
"""

import os
import sys
import warnings
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.makedirs("logs", exist_ok=True)
os.makedirs(r"C:\Users\lebro\data_store", exist_ok=True)

# Pasta para uploads temporarios
UPLOAD_FOLDER = r"C:\Users\lebro\data_store\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from agent.jarvis import Jarvis
from tools.agenda import consultar_agenda
from tools.tasks import listar_tarefas

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max
jarvis = Jarvis()

print("JARVIS Web iniciado!")

# ---------------------------------------------------------------------------
# HTML da interface
# ---------------------------------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JARVIS Academico</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #333; }

/* Header */
.header { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white;
          padding: 15px 24px; display: flex; align-items: center; gap: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.header h1 { font-size: 20px; font-weight: 600; }
.header p  { font-size: 12px; opacity: 0.85; }
.logo { font-size: 28px; }

/* Layout */
.container { display: flex; gap: 16px; padding: 16px; max-width: 1200px;
             margin: 0 auto; height: calc(100vh - 70px); }

/* Chat */
.chat-area { flex: 3; display: flex; flex-direction: column; gap: 10px; }

/* Tabs */
.tabs { display: flex; gap: 4px; }
.tab { padding: 8px 16px; border: none; border-radius: 20px; cursor: pointer;
       font-size: 13px; background: white; color: #666; transition: all .2s; }
.tab.active { background: #1a73e8; color: white; }
.tab:hover:not(.active) { background: #e8f0fe; color: #1a73e8; }

/* Paineis de conteudo */
.panel-content { display: none; flex: 1; flex-direction: column; gap: 10px; }
.panel-content.active { display: flex; }

/* Chat box */
#chat { flex: 1; overflow-y: auto; background: white; border-radius: 12px;
        padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); min-height: 300px; }
.msg { margin: 8px 0; display: flex; }
.msg.user  { justify-content: flex-end; }
.msg.bot   { justify-content: flex-start; }
.bubble { max-width: 75%; padding: 10px 14px; border-radius: 18px;
          font-size: 14px; line-height: 1.5; white-space: pre-wrap; }
.msg.user .bubble { background: #1a73e8; color: white; border-bottom-right-radius: 4px; }
.msg.bot  .bubble { background: #f1f3f4; color: #333; border-bottom-left-radius: 4px; }
.msg.bot.typing .bubble { color: #999; font-style: italic; }

/* Input area */
.input-row { display: flex; gap: 8px; align-items: flex-end; }
#msg { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 24px;
       font-size: 14px; resize: none; height: 44px; max-height: 120px;
       font-family: inherit; outline: none; transition: border .2s; }
#msg:focus { border-color: #1a73e8; }
.btn-send { background: #1a73e8; color: white; border: none; border-radius: 50%;
            width: 44px; height: 44px; cursor: pointer; font-size: 18px;
            display: flex; align-items: center; justify-content: center;
            transition: background .2s; flex-shrink: 0; }
.btn-send:hover { background: #1557b0; }

/* Upload area */
.upload-area { background: white; border-radius: 12px; padding: 16px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
.upload-area h3 { font-size: 14px; color: #444; margin-bottom: 10px; }
.drop-zone { border: 2px dashed #1a73e8; border-radius: 10px; padding: 20px;
             text-align: center; cursor: pointer; transition: all .2s;
             background: #f8f9ff; }
.drop-zone:hover, .drop-zone.dragover { background: #e8f0fe; border-color: #0d47a1; }
.drop-zone p { color: #666; font-size: 13px; margin-top: 6px; }
.drop-zone .icon { font-size: 28px; }
#fileInput { display: none; }
.upload-controls { margin-top: 10px; display: flex; gap: 8px; align-items: center; }
#instrucao { flex: 1; padding: 8px 12px; border: 1px solid #ddd; border-radius: 8px;
             font-size: 13px; outline: none; }
#instrucao:focus { border-color: #1a73e8; }
.btn-upload { background: #1a73e8; color: white; border: none; padding: 8px 16px;
              border-radius: 8px; cursor: pointer; font-size: 13px; white-space: nowrap; }
.btn-upload:disabled { background: #ccc; cursor: not-allowed; }
#fileName { font-size: 12px; color: #666; margin-top: 6px; }
#uploadResult { margin-top: 10px; padding: 10px; border-radius: 8px;
                font-size: 13px; display: none; white-space: pre-wrap; line-height: 1.5; }
#uploadResult.ok    { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
#uploadResult.erro  { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }

/* Sidebar */
.sidebar { flex: 1; display: flex; flex-direction: column; gap: 12px;
           min-width: 220px; max-width: 280px; overflow-y: auto; }
.side-card { background: white; border-radius: 12px; padding: 14px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
.side-card h3 { font-size: 13px; font-weight: 600; color: #444;
                margin-bottom: 8px; display: flex; justify-content: space-between; }
.side-card h3 button { background: none; border: none; cursor: pointer;
                        color: #1a73e8; font-size: 12px; padding: 0; }
.side-content { font-size: 12px; color: #555; line-height: 1.6;
                white-space: pre-wrap; max-height: 180px; overflow-y: auto; }
.btn-clear { width: 100%; padding: 6px; background: #f5f5f5; border: 1px solid #ddd;
             border-radius: 6px; cursor: pointer; font-size: 12px; color: #666;
             margin-top: 8px; }
.btn-clear:hover { background: #eee; }

/* Active Recall e Exercicios */
.form-group { margin-bottom: 10px; }
.form-group label { font-size: 12px; color: #666; display: block; margin-bottom: 4px; }
.form-input { width: 100%; padding: 8px 10px; border: 1px solid #ddd;
              border-radius: 8px; font-size: 13px; outline: none; }
.form-input:focus { border-color: #1a73e8; }
.btn-action { background: #1a73e8; color: white; border: none; padding: 8px 14px;
              border-radius: 8px; cursor: pointer; font-size: 13px; width: 100%; }
.btn-action:hover { background: #1557b0; }
.result-box { margin-top: 10px; background: #f8f9ff; border-radius: 8px;
              padding: 10px; font-size: 12px; color: #333; line-height: 1.6;
              white-space: pre-wrap; max-height: 200px; overflow-y: auto;
              display: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 4px; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">🤖</div>
  <div>
    <h1>JARVIS Academico</h1>
    <p>Assistente Inteligente para Estudantes | Gemma 12B (FACOM/UFMS)</p>
  </div>
</div>

<div class="container">

  <!-- Area principal -->
  <div class="chat-area">

    <!-- Tabs -->
    <div class="tabs">
      <button class="tab active" onclick="showTab('chat')">💬 Chat</button>
      <button class="tab" onclick="showTab('pdf')">📄 Enviar PDF</button>
      <button class="tab" onclick="showTab('recall')">🧠 Active Recall</button>
      <button class="tab" onclick="showTab('exercicios')">📝 Exercicios</button>
    </div>

    <!-- Tab: Chat -->
    <div id="tab-chat" class="panel-content active">
      <div id="chat">
        <div class="msg bot">
          <div class="bubble">Ola! Sou o JARVIS, seu assistente academico. Como posso ajudar?</div>
        </div>
      </div>
      <div class="input-row">
        <textarea id="msg" placeholder="Pergunte algo... (Enter para enviar)"
                  onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();enviar()}"></textarea>
        <button class="btn-send" onclick="enviar()">➤</button>
      </div>
      <button class="btn-clear" onclick="limpar()">🗑️ Limpar conversa</button>
    </div>

    <!-- Tab: PDF -->
    <div id="tab-pdf" class="panel-content">
      <div class="upload-area">
        <h3>📄 Enviar PDF para o JARVIS</h3>
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()"
             ondragover="dragOver(event)" ondragleave="dragLeave(event)" ondrop="dropFile(event)">
          <div class="icon">📂</div>
          <p>Clique para selecionar ou arraste um PDF aqui</p>
          <p style="font-size:11px;margin-top:4px;color:#999">Tamanho maximo: 16MB</p>
        </div>
        <input type="file" id="fileInput" accept=".pdf" onchange="fileSelected(this)">
        <div id="fileName"></div>
        <div class="upload-controls">
          <input type="text" id="instrucao" class="form-input" style="font-size:13px"
                 placeholder="O que fazer com o PDF? Ex: resume o conteudo, adiciona na agenda, cadastra notas...">
          <button class="btn-upload" id="btnUpload" onclick="enviarPDF()" disabled>Enviar</button>
        </div>
        <div id="uploadResult"></div>
      </div>

      <!-- Exemplos de instrucoes -->
      <div class="side-card">
        <h3>Exemplos de instrucoes</h3>
        <div style="font-size:12px;color:#555;line-height:2">
          <div style="cursor:pointer;color:#1a73e8" onclick="setInstrucao('Adiciona as aulas na agenda')">
            📅 Adiciona as aulas na agenda
          </div>
          <div style="cursor:pointer;color:#1a73e8" onclick="setInstrucao('Cadastra as disciplinas e notas')">
            📊 Cadastra as disciplinas e notas
          </div>
          <div style="cursor:pointer;color:#1a73e8" onclick="setInstrucao('Resume o conteudo')">
            📋 Resume o conteudo
          </div>
          <div style="cursor:pointer;color:#1a73e8" onclick="setInstrucao('Extraia os exercicios')">
            📝 Extraia os exercicios
          </div>
          <div style="cursor:pointer;color:#1a73e8" onclick="setInstrucao('Quais sao os topicos principais?')">
            🔍 Quais sao os topicos principais?
          </div>
          <div style="cursor:pointer;color:#1a73e8" onclick="setInstrucao('Liste os requisitos do documento')">
            📌 Liste os requisitos do documento
          </div>
        </div>
      </div>
    </div>

    <!-- Tab: Active Recall -->
    <div id="tab-recall" class="panel-content">
      <div class="side-card" style="flex:none">
        <h3>🧠 Active Recall</h3>
        <div class="form-group">
          <label>Topico de estudo</label>
          <input type="text" id="topicoRecall" class="form-input"
                 placeholder="Ex: teste de caixa preta, KNN, embeddings...">
        </div>
        <button class="btn-action" onclick="gerarPergunta()">Gerar pergunta</button>
        <div id="perguntaBox" class="result-box"></div>
        <div id="respostaArea" style="display:none;margin-top:10px">
          <div class="form-group">
            <label>Sua resposta</label>
            <textarea id="respostaAluno" class="form-input" rows="3"
                      placeholder="Digite sua resposta aqui..."></textarea>
          </div>
          <button class="btn-action" onclick="avaliarResposta()" style="background:#2e7d32">Avaliar resposta</button>
        </div>
        <div id="avaliacaoBox" class="result-box"></div>
      </div>
    </div>

    <!-- Tab: Exercicios -->
    <div id="tab-exercicios" class="panel-content">
      <div class="side-card" style="flex:none">
        <h3>📝 Gerar Exercicios</h3>
        <div class="form-group">
          <label>Topico</label>
          <input type="text" id="topicoEx" class="form-input"
                 placeholder="Ex: teste estrutural, piramide de testes...">
        </div>
        <div class="form-group">
          <label>Quantidade</label>
          <select id="qtdEx" class="form-input">
            <option value="3">3 exercicios</option>
            <option value="5">5 exercicios</option>
            <option value="10">10 exercicios</option>
            <option value="15">15 exercicios</option>
          </select>
        </div>
        <button class="btn-action" onclick="gerarExercicios()">Gerar exercicios</button>
        <div id="exerciciosBox" class="result-box"></div>
        <div id="gabaritoArea" style="display:none;margin-top:8px">
          <button class="btn-action" onclick="verGabarito()" style="background:#e65100">Ver gabarito</button>
        </div>
        <div id="gabaritoBox" class="result-box"></div>
      </div>
    </div>

  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <div class="side-card">
      <h3>📅 Agenda de Hoje <button onclick="carregarAgenda()">↻</button></h3>
      <div id="agendaBox" class="side-content">Carregando...</div>
    </div>

    <div class="side-card">
      <h3>✅ Tarefas Pendentes <button onclick="carregarTarefas()">↻</button></h3>
      <div id="tarefasBox" class="side-content">Carregando...</div>
    </div>

    <div class="side-card">
      <h3>📊 Minhas Notas <button onclick="carregarNotas()">↻</button></h3>
      <div id="notasBox" class="side-content">Carregando...</div>
    </div>

  </div>
</div>

<script>
let arquivoSelecionado = null;
let topicoAtualExercicios = '';
let qtdAtualExercicios = 3;

// ---------- Tabs ----------
function showTab(name) {
  document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ---------- Chat ----------
function addMsg(texto, tipo) {
  const chat = document.getElementById('chat');
  const div  = document.createElement('div');
  div.className = 'msg ' + tipo;
  div.innerHTML = `<div class="bubble">${texto.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function enviar() {
  const input = document.getElementById('msg');
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = '';
  input.style.height = '44px';

  addMsg(msg, 'user');
  const typing = addMsg('Pensando...', 'bot typing');

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const d = await r.json();
    typing.remove();
    addMsg(d.response, 'bot');
  } catch(e) {
    typing.remove();
    addMsg('Erro de conexao: ' + e, 'bot');
  }
  carregarPaineis();
}

async function limpar() {
  await fetch('/limpar', {method: 'POST'});
  document.getElementById('chat').innerHTML =
    '<div class="msg bot"><div class="bubble">Historico limpo! Como posso ajudar?</div></div>';
}

// ---------- Upload PDF ----------
function dragOver(e)  { e.preventDefault(); document.getElementById('dropZone').classList.add('dragover'); }
function dragLeave(e) { document.getElementById('dropZone').classList.remove('dragover'); }

function dropFile(e) {
  e.preventDefault();
  dragLeave(e);
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.pdf')) {
    arquivoSelecionado = file;
    document.getElementById('fileName').textContent = 'Arquivo: ' + file.name;
    document.getElementById('btnUpload').disabled = false;
  } else {
    alert('Por favor, selecione apenas arquivos PDF.');
  }
}

function fileSelected(input) {
  const file = input.files[0];
  if (file) {
    arquivoSelecionado = file;
    document.getElementById('fileName').textContent = 'Arquivo: ' + file.name;
    document.getElementById('btnUpload').disabled = false;
  }
}

function setInstrucao(texto) {
  document.getElementById('instrucao').value = texto;
}

async function enviarPDF() {
  if (!arquivoSelecionado) return;
  const instrucao = document.getElementById('instrucao').value.trim() || 'resume o conteudo';
  const resultBox = document.getElementById('uploadResult');

  resultBox.style.display = 'block';
  resultBox.className = 'result-box ok';
  resultBox.textContent = 'Processando PDF... aguarde.';

  const formData = new FormData();
  formData.append('file', arquivoSelecionado);
  formData.append('instrucao', instrucao);

  try {
    const r = await fetch('/upload_pdf', {method: 'POST', body: formData});
    const d = await r.json();

    if (d.status === 'erro') {
      resultBox.className = 'result-box erro';
      resultBox.textContent = d.mensagem;
    } else {
      resultBox.className = 'result-box ok';
      resultBox.textContent = d.mensagem;

      // Se precisar de confirmacao, adiciona ao chat
      if (d.precisa_confirmacao) {
        showTab('chat');
        document.querySelectorAll('.tab')[0].classList.add('active');
        addMsg('[PDF: ' + arquivoSelecionado.name + ']\n' + instrucao, 'user');
        addMsg(d.mensagem, 'bot');
      }
    }
  } catch(e) {
    resultBox.className = 'result-box erro';
    resultBox.textContent = 'Erro ao enviar o arquivo: ' + e;
  }
  carregarPaineis();
}

// ---------- Active Recall ----------
async function gerarPergunta() {
  const topico = document.getElementById('topicoRecall').value.trim();
  if (!topico) { alert('Informe um topico.'); return; }

  const box = document.getElementById('perguntaBox');
  box.style.display = 'block';
  box.textContent = 'Gerando pergunta...';
  document.getElementById('respostaArea').style.display = 'none';
  document.getElementById('avaliacaoBox').style.display = 'none';

  try {
    const r = await fetch('/active_recall', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topico})
    });
    const d = await r.json();
    box.textContent = d.pergunta;
    document.getElementById('respostaArea').style.display = 'block';
    document.getElementById('respostaAluno').value = '';
  } catch(e) {
    box.textContent = 'Erro: ' + e;
  }
}

async function avaliarResposta() {
  const pergunta = document.getElementById('perguntaBox').textContent;
  const resposta = document.getElementById('respostaAluno').value.trim();
  if (!resposta) { alert('Digite sua resposta.'); return; }

  const box = document.getElementById('avaliacaoBox');
  box.style.display = 'block';
  box.textContent = 'Avaliando...';

  try {
    const r = await fetch('/avaliar_recall', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pergunta, resposta})
    });
    const d = await r.json();
    box.textContent = d.avaliacao;
  } catch(e) {
    box.textContent = 'Erro: ' + e;
  }
}

// ---------- Exercicios ----------
async function gerarExercicios() {
  const topico = document.getElementById('topicoEx').value.trim();
  const qtd    = parseInt(document.getElementById('qtdEx').value);
  if (!topico) { alert('Informe um topico.'); return; }

  topicoAtualExercicios = topico;
  qtdAtualExercicios    = qtd;

  const box = document.getElementById('exerciciosBox');
  box.style.display = 'block';
  box.textContent   = 'Gerando exercicios...';
  document.getElementById('gabaritoArea').style.display = 'none';
  document.getElementById('gabaritoBox').style.display  = 'none';

  try {
    const r = await fetch('/gerar_exercicios', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topico, quantidade: qtd})
    });
    const d = await r.json();
    box.textContent = d.exercicios;
    document.getElementById('gabaritoArea').style.display = 'block';
  } catch(e) {
    box.textContent = 'Erro: ' + e;
  }
}

async function verGabarito() {
  const box = document.getElementById('gabaritoBox');
  box.style.display = 'block';
  box.textContent   = 'Gerando gabarito...';

  try {
    const r = await fetch('/gerar_exercicios_gabarito', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topico: topicoAtualExercicios, quantidade: qtdAtualExercicios})
    });
    const d = await r.json();
    box.textContent = d.exercicios;
  } catch(e) {
    box.textContent = 'Erro: ' + e;
  }
}

// ---------- Sidebar ----------
async function carregarAgenda() {
  const r = await fetch('/agenda');
  const d = await r.json();
  document.getElementById('agendaBox').textContent = d.mensagem;
}

async function carregarTarefas() {
  const r = await fetch('/tarefas');
  const d = await r.json();
  document.getElementById('tarefasBox').textContent = d.mensagem;
}

async function carregarNotas() {
  const r = await fetch('/notas');
  const d = await r.json();
  document.getElementById('notasBox').textContent = d.mensagem;
}

function carregarPaineis() {
  carregarAgenda();
  carregarTarefas();
  carregarNotas();
}

// Auto-resize textarea
document.getElementById('msg').addEventListener('input', function() {
  this.style.height = '44px';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Inicializa
carregarPaineis();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML)


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

    # Salva o arquivo temporariamente
    nome_seguro = secure_filename(file.filename)
    caminho     = os.path.join(UPLOAD_FOLDER, nome_seguro)
    file.save(caminho)

    try:
        from tools.pdf_reader import processar_pdf_com_instrucao
        resultado = processar_pdf_com_instrucao(caminho, instrucao)

        # Se precisar de confirmacao, registra no historico do jarvis
        precisa_confirmacao = bool(resultado.get('acoes'))
        if precisa_confirmacao:
            jarvis.historico.append({
                "role": "assistant",
                "content": resultado['mensagem']
            })
            # Armazena dados para confirmacao posterior
            jarvis._pdf_pendente = resultado

        return jsonify({
            'status':               resultado.get('status', 'ok'),
            'mensagem':             resultado.get('mensagem', ''),
            'precisa_confirmacao':  precisa_confirmacao,
            'tipo':                 resultado.get('tipo', 'geral'),
        })

    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': f'Erro ao processar PDF: {e}'})
    finally:
        # Remove arquivo temporario
        try:
            os.remove(caminho)
        except Exception:
            pass


@app.route('/agenda')
def agenda():
    try:
        r = consultar_agenda()
        return jsonify({'mensagem': r.get('mensagem', 'Sem eventos.')})
    except Exception as e:
        return jsonify({'mensagem': f'Erro: {e}'})


@app.route('/tarefas')
def tarefas():
    try:
        r = listar_tarefas()
        return jsonify({'mensagem': r.get('mensagem', 'Sem tarefas.')})
    except Exception as e:
        return jsonify({'mensagem': f'Erro: {e}'})


@app.route('/notas')
def notas():
    try:
        from tools.notas import listar_disciplinas
        r = listar_disciplinas()
        return jsonify({'mensagem': r.get('mensagem', 'Nenhuma disciplina cadastrada.')})
    except Exception as e:
        return jsonify({'mensagem': f'Erro: {e}'})


@app.route('/active_recall', methods=['POST'])
def active_recall():
    topico = request.json.get('topico', '')
    try:
        from agent.learning import LearningModule
        lm = LearningModule()
        r  = lm.gerar_pergunta_active_recall(topico)
        return jsonify({'pergunta': r.get('pergunta', 'Erro ao gerar pergunta.')})
    except Exception as e:
        return jsonify({'pergunta': f'Erro: {e}'})


@app.route('/avaliar_recall', methods=['POST'])
def avaliar_recall():
    pergunta = request.json.get('pergunta', '')
    resposta = request.json.get('resposta', '')
    try:
        from agent.learning import LearningModule
        lm = LearningModule()
        r  = lm.avaliar_resposta(pergunta, resposta)
        return jsonify({'avaliacao': r.get('avaliacao', 'Erro na avaliacao.')})
    except Exception as e:
        return jsonify({'avaliacao': f'Erro: {e}'})


@app.route('/gerar_exercicios', methods=['POST'])
def gerar_exercicios():
    topico    = request.json.get('topico', '')
    quantidade = int(request.json.get('quantidade', 3))
    try:
        from agent.learning import LearningModule
        lm = LearningModule()
        r  = lm.gerar_exercicios(topico, quantidade)
        exercicios = r.get('exercicios', '')
        # Remove a linha do gabarito (sera mostrada pelo botao)
        exercicios = exercicios.replace(
            "\n\nDeseja ver as respostas? Digite 'sim' para ver o gabarito.", ""
        ).strip()
        return jsonify({'exercicios': exercicios})
    except Exception as e:
        return jsonify({'exercicios': f'Erro: {e}'})


@app.route('/gerar_exercicios_gabarito', methods=['POST'])
def gerar_exercicios_gabarito():
    topico    = request.json.get('topico', '')
    quantidade = int(request.json.get('quantidade', 3))
    try:
        from agent.learning import LearningModule
        lm = LearningModule()
        r  = lm.gerar_exercicios_com_gabarito(topico, quantidade)
        return jsonify({'exercicios': r.get('exercicios', '')})
    except Exception as e:
        return jsonify({'exercicios': f'Erro: {e}'})


# ---------------------------------------------------------------------------
# Inicializacao
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\nIniciando JARVIS Web...")
    print("Acesse em: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)