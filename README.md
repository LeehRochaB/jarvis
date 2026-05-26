# 🤖 JARVIS Acadêmico

Assistente inteligente para estudantes desenvolvido como trabalho prático da disciplina de Inteligência Artificial — FACOM/UFMS (2026.1).

O sistema integra **RAG** (Retrieval-Augmented Generation), **Tool Calling** e o modelo de linguagem **Gemma 12B**, permitindo que estudantes organizem seus estudos, consultem materiais acadêmicos e melhorem seu desempenho por meio de recursos interativos.

---

## 📋 Funcionalidades

- **Consulta a materiais de estudo (RAG)** — perguntas sobre os PDFs das aulas de VVT indexados no ChromaDB
- **Agenda acadêmica** — adicionar, consultar e importar eventos a partir de PDFs
- **Lista de tarefas** — CRUD completo com SQLite, prioridades, horários e prazos
- **Planejamento de estudos** — sugestões baseadas em tarefas, agenda e materiais
- **Active Recall interativo** — o sistema gera perguntas, o usuário responde e o sistema avalia
- **Geração de exercícios** — questões sobre qualquer tópico com gabarito
- **Gerenciamento de notas** — cadastro de disciplinas, registro de notas e cálculo de média
- **Upload de PDF** — leitura e importação de agenda e notas a partir de documentos

---

## 🗂️ Estrutura do Projeto

```
jarvis/
├── agent/
│   ├── jarvis.py          # Agente principal (LLM, tool calling, histórico)
│   └── learning.py        # Active recall, exercícios e avaliação
├── data/                  # Dataset: PDFs das aulas de VVT
│   ├── Introducao ao Teste de Software (Marcio Eduardo Delamaro).pdf
│   ├── Simulacao da avaliacao 1 - vvt 2026_1 GABARITO.pdf
│   ├── VVT 2026_1 Aula 00 (2).pdf
│   ├── VVT 2026_1 Aula 01 (2).pdf
│   ├── VVT 2026_1 Aula 02 - revisao e inspecao (2).pdf
│   ├── VVT 2026_1 Aula 03 (2).pdf
│   ├── VVT 2026_1 Aula 04 (1).pdf
│   ├── VVT 2026_1 Aula 06 (1).pdf
│   ├── VVT 2026_1 Aula 07 e 08.pdf
│   └── VVT 2026_1 Aula Teste Estrutural (1).pdf
├── data_store/
│   ├── uploads/           # PDFs enviados temporariamente
│   ├── agenda.json        # Eventos acadêmicos
│   └── jarvis.db          # Banco SQLite (tarefas e notas)
├── logs/
│   └── jarvis.log         # Registro de chamadas de ferramentas
├── rag/
│   ├── chunker.py         # Divisão em chunks
│   ├── embedder.py        # Geração de embeddings
│   ├── loader.py          # Carregamento de documentos
│   └── retriever.py       # Busca por similaridade
├── templates/
│   └── index.html         # Frontend HTML/CSS/JS
├── tools/
│   ├── agenda.py          # CRUD de eventos
│   ├── notas.py           # Cadastro e cálculo de notas
│   ├── pdf_reader.py      # Leitura e importação de PDFs
│   ├── rag_tool.py        # Interface com o pipeline RAG
│   └── tasks.py           # CRUD de tarefas (SQLite)
├── app.py                 # Servidor Flask (rotas REST)
├── main.py                # CLI do JARVIS
└── requirements.txt       # Dependências Python
```

---

## ⚙️ Instalação

### Pré-requisitos

- Python 3.10+
- pip

### 1. Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO/jarvis
```

> **Importante:** Execute o projeto **fora** de pastas sincronizadas pelo OneDrive para evitar bloqueios no SQLite.

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o sistema

```bash
# Windows
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"; python app.py

# Linux / Mac
PYTHONIOENCODING=utf-8 python app.py
```

### 4. Acesse no navegador

```
http://localhost:5000
```

---

## 🔧 Configuração

A API do modelo está configurada em `agent/jarvis.py`:

```python
client = OpenAI(
    base_url="https://llm.liaufms.org/v1/gemma-3-12b-it",
    api_key="SUA_API_KEY",
)
MODEL = "google/gemma-3-12b-it"
```

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| LLM | Gemma 12B (API FACOM/UFMS) |
| Banco de dados | SQLite com WAL mode |
| Framework web | Flask |
| RAG — Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| RAG — Vector Store | ChromaDB |
| RAG — Framework | LangChain |
| Leitura de PDF | pypdf |
| Frontend | HTML5 + CSS3 + JavaScript vanilla |

---

## 🔨 Ferramentas Disponíveis

O agente possui 22 ferramentas acionadas automaticamente pela LLM:

| Categoria | Ferramentas |
|---|---|
| Agenda | `consultar_agenda` |
| Tarefas | `listar_tarefas`, `adicionar_tarefa`, `concluir_tarefa_por_nome`, `remover_tarefa_por_nome`, `atualizar_data_entrega`, `tarefas_proximas` |
| RAG | `buscar_material_rag` |
| Aprendizado | `gerar_exercicios`, `gerar_exercicios_com_gabarito`, `active_recall` |
| Notas | `cadastrar_disciplina`, `registrar_nota`, `consultar_notas`, `calcular_media`, `nota_necessaria`, `listar_disciplinas`, `remover_disciplina` |
| PDF | `processar_pdf`, `confirmar_importacao_notas`, `confirmar_importacao_agenda` |

---

## 📊 Dataset

O dataset é composto por 10 documentos das aulas de **Verificação, Validação e Teste de Software (VVT)** da FACOM/UFMS, incluindo slides das aulas (00 a 08), livro de referência (Delamaro) e simulação de avaliação com gabarito.

**Estratégia de chunking:**
- Tamanho: 500 tokens por chunk
- Overlap: 50 tokens
- Armazenamento: ChromaDB

---

## 📝 Logs

Todas as chamadas de ferramentas são registradas em `logs/jarvis.log` em formato JSON:

```json
{
  "timestamp": "2026-05-25T19:46:41",
  "ferramenta": "listar_tarefas",
  "entrada": {},
  "saida": "[...]"
}
```

---

## 🔗 Links

- 📁 [Pasta com documentos utilizados pelo assistente](https://drive.google.com/drive/folders/1-599rvg64TQhGkGtOhCUVczzrKcj3wW3?usp=drive_link)
- 📄 [Arquivo com detalhes do sistema](https://drive.google.com/file/d/107hyvMQdu2O5NKUaFLtsX2AzMABUiaU6/view?usp=drive_link)

---

## 👩‍💻 Autora

**Leticia Batista Rocha**  
Engenharia de Software — FACOM/UFMS  
2026.1
