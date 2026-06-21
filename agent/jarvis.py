"""
agent/jarvis.py
---------------
Agente principal do JARVIS Academico.
"""

import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta

from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.agenda import consultar_agenda
from tools.tasks import (
    listar_tarefas, adicionar_tarefa, concluir_tarefa, concluir_tarefa_por_nome,
    atualizar_data_entrega, atualizar_horario_por_nome, tarefas_proximas,
    remover_tarefa, remover_tarefa_por_nome,
)
from tools.rag_tool import buscar_material_rag
from tools.notas import (
    cadastrar_disciplina, registrar_nota, consultar_notas,
    calcular_media, nota_necessaria, listar_disciplinas, remover_disciplina,
)
from tools.pdf_reader import (
    processar_pdf_com_instrucao, confirmar_importacao_agenda, confirmar_importacao_notas,
)

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "jarvis.log")

logger = logging.getLogger("jarvis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.FileHandler(_LOG_FILE, encoding="utf-8", delay=True)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Cliente API
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key="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",
    default_headers={"anthropic-version": "2023-06-01"},
)
MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Mapa de ferramentas
# ---------------------------------------------------------------------------
from agent.learning import LearningModule
_learning = LearningModule()

TOOL_MAP = {
    "consultar_agenda":              consultar_agenda,
    "listar_tarefas":                listar_tarefas,
    "adicionar_tarefa":              adicionar_tarefa,
    "concluir_tarefa":               concluir_tarefa,
    "concluir_tarefa_por_nome":      concluir_tarefa_por_nome,
    "atualizar_data_entrega":        atualizar_data_entrega,
    "atualizar_horario_por_nome":    atualizar_horario_por_nome,
    "tarefas_proximas":              tarefas_proximas,
    "remover_tarefa":                remover_tarefa,
    "remover_tarefa_por_nome":       remover_tarefa_por_nome,
    "buscar_material_rag":           buscar_material_rag,
    "gerar_exercicios":              lambda topico, quantidade=3: _learning.gerar_exercicios(topico, int(quantidade)),
    "gerar_exercicios_com_gabarito": lambda topico, quantidade=3: _learning.gerar_exercicios_com_gabarito(topico, int(quantidade)),
    "active_recall":                 lambda topico: _learning.gerar_pergunta_active_recall(topico),
    "cadastrar_disciplina":          cadastrar_disciplina,
    "registrar_nota":                registrar_nota,
    "consultar_notas":               consultar_notas,
    "calcular_media":                calcular_media,
    "nota_necessaria":               nota_necessaria,
    "listar_disciplinas":            listar_disciplinas,
    "remover_disciplina":            remover_disciplina,
    "processar_pdf":                 processar_pdf_com_instrucao,
    "confirmar_importacao_agenda":   confirmar_importacao_agenda,
    "confirmar_importacao_notas":    confirmar_importacao_notas,
}

FERRAMENTAS_RESPOSTA_LOCAL = {
    "listar_tarefas", "tarefas_proximas", "consultar_agenda",
    "adicionar_tarefa", "concluir_tarefa", "concluir_tarefa_por_nome",
    "remover_tarefa", "remover_tarefa_por_nome", "atualizar_data_entrega",
    "atualizar_horario_por_nome",
    "listar_disciplinas", "registrar_nota", "calcular_media",
    "nota_necessaria", "remover_disciplina",
}

# Palavras que indicam pedido de planejamento
PALAVRAS_PLANEJAMENTO = [
    "monte um plano", "montar um plano", "plano de estudos",
    "planejamento de estudos", "organize meu estudo",
    "roteiro de estudo", "como devo estudar",
    "o que devo priorizar", "o que priorizar hoje",
]

# Palavras que indicam acao direta — nunca tratar como planejamento
PALAVRAS_ACAO_DIRETA = [
    "adiciona", "adicionar", "cria", "criar", "coloca", "colocar",
    "remove", "remover", "apaga", "apagar", "conclui", "concluir",
    "marca", "marcar", "anota", "anotar", "registra", "registrar",
    "altera", "alterar", "muda", "mudar", "atualiza", "atualizar",
    "troca", "trocar", "corrige", "corrigir", "edita", "editar",
]


def _e_pedido_planejamento(mensagem: str) -> bool:
    msg = mensagem.lower()
    if any(p in msg for p in PALAVRAS_ACAO_DIRETA):
        return False
    return any(p in msg for p in PALAVRAS_PLANEJAMENTO)


# ---------------------------------------------------------------------------
# Contexto do banco
# ---------------------------------------------------------------------------
def _contexto_tarefas() -> str:
    try:
        tasks = listar_tarefas()
        if not tasks:
            return "TAREFAS NO BANCO: nenhuma tarefa cadastrada."
        hoje = date.today().isoformat()
        linhas = ["TAREFAS NO BANCO (use estes indices e nomes EXATOS):"]
        for i, t in enumerate(tasks):
            status = "CONCLUIDA" if t.get("concluida") else "pendente"
            prazo  = t.get("data_entrega", "sem data")
            if prazo and prazo != "sem data":
                p = prazo.split("-")
                prazo_fmt = f"{p[2]}/{p[1]}/{p[0]}"
                if not t.get("concluida") and prazo < hoje:
                    prazo_fmt += " (ATRASADA)"
            else:
                prazo_fmt = "sem data"
            linhas.append(
                f"  indice={i} | '{t['descricao']}' | {status} | "
                f"prioridade={t.get('prioridade','normal')} | entrega={prazo_fmt} {t.get('horario','23:59')}"
            )
        return "\n".join(linhas)
    except Exception:
        return ""


def _contexto_agenda() -> str:
    try:
        r = consultar_agenda()
        if isinstance(r, dict):
            eventos = r.get("eventos", [])
            if not eventos:
                return f"AGENDA HOJE: {r.get('mensagem', 'sem eventos')}"
            linhas = ["AGENDA HOJE:"]
            for e in eventos:
                linhas.append(f"  {e.get('hora','')}: {e.get('evento','')}")
            return "\n".join(linhas)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Respostas locais
# ---------------------------------------------------------------------------
def _resposta_local(tool_name: str, resultado_str: str) -> str:
    try:
        dados = json.loads(resultado_str)
    except Exception:
        return resultado_str

    hoje = date.today().isoformat()

    if tool_name in ("listar_tarefas", "tarefas_proximas"):
        if not isinstance(dados, list) or len(dados) == 0:
            return "Voce nao tem tarefas no momento."
        pendentes  = [t for t in dados if not t.get("concluida")]
        concluidas = [t for t in dados if t.get("concluida")]
        linhas = []
        if pendentes:
            linhas.append("Tarefas pendentes:")
            for i, t in enumerate(pendentes):
                linha = f"  {i+1}. {t.get('descricao', '')}"
                prazo   = t.get("data_entrega")
                horario = t.get("horario", "23:59")
                if prazo:
                    p = prazo.split("-")
                    fmt = f"{p[2]}/{p[1]}/{p[0]} {horario}"
                    if prazo < hoje:    linha += f" - ATRASADA ({fmt})"
                    elif prazo == hoje: linha += f" - VENCE HOJE ({fmt})"
                    else:               linha += f" - entrega: {fmt}"
                prio = t.get("prioridade")
                if prio and prio != "normal":
                    linha += f" [{prio}]"
                linhas.append(linha)
        if concluidas:
            linhas.append(f"\nConcluidas: {len(concluidas)} tarefa(s).")
        return "\n".join(linhas)

    if tool_name == "consultar_agenda":
        if isinstance(dados, dict):
            eventos = dados.get("eventos", [])
            if not eventos:
                return dados.get("mensagem", "Nenhum evento encontrado.")
            return "Agenda:\n" + "\n".join(
                f"  - {e.get('hora','')}: {e.get('evento','')}" for e in eventos
            )

    if tool_name == "adicionar_tarefa":
        if isinstance(dados, dict) and dados.get("status") == "ok":
            t = dados.get("tarefa", {})
            msg = f"Tarefa adicionada: {t.get('descricao', '')}"
            prazo = t.get("data_entrega")
            horario = t.get("horario", "23:59")
            if prazo:
                p = prazo.split("-")
                msg += f" (entrega: {p[2]}/{p[1]}/{p[0]} as {horario})"
            prio = t.get("prioridade")
            if prio and prio != "normal":
                msg += f" [{prio}]"
            return msg
        return dados.get("mensagem", str(dados)) if isinstance(dados, dict) else str(dados)

    if tool_name in ("concluir_tarefa", "concluir_tarefa_por_nome",
                     "remover_tarefa", "remover_tarefa_por_nome",
                     "atualizar_data_entrega", "atualizar_horario_por_nome"):
        if isinstance(dados, dict):
            return dados.get("mensagem", "Operacao realizada com sucesso.")

    if tool_name == "listar_disciplinas":
        if isinstance(dados, dict):
            return dados.get("mensagem", str(dados))
        if isinstance(dados, list):
            if not dados:
                return "Nenhuma disciplina cadastrada."
            linhas = ["Suas disciplinas:"]
            for d in dados:
                linha = f"  - {d.get('nome','')}"
                media = d.get("media_atual")
                if media is not None:
                    linha += f" | Media: {media:.1f}"
                situacao = d.get("situacao", "")
                if situacao:
                    linha += f" | {situacao}"
                linhas.append(linha)
            return "\n".join(linhas)

    if tool_name in ("registrar_nota", "calcular_media", "nota_necessaria", "remover_disciplina"):
        if isinstance(dados, dict):
            return dados.get("mensagem", "Operacao realizada com sucesso.")

    if isinstance(dados, dict):
        return dados.get("mensagem", str(dados))
    if isinstance(dados, list):
        return "\n".join(str(item) for item in dados)
    return str(dados)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def _build_system_prompt() -> str:
    hoje     = date.today().strftime("%d/%m/%Y")
    amanha   = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    hoje_iso = date.today().strftime("%Y-%m-%d")

    return f"""Voce e o JARVIS, um assistente academico inteligente e prestativo.
Hoje e {hoje} ({hoje_iso} em formato ISO).

## FERRAMENTAS DISPONÍVEIS

Quando precisar usar uma ferramenta, responda APENAS com um bloco JSON no seguinte formato,
sem nenhum texto antes ou depois:

```json
{{"tool": "nome_da_ferramenta", "args": {{"parametro": "valor"}}}}
```

### AGENDA
1. consultar_agenda - args: {{"dia": "YYYY-MM-DD"}}

### TAREFAS
2. listar_tarefas   - args: {{"filtro": "pendentes|concluidas|atrasadas|hoje"}} ou {{}}
3. adicionar_tarefa - args: {{"descricao": "texto", "data_entrega": "DD/MM/YYYY", "horario": "HH:MM", "prioridade": "alta|normal|baixa"}}
   - horario e opcional; se nao informado usa 23:59
4. concluir_tarefa_por_nome - PREFERIR. args: {{"nome": "nome EXATO da tarefa"}}
5. concluir_tarefa  - args: {{"indice": 0}}
6. atualizar_horario_por_nome - PREFERIR para alterar horario ou data. args: {{"nome": "nome da tarefa", "novo_horario": "HH:MM", "nova_data": "DD/MM/YYYY"}}
   - nova_data e opcional; se omitido mantem a data atual
7. atualizar_data_entrega - args: {{"indice": 0, "nova_data": "DD/MM/YYYY", "novo_horario": "HH:MM"}}
8. tarefas_proximas - args: {{"dias": 7}}
9. remover_tarefa_por_nome - PREFERIR. args: {{"nome": "nome EXATO da tarefa"}}
10. remover_tarefa  - args: {{"indice": 0}}

### RAG
11. buscar_material_rag - args: {{"query": "termo ou pergunta"}}

### APRENDIZADO
12. gerar_exercicios              - args: {{"topico": "nome", "quantidade": 3}}
13. gerar_exercicios_com_gabarito - args: {{"topico": "nome", "quantidade": 3}}
14. active_recall                 - args: {{"topico": "nome"}}

### NOTAS
15. cadastrar_disciplina - args: {{"nome": "nome", "avaliacoes": ["P1","P2"], "formula": "media_simples|ponderada|maior_nota|soma_direta|personalizada", "pesos": {{}}, "nota_minima": 6.0}}
16. registrar_nota       - args: {{"disciplina": "nome", "avaliacao": "P1", "nota": 7.5}}
17. consultar_notas      - args: {{"disciplina": "nome"}} ou {{}}
18. calcular_media       - args: {{"disciplina": "nome"}}
19. nota_necessaria      - args: {{"disciplina": "nome", "avaliacao_faltante": "P2"}}
20. listar_disciplinas   - args: {{}}
21. remover_disciplina   - args: {{"nome": "nome"}}

### PDF
22. processar_pdf               - args: {{"caminho": "C:/path/file.pdf", "instrucao": "o que fazer"}}
23. confirmar_importacao_agenda - args: {{"eventos": [lista]}}
24. confirmar_importacao_notas  - args: {{"disciplinas": [lista]}}

## REGRAS
- Responda APENAS com o JSON quando usar ferramenta. Sem texto antes ou depois.
- Nao use markdown nas respostas de texto.
- Use aspas retas " nao aspas curvas.
- SEMPRE use remover_tarefa_por_nome e concluir_tarefa_por_nome quando o usuario citar o nome.
- SEMPRE use atualizar_horario_por_nome quando o usuario quiser mudar horario ou data de uma tarefa pelo nome. NUNCA crie uma nova tarefa para substituir uma existente.
- Hoje e {hoje_iso}, amanha e {amanha}.
- Para pedidos de planejamento, priorizacao ou 'o que estudar', use APENAS: listar_tarefas, consultar_agenda e buscar_material_rag. NUNCA chame adicionar_tarefa durante planejamento.
"""


# ---------------------------------------------------------------------------
# Planejamento direto
# ---------------------------------------------------------------------------
def _executar_planejamento(mensagem_usuario: str, system_completo: str) -> str:
    # 1. Coleta tarefas
    try:
        tasks = listar_tarefas()
        hoje = date.today().isoformat()
        pendentes = [t for t in tasks if not t.get("concluida")] if tasks else []
        linhas_tarefas = []
        for t in pendentes:
            prazo = t.get("data_entrega", "")
            horario = t.get("horario", "23:59")
            descricao = t.get("descricao", "")
            prio = t.get("prioridade", "normal")
            sufixo = ""
            if prazo:
                p = prazo.split("-")
                fmt = f"{p[2]}/{p[1]}/{p[0]}"
                if prazo < hoje:
                    sufixo = f" [ATRASADA - {fmt}]"
                elif prazo == hoje:
                    sufixo = f" [VENCE HOJE - {fmt}]"
                else:
                    sufixo = f" [entrega: {fmt} as {horario}]"
            linhas_tarefas.append(f"- {descricao}{sufixo} (prioridade: {prio})")
        dados_tarefas = "\n".join(linhas_tarefas) if linhas_tarefas else "Nenhuma tarefa pendente."
        logger.info(json.dumps({"timestamp": datetime.now().isoformat(),
                                "ferramenta": "listar_tarefas", "entrada": {},
                                "saida": dados_tarefas[:200]}, ensure_ascii=False))
    except Exception as e:
        dados_tarefas = f"Erro ao carregar tarefas: {e}"

    # 2. Coleta agenda
    try:
        agenda = consultar_agenda()
        if isinstance(agenda, dict):
            eventos = agenda.get("eventos", [])
            if eventos:
                dados_agenda = "\n".join(f"- {e.get('hora','')}: {e.get('evento','')}" for e in eventos)
            else:
                dados_agenda = agenda.get("mensagem", "Sem eventos hoje.")
        else:
            dados_agenda = str(agenda)
        logger.info(json.dumps({"timestamp": datetime.now().isoformat(),
                                "ferramenta": "consultar_agenda", "entrada": {},
                                "saida": dados_agenda[:200]}, ensure_ascii=False))
    except Exception as e:
        dados_agenda = f"Erro ao carregar agenda: {e}"

    # 3. Coleta RAG
    topico_rag = "criterios de teste verificacao validacao software VVT"
    for t in pendentes[:3]:
        desc = t.get("descricao", "").lower()
        if any(p in desc for p in ["vvt", "teste", "software", "verificacao", "validacao"]):
            topico_rag = "criterios de teste verificacao validacao software VVT"
            break
        if any(p in desc for p in ["gcs", "configuracao", "versionamento"]):
            topico_rag = "gerencia de configuracao de software"
            break

    try:
        resultado_rag = buscar_material_rag(topico_rag, top_k=3)
        dados_rag = resultado_rag.get("contexto", "Nenhum material encontrado.")
        logger.info(json.dumps({"timestamp": datetime.now().isoformat(),
                                "ferramenta": "buscar_material_rag",
                                "entrada": {"query": topico_rag},
                                "saida": dados_rag[:200]}, ensure_ascii=False))
    except Exception as e:
        dados_rag = f"Erro ao buscar materiais: {e}"

    # 4. Sintese final
    mensagens_sintese = [
        {"role": "system", "content": system_completo},
        {
            "role": "user",
            "content": (
                f"Solicitacao do estudante: {mensagem_usuario}\n\n"
                f"TAREFAS PENDENTES:\n{dados_tarefas}\n\n"
                f"AGENDA DE HOJE:\n{dados_agenda}\n\n"
                f"MATERIAIS DE ESTUDO DISPONIVEIS:\n{dados_rag[:1500]}\n\n"
                "Com base nessas informacoes reais, gere um plano de estudos organizado por dia. "
                "Seja especifico sobre o que estudar em cada dia, considerando os prazos das tarefas. "
                "Responda em texto corrido, sem JSON, sem markdown, sem asteriscos."
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=mensagens_sintese,
            temperature=0.4,
            max_tokens=2048,
        )
        resposta = response.choices[0].message.content.strip()
        resposta = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", resposta, flags=re.DOTALL)
        resposta = re.sub(r'\{"tool".*?\}', "", resposta, flags=re.DOTALL)
        resposta = resposta.replace("**", "").replace("__", "")
        resposta = re.sub(r"^#{1,6}\s+", "", resposta, flags=re.MULTILINE)
        return resposta.strip()
    except Exception as e:
        return f"Erro ao gerar plano: {e}"


# ---------------------------------------------------------------------------
# Classe Jarvis
# ---------------------------------------------------------------------------
class Jarvis:

    def __init__(self):
        self.historico: list[dict] = []
        self.system_prompt = _build_system_prompt()
        print("JARVIS inicializado - Qwen 2.5 14B via API do professor")

    def _chamar_api(self, mensagens: list[dict]) -> str:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=mensagens,
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERRO] Falha na comunicacao com o modelo: {e}"

    def _limpar(self, texto: str) -> str:
        texto = texto.replace("**", "").replace("__", "")
        texto = re.sub(r"^#{1,6}\s+", "", texto, flags=re.MULTILINE)
        texto = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", texto, flags=re.DOTALL)
        texto = re.sub(r"```\s*\{.*?\}\s*```", "", texto, flags=re.DOTALL)
        return texto.strip()

    def _normalizar_aspas(self, texto: str) -> str:
        texto = texto.replace('\u201c', '"').replace('\u201d', '"')
        texto = texto.replace('\u2018', "'").replace('\u2019', "'")
        texto = texto.replace('\u00ab', '"').replace('\u00bb', '"')
        return texto

    def _extrair_tool_call(self, texto: str) -> dict | None:
        texto = self._normalizar_aspas(texto)

        # Tenta encontrar qualquer bloco JSON com "tool"
        padroes = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r'(\{[^{}]*"tool"[^{}]*\{[^{}]*\}[^{}]*\})',  # JSON aninhado
            r"(\{[^{}]*\"tool\"[^{}]*\})",
            r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})',
        ]
        for padrao in padroes:
            for match in re.finditer(padrao, texto, re.DOTALL):
                candidato = match.group(1).strip()
                # Tenta parse direto
                try:
                    dados = json.loads(candidato)
                    if "tool" in dados and dados["tool"] in TOOL_MAP:
                        return dados
                except json.JSONDecodeError:
                    pass
                # Tenta limpar e fazer parse novamente
                try:
                    limpo = re.sub(r'[\x00-\x1f\x7f]', '', candidato)
                    dados = json.loads(limpo)
                    if "tool" in dados and dados["tool"] in TOOL_MAP:
                        return dados
                except json.JSONDecodeError:
                    pass

        # Tentativa final: extrai manualmente tool e args
        try:
            tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', texto)
            args_match = re.search(r'"args"\s*:\s*(\{.*?\})', texto, re.DOTALL)
            if tool_match and args_match:
                tool_name = tool_match.group(1)
                args = json.loads(args_match.group(1))
                if tool_name in TOOL_MAP:
                    return {"tool": tool_name, "args": args}
        except Exception:
            pass

        return None

    def _executar_ferramenta(self, tool_call: dict) -> str:
        nome = tool_call.get("tool")
        args = tool_call.get("args", {})
        if nome not in TOOL_MAP:
            return f"[ERRO] Ferramenta '{nome}' nao encontrada."
        try:
            resultado = TOOL_MAP[nome](**args)
            resultado_str = (
                json.dumps(resultado, ensure_ascii=False, indent=2)
                if not isinstance(resultado, str) else resultado
            )
            try:
                logger.info(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "ferramenta": nome,
                    "entrada": args,
                    "saida": resultado_str[:500],
                }, ensure_ascii=False))
            except Exception:
                pass
            return resultado_str
        except TypeError as e:
            return f"[ERRO] Argumentos invalidos para '{nome}': {e}"
        except Exception as e:
            import traceback; traceback.print_exc()
            return f"[ERRO] Falha ao executar '{nome}': {e}"

    def _validar_historico(self, historico: list) -> list:
        if not historico:
            return historico
        validado = [historico[0]]
        for msg in historico[1:]:
            if msg["role"] == validado[-1]["role"]:
                validado[-1] = msg
            else:
                validado.append(msg)
        return validado

    def chat(self, mensagem_usuario: str) -> str:
        self.historico.append({"role": "user", "content": mensagem_usuario})

        contexto_banco  = _contexto_tarefas() + "\n" + _contexto_agenda()
        system_completo = self.system_prompt + "\n\n" + contexto_banco

        # Planejamento: executa diretamente sem loop de tool calling
        if _e_pedido_planejamento(mensagem_usuario):
            resposta_final = _executar_planejamento(mensagem_usuario, system_completo)
            self.historico.append({"role": "assistant", "content": resposta_final})
            return resposta_final

        # Fluxo normal
        historico_valido = self._validar_historico(self.historico)
        mensagens = [
            {"role": "system", "content": system_completo},
            *historico_valido,
        ]

        for _ in range(5):
            resposta_modelo = self._chamar_api(mensagens)
            resposta_norm   = self._normalizar_aspas(resposta_modelo)
            tool_call       = self._extrair_tool_call(resposta_norm)

            if not tool_call:
                resposta_final = self._limpar(resposta_modelo)
                self.historico.append({"role": "assistant", "content": resposta_final})
                return resposta_final

            tool_name = tool_call.get("tool", "")
            resultado = self._executar_ferramenta(tool_call)

            # CRUD: resposta local imediata
            if tool_name in FERRAMENTAS_RESPOSTA_LOCAL:
                resposta_final = _resposta_local(tool_name, resultado)
                self.historico.append({"role": "assistant", "content": resposta_final})
                return resposta_final

            # gerar_exercicios: resposta imediata
            if tool_name == "gerar_exercicios":
                try:
                    d = json.loads(resultado)
                    exercicios = d.get("exercicios", resultado)
                except Exception:
                    exercicios = resultado
                resposta_final = self._limpar(str(exercicios))
                if "Deseja ver as respostas" not in resposta_final:
                    resposta_final += "\n\nDeseja ver as respostas? Digite 'sim' para ver o gabarito."
                self.historico.append({"role": "assistant", "content": resposta_final})
                return resposta_final

            # RAG e ferramentas complexas: injeta e continua
            mensagens.append({
                "role": "assistant",
                "content": f"Executando ferramenta '{tool_name}'..."
            })
            mensagens.append({
                "role": "user",
                "content": (
                    f"[Resultado da ferramenta '{tool_name}']: {resultado}\n"
                    "Se precisar de mais informacoes, chame outra ferramenta. "
                    "Caso contrario, responda ao usuario em portugues, sem markdown."
                )
            })

        # Esgotou o loop
        resposta_final = self._limpar(self._chamar_api(mensagens))
        self.historico.append({"role": "assistant", "content": resposta_final})
        return resposta_final

    def limpar_historico(self):
        self.historico = []
        print("Historico limpo.")

    def mostrar_historico(self):
        print("\n--- Historico ---")
        for msg in self.historico:
            role = "Voce" if msg["role"] == "user" else "JARVIS"
            print(f"{role}: {msg['content'][:150]}...")
        print("-----------------\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    jarvis = Jarvis()
    print("\n" + "=" * 50)
    print("  JARVIS ACADEMICO")
    print("  Comandos: /limpar, /hist, /sair")
    print("=" * 50 + "\n")
    while True:
        try:
            entrada = input("Voce: ").strip()
            if not entrada:
                continue
            if entrada.lower() == "/sair":
                print("JARVIS: Ate logo!")
                break
            elif entrada.lower() == "/limpar":
                jarvis.limpar_historico()
                continue
            elif entrada.lower() == "/hist":
                jarvis.mostrar_historico()
                continue
            print("JARVIS: ", end="", flush=True)
            print(jarvis.chat(entrada))
            print()
        except KeyboardInterrupt:
            print("\nJARVIS: Encerrando!")
            break
        except Exception as e:
            print(f"\n[ERRO] {e}")


if __name__ == "__main__":
    main()