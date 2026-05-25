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
    atualizar_data_entrega, tarefas_proximas, remover_tarefa, remover_tarefa_por_nome,
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
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("jarvis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    try:
        _handler = logging.FileHandler("logs/jarvis.log", encoding="utf-8", delay=True)
    except OSError:
        import tempfile
        _handler = logging.FileHandler(
            os.path.join(tempfile.gettempdir(), "jarvis.log"), 
            encoding="utf-8", delay=True
        )
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Cliente API
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url="https://llm.liaufms.org/v1/gemma-3-12b-it",
    api_key="Cxt2ftLF7d3mHS2JdiFqB-eSDAQeZvFATPXPs02lV9A",
)
MODEL = "google/gemma-3-12b-it"

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
    "listar_disciplinas", "registrar_nota", "calcular_media",
    "nota_necessaria", "remover_disciplina",
    "cadastrar_disciplina", "consultar_notas",  # adicionar estas duas
}

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
                f"prioridade={t.get('prioridade','normal')} | entrega={prazo_fmt}"
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
                    if prazo < hoje:   linha += f" - ATRASADA ({fmt})"
                    elif prazo == hoje: linha += f" - VENCE HOJE ({fmt})"
                    else:              linha += f" - entrega: {fmt}"
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
            return "Agenda:\n" + "\n".join(f"  - {e.get('hora','')}: {e.get('evento','')}" for e in eventos)

    if tool_name == "adicionar_tarefa":
        if isinstance(dados, dict) and dados.get("status") == "ok":
            t = dados.get("tarefa", {})
            msg = f"Tarefa adicionada: {t.get('descricao', '')}"
            prazo = t.get("data_entrega")
            if prazo:
                p = prazo.split("-")
                msg += f" (entrega: {p[2]}/{p[1]}/{p[0]})"
            prio = t.get("prioridade")
            if prio and prio != "normal":
                msg += f" [{prio}]"
            return msg
        return dados.get("mensagem", str(dados)) if isinstance(dados, dict) else str(dados)

    if tool_name in ("concluir_tarefa", "concluir_tarefa_por_nome",
                     "remover_tarefa", "remover_tarefa_por_nome", "atualizar_data_entrega"):
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
   - Exemplos de horario: "14:00", "9h30", "08:00"
4. concluir_tarefa_por_nome - PREFERIR. args: {{"nome": "nome EXATO da tarefa"}}
5. concluir_tarefa  - args: {{"indice": 0}}
6. atualizar_data_entrega - args: {{"indice": 0, "nova_data": "DD/MM/YYYY"}}
7. tarefas_proximas - args: {{"dias": 7}}
8. remover_tarefa_por_nome - PREFERIR. args: {{"nome": "nome EXATO da tarefa"}}
9. remover_tarefa   - args: {{"indice": 0}}

### RAG
10. buscar_material_rag - args: {{"query": "termo ou pergunta"}}

### APRENDIZADO
11. gerar_exercicios              - args: {{"topico": "nome", "quantidade": 3}}
12. gerar_exercicios_com_gabarito - args: {{"topico": "nome", "quantidade": 3}}
13. active_recall                 - args: {{"topico": "nome"}}

### NOTAS
14. cadastrar_disciplina - args: {{"nome": "nome", "avaliacoes": ["P1","P2"], "formula": "media_simples|ponderada|maior_nota|soma_direta|personalizada", "pesos": {{}}, "nota_minima": 6.0}}
15. registrar_nota       - args: {{"disciplina": "nome", "avaliacao": "P1", "nota": 7.5}}
16. consultar_notas      - args: {{"disciplina": "nome"}} ou {{}}
17. calcular_media       - args: {{"disciplina": "nome"}}
18. nota_necessaria      - args: {{"disciplina": "nome", "avaliacao_faltante": "P2"}}
19. listar_disciplinas   - args: {{}}
20. remover_disciplina   - args: {{"nome": "nome"}}

### PDF
21. processar_pdf               - args: {{"caminho": "C:/path/file.pdf", "instrucao": "o que fazer"}}
22. confirmar_importacao_agenda - args: {{"eventos": [lista]}}
23. confirmar_importacao_notas  - args: {{"disciplinas": [lista]}}

## REGRAS
- Responda APENAS com o JSON quando usar ferramenta. Sem texto antes ou depois.
- Nao use markdown nas respostas de texto.
- Use aspas retas " nao aspas curvas.
- SEMPRE use remover_tarefa_por_nome e concluir_tarefa_por_nome quando o usuario citar o nome.
- Hoje e {hoje_iso}, amanha e {amanha}.
"""

# ---------------------------------------------------------------------------
# Classe Jarvis
# ---------------------------------------------------------------------------
class Jarvis:

    def __init__(self):
        self.historico: list[dict] = []
        self.system_prompt = _build_system_prompt()
        print("JARVIS inicializado - Gemma 12B via API do professor")

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
        # Remove blocos json visiveis
        texto = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", texto, flags=re.DOTALL)
        texto = re.sub(r"```\s*\{.*?\}\s*```", "", texto, flags=re.DOTALL)
        return texto.strip()

    def _normalizar_aspas(self, texto: str) -> str:
        """Converte aspas tipograficas para aspas retas."""
        texto = texto.replace('\u201c', '"').replace('\u201d', '"')
        texto = texto.replace('\u2018', "'").replace('\u2019', "'")
        texto = texto.replace('\u00ab', '"').replace('\u00bb', '"')
        return texto

    def _extrair_tool_call(self, texto: str) -> dict | None:
        # Normaliza aspas antes de tentar extrair
        texto = self._normalizar_aspas(texto)

        padroes = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*\"tool\"[^{}]*\})",
            # Captura JSON sem bloco de codigo
            r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})',
        ]
        for padrao in padroes:
            match = re.search(padrao, texto, re.DOTALL)
            if match:
                try:
                    dados = json.loads(match.group(1))
                    if "tool" in dados and dados["tool"] in TOOL_MAP:
                        return dados
                    for chave in dados:
                        if chave in TOOL_MAP and isinstance(dados[chave], dict):
                            return {"tool": chave, "args": dados[chave]}
                except json.JSONDecodeError:
                    continue
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
                    "ferramenta": nome, "entrada": args,
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
        """Garante alternancia user/assistant."""
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

        contexto_banco = _contexto_tarefas() + "\n" + _contexto_agenda()
        system_completo = self.system_prompt + "\n\n" + contexto_banco
        historico_valido = self._validar_historico(self.historico)

        mensagens = [
            {"role": "system", "content": system_completo},
            *historico_valido,
        ]

        resposta_modelo = self._chamar_api(mensagens)
        # Normaliza aspas antes de extrair tool call
        resposta_modelo_norm = self._normalizar_aspas(resposta_modelo)
        tool_call = self._extrair_tool_call(resposta_modelo_norm)

        if tool_call:
            tool_name = tool_call.get("tool", "")
            resultado_ferramenta = self._executar_ferramenta(tool_call)

            if tool_name in FERRAMENTAS_RESPOSTA_LOCAL:
                resposta_final = _resposta_local(tool_name, resultado_ferramenta)
                self.historico.append({"role": "assistant", "content": resposta_final})
                return resposta_final

            if tool_name == "gerar_exercicios":
                try:
                    d = json.loads(resultado_ferramenta)
                    exercicios = d.get("exercicios", resultado_ferramenta)
                except Exception:
                    exercicios = resultado_ferramenta
                resposta_final = self._limpar(str(exercicios))
                if "Deseja ver as respostas" not in resposta_final:
                    resposta_final += "\n\nDeseja ver as respostas? Digite 'sim' para ver o gabarito."
                self.historico.append({"role": "assistant", "content": resposta_final})
                return resposta_final

            mensagens_finais = [
                {"role": "system", "content": system_completo},
                {"role": "user", "content": mensagem_usuario},
                {"role": "assistant", "content": "Executando ferramenta..."},
                {
                    "role": "user",
                    "content": (
                        f"[Resultado da ferramenta '{tool_name}']: {resultado_ferramenta}. "
                        "Responda ao usuario em portugues. "
                        "Nao use markdown, asteriscos ou blocos de codigo. "
                        "Texto simples apenas."
                    ),
                },
            ]
            resposta_final = self._limpar(self._chamar_api(mensagens_finais))
            self.historico.append({"role": "assistant", "content": resposta_final})
            return resposta_final

        else:
            # Se a resposta parece ser um tool call mas nao foi reconhecido,
            # tenta extrair manualmente qualquer JSON com nome de ferramenta
            resposta_modelo = self._limpar(resposta_modelo)
            self.historico.append({"role": "assistant", "content": resposta_modelo})
            return resposta_modelo

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
            if not entrada: continue
            if entrada.lower() == "/sair":
                print("JARVIS: Ate logo!"); break
            elif entrada.lower() == "/limpar":
                jarvis.limpar_historico(); continue
            elif entrada.lower() == "/hist":
                jarvis.mostrar_historico(); continue
            print("JARVIS: ", end="", flush=True)
            print(jarvis.chat(entrada))
            print()
        except KeyboardInterrupt:
            print("\nJARVIS: Encerrando!"); break
        except Exception as e:
            print(f"\n[ERRO] {e}")

if __name__ == "__main__":
    main()