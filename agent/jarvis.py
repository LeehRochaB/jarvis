"""
agent/jarvis.py
---------------
Agente principal do JARVIS Academico.

Integra:
  - Gemma 12B via API do professor (compativel com OpenAI)
  - Tool calling via prompt estruturado
  - 11 ferramentas: agenda, tarefas, RAG, aprendizado, notas
  - Logs automaticos de cada chamada de ferramenta
  - Historico de conversa (memoria da sessao)
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
    listar_tarefas,
    adicionar_tarefa,
    concluir_tarefa,
    atualizar_data_entrega,
    tarefas_proximas,
    remover_tarefa,
)
from tools.rag_tool import buscar_material_rag
from tools.notas import (
    cadastrar_disciplina,
    registrar_nota,
    consultar_notas,
    calcular_media,
    nota_necessaria,
    listar_disciplinas,
    remover_disciplina,
)
from tools.pdf_reader import (
    processar_pdf_com_instrucao,
    confirmar_importacao_agenda,
    confirmar_importacao_notas,
)

# ---------------------------------------------------------------------------
# Configuracao de logs
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cliente da API do professor
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url="https://llm.liaufms.org/v1/gemma-3-12b-it",
    api_key="Cxt2ftLF7d3mHS2JdiFqB-eSDAQeZvFATPXPs02lV9A",
)

MODEL = "google/gemma-3-12b-it"

# ---------------------------------------------------------------------------
# Mapa de ferramentas disponiveis
# ---------------------------------------------------------------------------
from agent.learning import LearningModule
_learning = LearningModule()

TOOL_MAP = {
    # Agenda
    "consultar_agenda":              consultar_agenda,
    # Tarefas
    "listar_tarefas":                listar_tarefas,
    "adicionar_tarefa":              adicionar_tarefa,
    "concluir_tarefa":               concluir_tarefa,
    "atualizar_data_entrega":        atualizar_data_entrega,
    "tarefas_proximas":              tarefas_proximas,
    "remover_tarefa":                remover_tarefa,
    # RAG
    "buscar_material_rag":           buscar_material_rag,
    # Aprendizado
    "gerar_exercicios":              lambda topico, quantidade=3: _learning.gerar_exercicios(topico, int(quantidade)),
    "gerar_exercicios_com_gabarito": lambda topico, quantidade=3: _learning.gerar_exercicios_com_gabarito(topico, int(quantidade)),
    "active_recall":                 lambda topico: _learning.gerar_pergunta_active_recall(topico),
    # Notas
    "cadastrar_disciplina":          cadastrar_disciplina,
    "registrar_nota":                registrar_nota,
    "consultar_notas":               consultar_notas,
    "calcular_media":                calcular_media,
    "nota_necessaria":               nota_necessaria,
    "listar_disciplinas":            listar_disciplinas,
    "remover_disciplina":            remover_disciplina,
    # PDF
    "processar_pdf":               processar_pdf_com_instrucao,
    "confirmar_importacao_agenda": confirmar_importacao_agenda,
    "confirmar_importacao_notas":  confirmar_importacao_notas,
}


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

1. consultar_agenda - Consulta eventos da agenda academica por dia.
   - args: {{"dia": "YYYY-MM-DD"}}
   - Use quando: "o que tenho hoje?", "tenho aula amanha?", "agenda da semana"

### TAREFAS

2. listar_tarefas - Lista tarefas com filtro opcional.
   - args: {{"filtro": "pendentes|concluidas|atrasadas|hoje"}} ou {{}} para todas
   - Use quando: "quais minhas tarefas?", "tenho tarefas atrasadas?"

3. adicionar_tarefa - Adiciona uma nova tarefa.
   - args: {{"descricao": "texto", "data_entrega": "DD/MM/YYYY", "prioridade": "alta|normal|baixa"}}
   - data_entrega e prioridade sao opcionais
   - Use quando: "adicionar tarefa X", "lembrar de fazer X ate dia Y"

4. concluir_tarefa - Marca uma tarefa como concluida pelo indice (comeca em 0).
   - args: {{"indice": 0}}
   - Use quando: "concluir tarefa 1", "marcar como feito"

5. atualizar_data_entrega - Atualiza a data de entrega de uma tarefa.
   - args: {{"indice": 0, "nova_data": "DD/MM/YYYY"}}
   - Use quando: "mudar prazo da tarefa X para Y"

6. tarefas_proximas - Lista tarefas com entrega nos proximos N dias.
   - args: {{"dias": 7}}
   - Use quando: "o que vence essa semana?", "tarefas dos proximos 3 dias"

7. remover_tarefa - Remove permanentemente uma tarefa pelo indice.
   - args: {{"indice": 0}}
   - Use quando: "deletar tarefa X", "remover tarefa"

### MATERIAIS DE ESTUDO (RAG)

8. buscar_material_rag - Busca informacoes nos materiais de estudo indexados.
   - args: {{"query": "termo ou pergunta de busca"}}
   - Use quando: "explique X", "o que e Y?", "como funciona Z?", "resuma sobre..."

### APRENDIZADO

9. gerar_exercicios - Gera exercicios SEM gabarito sobre um topico.
   - args: {{"topico": "nome do topico", "quantidade": 3}}
   - Use quando: "crie exercicios sobre X", "gere questoes sobre Y", "exercicios de Z"
   - IMPORTANTE: gera apenas enunciados, sem respostas

10. gerar_exercicios_com_gabarito - Gera exercicios COM gabarito/respostas.
    - args: {{"topico": "nome do topico", "quantidade": 3}}
    - Use quando: usuario disse "sim" apos ver exercicios, ou pediu "com gabarito"

11. active_recall - Gera uma pergunta de revisao interativa sobre um topico.
    - args: {{"topico": "nome do topico"}}
    - Use quando: "me teste sobre X", "pergunta sobre Y", "active recall de Z"

### NOTAS ACADEMICAS

12. cadastrar_disciplina - Cadastra uma disciplina com formula de calculo flexivel.
    - args: {{
        "nome": "nome da disciplina",
        "avaliacoes": ["P1", "P2", "Trabalho"],
        "formula": "media_simples|ponderada|maior_nota|soma_direta|personalizada",
        "pesos": {{"P1": 0.3, "P2": 0.3, "Trabalho": 0.4}},
        "nota_minima": 6.0,
        "descricao_formula": "descricao em texto da formula (opcional)"
      }}
    - formulas disponiveis:
        media_simples  = todas as avaliacoes tem peso igual
        ponderada      = cada avaliacao tem peso definido pelo aluno (use pesos)
        maior_nota     = descarta a menor nota antes de calcular
        soma_direta    = soma todas as notas (ex: pontos acumulados)
        personalizada  = aluno descreve a formula em texto (descricao_formula)
    - Use quando: "cadastrar disciplina X", "adicionar materia Y", "registrar formula de calculo"
    - Exemplos de uso:
        "Cadastra VVT com P1 e P2 valendo 30% cada e trabalho 40%"
        "Adiciona Calculo com 3 provas em media simples, minimo 5"
        "Registra IA com prova valendo 40% e trabalho 60%"

13. registrar_nota - Registra ou atualiza a nota de uma avaliacao.
    - args: {{"disciplina": "nome", "avaliacao": "P1", "nota": 7.5}}
    - Use quando: "tirei X em Y", "registrar nota Z na disciplina W", "lancei nota"

14. consultar_notas - Consulta notas de uma ou todas as disciplinas.
    - args: {{"disciplina": "nome"}} ou {{}} para todas
    - Use quando: "ver minhas notas", "como estou em X?", "notas de todas as materias"

15. calcular_media - Calcula a media atual de uma disciplina.
    - args: {{"disciplina": "nome"}}
    - Use quando: "qual minha media em X?", "estou aprovado em Y?"

16. nota_necessaria - Calcula a nota minima necessaria para aprovacao.
    - args: {{"disciplina": "nome", "avaliacao_faltante": "P2"}}
    - Use quando: "quanto preciso tirar em X?", "que nota preciso na prova final?"

17. listar_disciplinas - Lista todas as disciplinas cadastradas com situacao.
    - args: {{}}
    - Use quando: "minhas materias", "situacao de todas as disciplinas", "estou aprovado em quais?"

18. remover_disciplina - Remove uma disciplina e todas as suas notas.
    - args: {{"nome": "nome da disciplina"}}
    - Use quando: "remover disciplina X", "apagar materia Y"

19. processar_pdf - Le um PDF e executa qualquer instrucao do usuario.
    - args: {{"caminho": "C:/caminho/arquivo.pdf", "instrucao": "o que fazer"}}
    - Use quando: usuario mencionar "PDF", "arquivo", "anexei", "tenho um documento"
    - instrucao exemplos:
        "adiciona as aulas na agenda"
        "cadastra as disciplinas e notas"
        "resume o conteudo"
        "extraia os exercicios"
        "quais sao os requisitos?"

20. confirmar_importacao_agenda - Confirma adicao de eventos do PDF na agenda.
    - args: {{"eventos": [lista]}}
    - Use quando: usuario disse "sim" apos processar cronograma

21. confirmar_importacao_notas - Confirma cadastro de disciplinas do PDF.
    - args: {{"disciplinas": [lista]}}
    - Use quando: usuario disse "sim" apos processar boletim de notas

## REGRAS IMPORTANTES

- Se a pergunta exigir uma ferramenta, responda APENAS com o JSON da ferramenta.
- NUNCA escreva texto como "[tool]", "chamando ferramenta" antes do JSON.
- NUNCA narre o que esta fazendo, apenas execute.
- Apos receber o resultado, responda diretamente ao usuario em portugues.
- Seja conciso, organizado e focado no aprendizado do estudante.
- Nao use asteriscos (**) para negrito nas respostas.
- Nao use markdown (##, **, __, etc) nas respostas, use texto simples.
- Para notas, sempre mostre a situacao (aprovado/reprovado) e avaliacoes pendentes.
- Para indices de tarefas, lembre que comecam em 0 (tarefa 1 = indice 0).
- Hoje e {hoje_iso} e amanha e {amanha}.
- Quando o usuario perguntar sobre nota necessaria, use a ferramenta nota_necessaria.
- Quando cadastrar disciplina com pesos, confirme os pesos recebidos na resposta.
- Ao consultar notas, mostre apenas: disciplina, notas lancadas, media atual e situacao (aprovado/reprovado).
- NAO mostre detalhes do calculo (pesos, formula) a menos que o usuario pergunte explicitamente.
- So mostre a formula e os pesos se o usuario perguntar "como e calculada?", "qual a formula?" ou similar.
"""


# ---------------------------------------------------------------------------
# Classe principal do agente
# ---------------------------------------------------------------------------
class Jarvis:
    """
    Agente JARVIS com memoria de conversa e tool calling via prompt.
    """

    def __init__(self):
        self.historico: list[dict] = []
        self.system_prompt = _build_system_prompt()
        print("JARVIS inicializado - Gemma 12B via API do professor")

    def _chamar_api(self, mensagens: list[dict]) -> str:
        """Chama a API do Gemma 12B e retorna o texto da resposta."""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=mensagens,
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Erro na API: {e}")
            return f"[ERRO] Falha na comunicacao com o modelo: {e}"

    def _limpar_formatacao(self, texto: str) -> str:
        """Remove formatacao markdown da resposta."""
        texto = texto.replace("**", "")
        texto = texto.replace("__", "")
        texto = re.sub(r"^#{1,6}\s+", "", texto, flags=re.MULTILINE)
        return texto.strip()

    def _extrair_tool_call(self, texto: str) -> dict | None:
        """
        Tenta extrair um JSON de tool call da resposta do modelo.
        Retorna dict com 'tool' e 'args' ou None se nao for um tool call.
        """
        padroes = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*\"tool\"[^{}]*\})",
        ]

        for padrao in padroes:
            match = re.search(padrao, texto, re.DOTALL)
            if match:
                try:
                    dados = json.loads(match.group(1))
                    if "tool" in dados and dados["tool"] in TOOL_MAP:
                        return dados
                except json.JSONDecodeError:
                    continue

        return None

    def _executar_ferramenta(self, tool_call: dict) -> str:
        """
        Executa a ferramenta indicada e retorna o resultado como string.
        Registra log com ferramenta, entrada e saida.
        """
        nome = tool_call.get("tool")
        args = tool_call.get("args", {})

        if nome not in TOOL_MAP:
            return f"[ERRO] Ferramenta '{nome}' nao encontrada."

        try:
            resultado = TOOL_MAP[nome](**args)
            resultado_str = (
                json.dumps(resultado, ensure_ascii=False, indent=2)
                if not isinstance(resultado, str)
                else resultado
            )

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "ferramenta": nome,
                "entrada": args,
                "saida": resultado_str[:500],
            }
            logger.info(json.dumps(log_entry, ensure_ascii=False))

            return resultado_str

        except TypeError as e:
            erro = f"[ERRO] Argumentos invalidos para '{nome}': {e}"
            logger.error(erro)
            return erro
        except Exception as e:
            erro = f"[ERRO] Falha ao executar '{nome}': {e}"
            logger.error(erro)
            return erro

    def chat(self, mensagem_usuario: str) -> str:
        """
        Processa uma mensagem do usuario e retorna a resposta do JARVIS.
        """
        self.historico.append({
            "role": "user",
            "content": mensagem_usuario,
        })

        mensagens = [
            {"role": "system", "content": self.system_prompt},
            *self.historico,
        ]

        resposta_modelo = self._chamar_api(mensagens)
        tool_call = self._extrair_tool_call(resposta_modelo)

        if tool_call:
            resultado_ferramenta = self._executar_ferramenta(tool_call)

            # Para gerar_exercicios, retorna direto sem passar pela LLM
            if tool_call.get("tool") == "gerar_exercicios":
                try:
                    resultado_dict = json.loads(resultado_ferramenta)
                    exercicios = resultado_dict.get("exercicios", resultado_ferramenta)
                except Exception:
                    exercicios = resultado_ferramenta
                resposta_final = self._limpar_formatacao(str(exercicios))
                if "Deseja ver as respostas" not in resposta_final:
                    resposta_final += "\n\nDeseja ver as respostas? Digite 'sim' para ver o gabarito."
                self.historico.append({"role": "assistant", "content": resposta_final})
                return resposta_final

            mensagens_finais = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": mensagem_usuario},
                {"role": "assistant", "content": f"Vou usar a ferramenta {tool_call['tool']}."},
                {
                    "role": "user",
                    "content": (
                        f"[Resultado da ferramenta '{tool_call['tool']}']: {resultado_ferramenta}. "
                        "Responda ao usuario em portugues de forma clara e organizada. "
                        "Nao use markdown, asteriscos ou simbolos especiais de formatacao. "
                        "Use texto simples com quebras de linha quando necessario."
                        "Responda ao usuario em portugues de forma clara e organizada. "
                        "Para notas: mostre apenas disciplina, notas e situacao. "
                        "NAO mostre pesos, formula ou detalhes de calculo a menos que o usuario pergunte."
                    ),
                },
            ]

            resposta_final = self._chamar_api(mensagens_finais)
            resposta_final = self._limpar_formatacao(resposta_final)
            self.historico.append({"role": "assistant", "content": resposta_final})
            return resposta_final

        else:
            resposta_modelo = self._limpar_formatacao(resposta_modelo)
            self.historico.append({
                "role": "assistant",
                "content": resposta_modelo,
            })
            return resposta_modelo

    def limpar_historico(self):
        """Reinicia o historico de conversa."""
        self.historico = []
        print("Historico limpo.")

    def mostrar_historico(self):
        """Exibe o historico da conversa atual."""
        print("\n--- Historico da Conversa ---")
        for msg in self.historico:
            role = "Voce" if msg["role"] == "user" else "JARVIS"
            print(f"{role}: {msg['content'][:150]}...")
        print("-----------------------------\n")


# ---------------------------------------------------------------------------
# Loop principal CLI
# ---------------------------------------------------------------------------
def main():
    jarvis = Jarvis()

    print("\n" + "=" * 55)
    print("  JARVIS ACADEMICO - Assistente Inteligente")
    print("=" * 55)
    print("  Comandos: /limpar, /hist, /sair")
    print("=" * 55 + "\n")

    while True:
        try:
            entrada = input("Voce: ").strip()

            if not entrada:
                continue

            if entrada.lower() == "/sair":
                print("JARVIS: Ate logo! Bons estudos!")
                break
            elif entrada.lower() == "/limpar":
                jarvis.limpar_historico()
                continue
            elif entrada.lower() == "/hist":
                jarvis.mostrar_historico()
                continue

            print("JARVIS: ", end="", flush=True)
            resposta = jarvis.chat(entrada)
            print(resposta)
            print()

        except KeyboardInterrupt:
            print("\nJARVIS: Encerrando... Ate logo!")
            break
        except Exception as e:
            print(f"\n[ERRO INESPERADO] {e}")
            logger.error(f"Erro no loop principal: {e}")


if __name__ == "__main__":
    main()