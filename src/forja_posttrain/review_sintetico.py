"""Dataset sintético de review: bugs injetados em código Python, com gabarito exato.

Por que sintético: em CPU, destilar o Gemma 26B custa 12-22 min por PR, então centenas de exemplos
levariam dias. Injetar o bug dá o gabarito de graça e exato: a linha, a classe do bug e a correção.

Cada template é um trecho de código com duas versões, a com bug e a corrigida. A versão corrigida entra
no treino como exemplo negativo ("comentarios": []), para o modelo não acusar todo `requests.get`. O
split é por template: os templates de teste nunca aparecem no treino, então o teste mede se o modelo
generaliza para código novo de uma classe de bug conhecida (não se decorou o template).

Sintaxe dos templates (uma linha de código por linha; `$campo` etc. vêm do domínio sorteado):
    @@+k!|código   só na versão com o bug `k`, e é a linha comentada
    @@+k|código    só na versão com o bug `k`
    @@-k|código    só na versão sem o bug `k` (a correção)
    @@!k|código    nas duas versões; comentada quando o bug `k` está ativo
"""

from __future__ import annotations

import ast
import difflib
import json
import random
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from string import Template
from typing import Any

# --------------------------------------------------------------------------- domínios

DOMINIOS: list[dict[str, str]] = [
    dict(col="pedidos", item="pedido", campo="valor", tab="pedidos", chave="cliente_id", recurso="pedidos"),
    dict(col="alunos", item="aluno", campo="nota", tab="alunos", chave="turma", recurso="alunos"),
    dict(col="leituras", item="leitura", campo="temperatura", tab="leituras", chave="sensor_id", recurso="sensores"),
    dict(col="produtos", item="produto", campo="preco", tab="produtos", chave="categoria", recurso="produtos"),
    dict(col="faturas", item="fatura", campo="total", tab="faturas", chave="empresa_id", recurso="faturas"),
    dict(col="viagens", item="viagem", campo="distancia", tab="viagens", chave="motorista_id", recurso="viagens"),
    dict(col="funcionarios", item="funcionario", campo="salario", tab="funcionarios", chave="setor", recurso="rh"),
    dict(col="tickets", item="ticket", campo="prioridade", tab="tickets", chave="responsavel", recurso="suporte"),
    dict(col="partidas", item="partida", campo="pontos", tab="partidas", chave="jogador_id", recurso="jogos"),
    dict(col="consultas", item="consulta", campo="duracao", tab="consultas", chave="medico_id", recurso="agenda"),
    dict(col="orders", item="order", campo="amount", tab="orders", chave="customer_id", recurso="orders"),
    dict(col="readings", item="reading", campo="value", tab="readings", chave="device_id", recurso="devices"),
    dict(col="invoices", item="invoice", campo="balance", tab="invoices", chave="account_id", recurso="billing"),
    dict(col="students", item="student", campo="score", tab="students", chave="classroom", recurso="school"),
]
for _d in DOMINIOS:
    _d["Ent"] = _d["item"].capitalize()

PASTAS = ["app", "svc", "core", "api", "src/servicos", "lib", "backend", "jobs"]

# --------------------------------------------------------------------------- templates


@dataclass(frozen=True)
class Bug:
    severidade: str
    frases: tuple[str, ...]  # variações do comentário; uma é sorteada
    palavras_chave: tuple[str, ...]  # para o checador de recall (o comentário fala do problema?)


@dataclass(frozen=True)
class Tpl:
    classe: str
    nome: str
    desc: str  # pedaço do resumo: "calcula a média de $campo"
    codigo: str
    bugs: dict[str, Bug]
    imports: tuple[str, ...] = ()
    imports_fix: tuple[str, ...] = ()  # só na versão corrigida (ex.: contextlib.closing)
    teste: bool = False


KW_ZERO = ("vazi", "zero", "zerodivision")
KW_MUT = ("mutavel", "mutável", "padrao", "padrão", "default", "none", "compartilhad")
KW_FECHA = ("with", "fecha", "close", "fechad", "vaza", "release")
KW_SQL = ("injec", "injeç", "parametr", "placeholder")
KW_TIMEOUT = ("timeout",)
KW_STATUS = ("raise_for_status", "status")
KW_OFF = ("indexerror", "fora do intervalo", "off-by-one", "- 1", "ultim", "últim")
KW_IS = ("identidade", "==", "not ", "is ")
KW_ITER = ("iter", "pula", "nova lista", "cópia", "copia", "compreens", "runtimeerror", "tamanho")
KW_EXC = ("except", "engol", "silenc", "esconde", "log")
KW_CMD = ("shell", "injec", "injeç", "lista de argumentos", "subprocess")
KW_INV = ("invertid", "contrário", "contrario", ">=", "<=", " < ", " > ")
KW_PAG = ("ceil", "última", "ultima", "arredond", "trunca", "sobra")

T: list[Tpl] = []


def tpl(classe: str, nome: str, desc: str, codigo: str, bugs: dict[str, Bug], **kw: Any) -> None:
    T.append(Tpl(classe, nome, desc, codigo.strip("\n"), bugs, **kw))


# 1. divisão por zero / lista vazia ------------------------------------------------------------
tpl("divisao_vazia", "media", "calcula a média de `$campo`", """
def media_$campo($col):
    total = sum(x["$campo"] for x in $col)
@@-a|    if not $col:
@@-a|        return 0.0
@@!a|    return total / len($col)
""", {"a": Bug("bug", (
    "Se `$col` estiver vazia, `len($col)` é 0 e a divisão levanta `ZeroDivisionError`. Trate o caso vazio antes: `if not $col: return 0.0`.",
    "Divisão por zero quando `$col` vem vazia: `len($col)` é 0. Retorne um valor padrão (0.0 ou None) se não houver itens.",
), KW_ZERO)})
tpl("divisao_vazia", "percentual", "calcula o percentual de itens acima de um limite", """
def percentual_acima($col, limite):
    acima = [x for x in $col if x["$campo"] > limite]
@@-a|    if not $col:
@@-a|        return 0.0
@@!a|    return 100 * len(acima) / len($col)
""", {"a": Bug("bug", (
    "Com `$col` vazia a divisão por `len($col)` levanta `ZeroDivisionError`. Verifique `if not $col: return 0.0` antes.",
    "`len($col)` pode ser 0 e aí o percentual quebra com `ZeroDivisionError`. Trate a lista vazia antes de dividir.",
), KW_ZERO)})
tpl("divisao_vazia", "taxa", "calcula a taxa de aprovação", """
def taxa_aprovacao(aprovados, total):
@@-a|    if total == 0:
@@-a|        return 0.0
@@!a|    return round(aprovados / total, 2)
""", {"a": Bug("bug", (
    "`total` pode ser 0 (nenhum $item no período) e a divisão levanta `ZeroDivisionError`. Teste `if total == 0` antes de dividir.",
    "Divisão por zero quando `total` é 0. Devolva 0.0 (ou None) nesse caso em vez de dividir.",
), KW_ZERO)})
tpl("divisao_vazia", "maximo", "devolve o $item com maior `$campo`", """
def maior_$campo($col):
@@-a|    if not $col:
@@-a|        return None
@@!a|    return max($col, key=lambda x: x["$campo"])
""", {"a": Bug("risco", (
    "`max` de uma sequência vazia levanta `ValueError`. Se `$col` pode vir vazia, trate antes ou use `max($col, key=..., default=None)`.",
), ("vazi", "valueerror", "default"))})
tpl("divisao_vazia", "media_filtrada", "calcula a média de `$campo` por `$chave`", """
def media_por_$chave($col, valor):
    selecionados = [x["$campo"] for x in $col if x["$chave"] == valor]
@@-a|    if not selecionados:
@@-a|        return None
@@!a|    return sum(selecionados) / len(selecionados)
""", {"a": Bug("bug", (
    "Se nenhum $item tiver esse `$chave`, `selecionados` fica vazia e a divisão levanta `ZeroDivisionError`. Devolva `None` (ou 0.0) quando a lista estiver vazia.",
), KW_ZERO)}, teste=True)

# 2. argumento padrão mutável ------------------------------------------------------------------
tpl("padrao_mutavel", "historico", "guarda os últimos eventos de um $item", """
@@+a!|def registrar_evento(evento, historico=[]):
@@-a|def registrar_evento(evento, historico=None):
@@-a|    if historico is None:
@@-a|        historico = []
    historico.append(evento)
    return historico[-10:]
""", {"a": Bug("bug", (
    "`historico=[]` nasce uma única vez, quando o `def` roda; toda chamada sem `historico` grava na mesma lista e eventos de um $item aparecem no de outro. Troque o padrão por `None` e crie `[]` no corpo.",
    "Padrão mutável: a lista `historico` é a mesma em todas as chamadas que não passam o argumento. Use `historico=None` e `if historico is None: historico = []`.",
), KW_MUT)})
tpl("padrao_mutavel", "indice", "indexa $col por id", """
@@+a!|def indexar_$col($col, indice={}):
@@-a|def indexar_$col($col, indice=None):
@@-a|    indice = {} if indice is None else indice
    for x in $col:
        indice[x["id"]] = x
    return indice
""", {"a": Bug("bug", (
    "`indice={}` é avaliado uma vez e compartilhado entre chamadas: a segunda chamada devolve também os $col da primeira. Use `indice=None` e crie o dicionário dentro da função.",
    "Dicionário padrão mutável: o índice vaza de uma chamada para a outra. Use `None` como padrão.",
), KW_MUT)})
tpl("padrao_mutavel", "classe", "cria uma fila de $col", """
class Fila$Ent:
@@+a!|    def __init__(self, itens=[]):
@@+a|        self.itens = itens
@@-a|    def __init__(self, itens=None):
@@-a|        self.itens = list(itens) if itens else []

    def adicionar(self, x):
        self.itens.append(x)
""", {"a": Bug("bug", (
    "`itens=[]` é avaliado uma vez: todas as instâncias criadas sem argumento compartilham a mesma lista, e o que se adiciona numa aparece nas outras. Use `itens=None` e `self.itens = list(itens or [])`.",
), KW_MUT)})
tpl("padrao_mutavel", "set", "registra visitas de usuários", """
@@+a!|def registrar_visita(usuario, vistos=set()):
@@-a|def registrar_visita(usuario, vistos=None):
@@-a|    vistos = set() if vistos is None else vistos
    novo = usuario not in vistos
    vistos.add(usuario)
    return novo
""", {"a": Bug("bug", (
    "O `set()` padrão é criado uma vez e compartilhado: depois da primeira chamada, todo usuário parece já visto em chamadas seguintes. Use `vistos=None` e crie o conjunto dentro da função.",
), KW_MUT)}, teste=True)

# 3. recurso não fechado ----------------------------------------------------------------------
tpl("recurso_aberto", "arquivo", "lê $col de um arquivo texto", """
def carregar_$col(caminho):
@@+a!|    f = open(caminho, encoding="utf-8")
@@+a|    linhas = f.readlines()
@@-a|    with open(caminho, encoding="utf-8") as f:
@@-a|        linhas = f.readlines()
    return [linha.strip() for linha in linhas if linha.strip()]
""", {"a": Bug("risco", (
    "O arquivo aberto com `open` nunca é fechado: o descritor vaza (e num erro fica aberto). Use `with open(caminho, encoding=\"utf-8\") as f:`.",
    "Falta fechar o arquivo. Use um bloco `with open(...) as f:` para fechar mesmo se der erro.",
), KW_FECHA)})
tpl("recurso_aberto", "csv", "exporta $col para CSV", """
def exportar_$col($col, caminho):
@@+a!|    saida = open(caminho, "w", newline="")
@@+a|    escritor = csv.writer(saida)
@@+a|    for x in $col:
@@+a|        escritor.writerow([x["$chave"], x["$campo"]])
@@-a|    with open(caminho, "w", newline="") as saida:
@@-a|        escritor = csv.writer(saida)
@@-a|        for x in $col:
@@-a|            escritor.writerow([x["$chave"], x["$campo"]])
""", {"a": Bug("bug", (
    "O arquivo `saida` nunca é fechado, então o buffer pode não ser gravado no disco (o CSV sai cortado). Use `with open(caminho, \"w\", newline=\"\") as saida:`.",
), KW_FECHA)}, imports=("import csv",))
tpl("recurso_aberto", "sqlite", "conta os registros de `$tab`", """
def total_$tab(db_path):
@@+a!|    conn = sqlite3.connect(db_path)
@@+a|    linha = conn.execute("SELECT COUNT(*) FROM $tab").fetchone()
@@-a|    with closing(sqlite3.connect(db_path)) as conn:
@@-a|        linha = conn.execute("SELECT COUNT(*) FROM $tab").fetchone()
    return linha[0]
""", {"a": Bug("risco", (
    "A conexão `conn` nunca é fechada. Use `with closing(sqlite3.connect(db_path)) as conn:` (de `contextlib`) ou feche num `finally`.",
    "Conexão SQLite aberta e não fechada: cada chamada deixa uma conexão pendurada. Feche com `closing(...)` ou `conn.close()` num `finally`.",
), KW_FECHA)}, imports=("import sqlite3",), imports_fix=("from contextlib import closing",))
tpl("recurso_aberto", "lock", "conta registros com um lock", """
_lock = threading.Lock()
_contador = {"total": 0}


def registrar_$item():
@@+a!|    _lock.acquire()
@@+a|    _contador["total"] += 1
@@-a|    with _lock:
@@-a|        _contador["total"] += 1
    return _contador["total"]
""", {"a": Bug("bug", (
    "`_lock.acquire()` sem `release()`: a segunda chamada trava para sempre, e uma exceção no meio também deixa o lock preso. Use `with _lock:`.",
), KW_FECHA + ("lock", "trava"))}, imports=("import threading",), teste=True)

# 4. SQL montado com string -------------------------------------------------------------------
tpl("sql_injecao", "fstring", "busca $col por `$chave`", """
def buscar_por_$chave(conn, valor):
@@+a!|    sql = f"SELECT * FROM $tab WHERE $chave = '{valor}'"
@@+a|    return conn.execute(sql).fetchall()
@@-a|    sql = "SELECT * FROM $tab WHERE $chave = ?"
@@-a|    return conn.execute(sql, (valor,)).fetchall()
""", {"a": Bug("bug", (
    "A consulta é montada com f-string a partir de `valor`: algo como `' OR '1'='1` muda o SQL (injeção). Use parâmetro: `conn.execute(\"SELECT * FROM $tab WHERE $chave = ?\", (valor,))`.",
    "SQL injection: `valor` entra direto no texto da query. Passe como parâmetro (`?`) em vez de interpolar.",
), KW_SQL)})
tpl("sql_injecao", "concat", "remove um $item pelo id", """
def remover_$item(cur, id_):
@@+a!|    cur.execute("DELETE FROM $tab WHERE id = " + str(id_))
@@-a|    cur.execute("DELETE FROM $tab WHERE id = %s", (id_,))
    return cur.rowcount
""", {"a": Bug("bug", (
    "Concatenar `id_` no SQL permite injeção (ex.: `1 OR 1=1` apaga a tabela inteira). Use parâmetro: `cur.execute(\"DELETE FROM $tab WHERE id = %s\", (id_,))`.",
), KW_SQL)})
tpl("sql_injecao", "percent", "atualiza `$campo` de um $item", """
def atualizar_$campo(cur, id_, novo):
@@+a!|    cur.execute("UPDATE $tab SET $campo = %s WHERE id = %s" % (novo, id_))
@@-a|    cur.execute("UPDATE $tab SET $campo = %s WHERE id = %s", (novo, id_))
""", {"a": Bug("bug", (
    "O `%` formata a string antes de chegar no driver, então os valores entram crus no SQL (injeção e erro com aspas). Passe a tupla como segundo argumento: `cur.execute(\"UPDATE ... %s ...\", (novo, id_))`.",
), KW_SQL)})
tpl("sql_injecao", "format", "pesquisa $col por nome", """
def pesquisar_$col(conn, termo):
@@+a!|    sql = "SELECT id FROM $tab WHERE nome LIKE '%{}%'".format(termo)
@@+a|    return [r[0] for r in conn.execute(sql)]
@@-a|    sql = "SELECT id FROM $tab WHERE nome LIKE ?"
@@-a|    return [r[0] for r in conn.execute(sql, (f"%{termo}%",))]
""", {"a": Bug("bug", (
    "`termo` vem do usuário e entra no SQL via `.format`: injeção. Use placeholder e passe o padrão como parâmetro: `conn.execute(\"... LIKE ?\", (f\"%{termo}%\",))`.",
), KW_SQL)}, teste=True)

# 5. rede sem timeout / sem status ------------------------------------------------------------
_RT = Bug("risco", (
    "Chamada HTTP sem `timeout`: se o servidor não responder, a função espera para sempre e prende o worker. Passe `timeout=10`.",
    "Sem `timeout`, um servidor lento trava esta chamada indefinidamente. Defina `timeout=` (ex.: 10 s).",
), KW_TIMEOUT)
_RS = Bug("risco", (
    "A resposta é usada sem checar o status: um 404/500 vira um `.json()` com a página de erro (ou `JSONDecodeError`). Chame `resp.raise_for_status()` antes.",
    "Não verifica o status HTTP antes de ler o corpo. Use `resp.raise_for_status()` para falhar com uma mensagem clara.",
), KW_STATUS)
tpl("rede", "get", "busca um $item na API", """
def buscar_$item(id_):
@@+t!|    resp = requests.get(f"https://api.interna.local/$recurso/{id_}")
@@-t|    resp = requests.get(f"https://api.interna.local/$recurso/{id_}", timeout=10)
@@-s|    resp.raise_for_status()
@@!s|    return resp.json()
""", {"t": _RT, "s": _RS}, imports=("import requests",))
tpl("rede", "post", "envia um $item para a API", """
def enviar_$item(payload):
@@+t!|    resp = requests.post("https://api.interna.local/$recurso", json=payload)
@@-t|    resp = requests.post("https://api.interna.local/$recurso", json=payload, timeout=15)
@@-s|    resp.raise_for_status()
@@!s|    return resp.json()["id"]
""", {"t": _RT, "s": _RS}, imports=("import requests",))
tpl("rede", "urlopen", "baixa a lista de $col", """
def baixar_$col(url):
@@+t!|    with urllib.request.urlopen(url) as resp:
@@-t|    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())
""", {"t": Bug("risco", (
    "`urlopen` sem `timeout` usa o padrão global (nenhum): uma conexão travada prende a função para sempre. Passe `timeout=10`.",
), KW_TIMEOUT)}, imports=("import json", "import urllib.request"))
tpl("rede", "paginado", "busca todas as páginas de $col", """
def todos_$col(base_url):
    pagina, itens = 1, []
    while True:
@@+t!|        resp = requests.get(base_url, params={"page": pagina})
@@-t|        resp = requests.get(base_url, params={"page": pagina}, timeout=10)
@@-s|        resp.raise_for_status()
@@!s|        dados = resp.json()
        if not dados:
            return itens
        itens.extend(dados)
        pagina += 1
""", {"t": _RT, "s": _RS}, imports=("import requests",), teste=True)

# 6. off-by-one ------------------------------------------------------------------------------
tpl("off_by_one", "range_mais_um", "soma `$campo` de todos os $col", """
def soma_$campo($col):
    total = 0
@@+a!|    for i in range(len($col) + 1):
@@-a|    for i in range(len($col)):
        total += $col[i]["$campo"]
    return total
""", {"a": Bug("bug", (
    "`range(len($col) + 1)` vai um índice além do fim: a última volta acessa `$col[len($col)]` e levanta `IndexError`. Use `range(len($col))` (ou itere direto: `for x in $col`).",
), KW_OFF + ("range",))})
tpl("off_by_one", "ultimo", "devolve o último $item", """
def ultimo_$item($col):
@@+a!|    return $col[len($col)]
@@-a|    return $col[-1] if $col else None
""", {"a": Bug("bug", (
    "`$col[len($col)]` está sempre fora do intervalo (o último índice é `len($col) - 1`) e levanta `IndexError`. Use `$col[-1]`, tratando a lista vazia.",
), KW_OFF)})
tpl("off_by_one", "pares", "calcula a variação entre $col consecutivos", """
def variacoes($col):
    saida = []
@@+a!|    for i in range(len($col)):
@@-a|    for i in range(len($col) - 1):
        saida.append($col[i + 1]["$campo"] - $col[i]["$campo"])
    return saida
""", {"a": Bug("bug", (
    "Na última volta `i + 1` é `len($col)` e `$col[i + 1]` levanta `IndexError`. O laço deve ir até `len($col) - 1`.",
), KW_OFF + ("range",))})
tpl("off_by_one", "janela", "calcula médias móveis de `$campo`", """
def medias_moveis(valores, k):
    saida = []
@@+a!|    for i in range(len(valores) - k + 2):
@@-a|    for i in range(len(valores) - k + 1):
        janela = valores[i:i + k]
        saida.append(sum(janela) / k)
    return saida
""", {"a": Bug("bug", (
    "Off-by-one: a última janela começa em `len(valores) - k`, então o `range` deve ir até `len(valores) - k + 1`. Com `+ 2`, a última janela tem menos de `k` itens e a média sai errada.",
), KW_OFF + ("+ 1", "janela"))}, teste=True)

# 7. `is` com literal ------------------------------------------------------------------------
tpl("identidade", "lista", "conta $col", """
def contar_$col($col):
@@+a!|    if $col is []:
@@-a|    if not $col:
        return 0
    return len($col)
""", {"a": Bug("bug", (
    "`$col is []` compara identidade com uma lista nova e é sempre False. Para testar lista vazia use `if not $col:`.",
), KW_IS)})
tpl("identidade", "string", "valida o nome de um $item", """
def nome_valido(nome):
@@+a!|    if nome is "":
@@-a|    if not nome:
        return False
    return len(nome) <= 80
""", {"a": Bug("bug", (
    "`is \"\"` compara identidade, não valor: pode dar False para uma string vazia vinda de input (e o Python avisa com `SyntaxWarning`). Use `if not nome:` ou `nome == \"\"`.",
), KW_IS)})
tpl("identidade", "status", "filtra $col ativos", """
def ativos($col):
@@+a!|    return [x for x in $col if x["status"] is "ativo"]
@@-a|    return [x for x in $col if x["status"] == "ativo"]
""", {"a": Bug("bug", (
    "`is \"ativo\"` testa identidade: strings iguais vindas do banco ou de JSON não são o mesmo objeto, e o filtro devolve vazio. Use `== \"ativo\"`.",
), KW_IS)})
tpl("identidade", "zero", "verifica se há saldo", """
def sem_saldo(total):
@@+a!|    return total is 0
@@-a|    return total == 0
""", {"a": Bug("bug", (
    "`total is 0` compara identidade: falha para `0.0` ou para inteiros que não vêm do cache do CPython. Use `total == 0`.",
), KW_IS)}, teste=True)

# 8. modificar coleção enquanto itera -------------------------------------------------------
tpl("remove_iterando", "remove", "remove $col com `$campo` negativo", """
def limpar_$col($col):
    for x in $col:
@@+a!|        if x["$campo"] < 0:
@@+a|            $col.remove(x)
@@-a|        pass
@@-a|    $col[:] = [x for x in $col if x["$campo"] >= 0]
    return $col
""", {"a": Bug("bug", (
    "Remover itens de `$col` enquanto itera sobre ela pula o elemento seguinte: dois negativos seguidos e o segundo fica. Monte uma nova lista: `[x for x in $col if x[\"$campo\"] >= 0]`.",
), KW_ITER)})
tpl("remove_iterando", "dict", "remove chaves vazias", """
def sem_vazios(dados):
@@+a!|    for chave in dados:
@@+a|        if dados[chave] is None:
@@+a|            del dados[chave]
@@+a|    return dados
@@-a|    return {k: v for k, v in dados.items() if v is not None}
""", {"a": Bug("bug", (
    "Apagar chaves de `dados` durante o `for` levanta `RuntimeError: dictionary changed size during iteration`. Itere sobre `list(dados)` ou monte um novo dicionário.",
), KW_ITER)})
tpl("remove_iterando", "pop", "descarta $col cancelados", """
def descartar_cancelados($col):
@@+a!|    for i, x in enumerate($col):
@@+a|        if x["cancelado"]:
@@+a|            $col.pop(i)
@@+a|    return $col
@@-a|    return [x for x in $col if not x["cancelado"]]
""", {"a": Bug("bug", (
    "`pop(i)` dentro do `enumerate` desloca os índices: o item seguinte a um cancelado nunca é verificado. Filtre numa nova lista: `[x for x in $col if not x[\"cancelado\"]]`.",
), KW_ITER)})
tpl("remove_iterando", "set", "remove ids inválidos", """
def remover_invalidos(ids):
@@+a!|    for i in ids:
@@+a|        if i <= 0:
@@+a|            ids.discard(i)
@@+a|    return ids
@@-a|    return {i for i in ids if i > 0}
""", {"a": Bug("bug", (
    "Alterar o conjunto `ids` enquanto itera levanta `RuntimeError: Set changed size during iteration`. Crie um novo conjunto: `{i for i in ids if i > 0}`.",
), KW_ITER)}, teste=True)

# 9. exceção engolida ----------------------------------------------------------------------
tpl("excecao_engolida", "pass", "salva um $item no repositório", """
def salvar_$item(repo, $item):
    try:
        repo.salvar($item)
@@+a!|    except Exception:
@@+a|        pass
@@-a|    except ConnectionError:
@@-a|        logger.exception("falha ao salvar $item")
@@-a|        raise
""", {"a": Bug("risco", (
    "`except Exception: pass` esconde qualquer erro: se `salvar` falhar, ninguém fica sabendo e o $item some. Capture só o erro esperado, registre com `logger.exception` e propague.",
    "A exceção é engolida em silêncio. Pelo menos registre (`logger.exception`) e decida se deve propagar.",
), KW_EXC)}, imports_fix=("import logging", "logger = logging.getLogger(__name__)"))
tpl("excecao_engolida", "bare", "lê a configuração de um $item", """
def ler_config(caminho):
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
@@+a!|    except:
@@-a|    except (OSError, json.JSONDecodeError):
        return None
""", {"a": Bug("risco", (
    "`except:` sem tipo captura tudo, inclusive `KeyboardInterrupt` e erros de programação, e devolve `None` como se o arquivo não existisse. Capture `(OSError, json.JSONDecodeError)`.",
), KW_EXC + ("keyboardinterrupt",))}, imports=("import json",))
tpl("excecao_engolida", "zero_padrao", "converte o `$campo` digitado", """
def ler_$campo(texto):
    try:
        return float(texto.replace(",", "."))
@@+a!|    except Exception:
@@+a|        return 0
@@-a|    except ValueError:
@@-a|        raise ValueError(f"$campo inválido: {texto!r}") from None
""", {"a": Bug("risco", (
    "Qualquer erro vira `0` em silêncio: um `$campo` digitado errado entra como zero e corrompe os dados. Capture só `ValueError` e devolva um erro claro.",
), KW_EXC + ("zero",))})
tpl("excecao_engolida", "print", "processa os pagamentos de $col", """
def processar_pagamentos($col, gateway):
    for x in $col:
        try:
            gateway.cobrar(x["id"], x["$campo"])
@@+a!|        except Exception as e:
@@+a|            print(e)
@@-a|        except gateway.Erro:
@@-a|            logger.exception("cobrança falhou")
@@-a|            raise
""", {"a": Bug("risco", (
    "O erro da cobrança é só impresso e o laço segue como se tivesse dado certo: o $item fica sem pagar e ninguém é avisado. Registre com `logger.exception` e propague (ou marque a falha).",
), KW_EXC + ("print",))}, imports_fix=("import logging", "logger = logging.getLogger(__name__)"), teste=True)

# 10. comando de shell montado com string ------------------------------------------------------
tpl("comando_injecao", "tar", "compacta uma pasta de $col", """
def compactar(pasta, destino):
@@+a!|    subprocess.run(f"tar -czf {destino} {pasta}", shell=True)
@@-a|    subprocess.run(["tar", "-czf", destino, pasta], check=True)
""", {"a": Bug("bug", (
    "`shell=True` com f-string: um nome de pasta como `x; rm -rf ~` executa comandos (injeção), e espaço no caminho quebra o comando. Passe uma lista de argumentos sem `shell=True`: `subprocess.run([\"tar\", \"-czf\", destino, pasta], check=True)`.",
), KW_CMD)}, imports=("import subprocess",))
tpl("comando_injecao", "system", "apaga a pasta temporária de um $item", """
def limpar_tmp(caminho):
@@+a!|    os.system("rm -rf " + caminho)
@@-a|    shutil.rmtree(caminho, ignore_errors=True)
""", {"a": Bug("bug", (
    "Concatenar `caminho` num comando de shell permite injeção (e um caminho com espaço apaga outra coisa). Use `shutil.rmtree(caminho)`, sem shell.",
), KW_CMD + ("rmtree",))}, imports=("import os",), imports_fix=("import shutil",))
tpl("comando_injecao", "git", "lista commits de um autor", """
def commits_de(autor):
@@+a!|    saida = subprocess.check_output(f"git log --author={autor} --oneline", shell=True, text=True)
@@-a|    saida = subprocess.check_output(["git", "log", f"--author={autor}", "--oneline"], text=True)
    return saida.splitlines()
""", {"a": Bug("bug", (
    "`autor` entra num comando com `shell=True`: injeção de comando. Passe uma lista de argumentos e tire o `shell=True`.",
), KW_CMD)}, imports=("import subprocess",))
tpl("comando_injecao", "popen", "testa se um host responde", """
def responde(host):
@@+a!|    saida = os.popen(f"ping -c 1 {host}").read()
@@+a|    return "1 received" in saida
@@-a|    r = subprocess.run(["ping", "-c", "1", host], capture_output=True)
@@-a|    return r.returncode == 0
""", {"a": Bug("bug", (
    "`os.popen` passa pelo shell: um `host` como `x; curl ... | sh` executa comandos. Use `subprocess.run([\"ping\", \"-c\", \"1\", host])` e o `returncode`.",
), KW_CMD + ("popen",))}, imports=("import os",), imports_fix=("import subprocess",), teste=True)

# 11. condição invertida ------------------------------------------------------------------
tpl("condicao_invertida", "token", "verifica se o token ainda vale", """
def token_valido(token, agora):
@@+a!|    return agora > token["expira_em"]
@@-a|    return agora < token["expira_em"]
""", {"a": Bug("bug", (
    "A condição está invertida: devolve True justamente para o token que já expirou. Deve ser `agora < token[\"expira_em\"]`.",
), KW_INV + ("expir",))})
tpl("condicao_invertida", "saque", "verifica se o saque é possível", """
def pode_sacar(saldo, valor):
@@+a!|    return saldo < valor
@@-a|    return saldo >= valor
""", {"a": Bug("bug", (
    "Condição invertida: permite o saque quando o saldo é menor que o valor e bloqueia quando há saldo. Deve ser `saldo >= valor`.",
), KW_INV)})
tpl("condicao_invertida", "filtro", "filtra $col acima do limite", """
def acima_do_limite($col, limite):
@@+a!|    return [x for x in $col if x["$campo"] < limite]
@@-a|    return [x for x in $col if x["$campo"] > limite]
""", {"a": Bug("bug", (
    "O nome diz \"acima do limite\", mas o filtro usa `<` e devolve os que estão abaixo. Troque para `x[\"$campo\"] > limite`.",
), KW_INV + ("abaixo",))})
tpl("condicao_invertida", "maioridade", "verifica se o $item é maior de idade", """
def maior_de_idade(idade):
@@+a!|    return idade < 18
@@-a|    return idade >= 18
""", {"a": Bug("bug", (
    "Condição invertida: devolve True para menores de 18 e False para adultos. Deve ser `idade >= 18`.",
), KW_INV)})
tpl("condicao_invertida", "estoque", "decide se é hora de reabastecer", """
def precisa_reabastecer(estoque, minimo):
@@+a!|    return estoque > minimo
@@-a|    return estoque < minimo
""", {"a": Bug("bug", (
    "Invertido: pede reabastecimento quando o estoque está acima do mínimo e não quando falta. Deve ser `estoque < minimo`.",
), KW_INV)}, teste=True)

# 12. divisão inteira que perde a sobra ------------------------------------------------------
tpl("paginacao", "paginas", "calcula o número de páginas de $col", """
def paginas_de_$col($col, por_pagina):
@@+a!|    return len($col) // por_pagina
@@-a|    return math.ceil(len($col) / por_pagina)
""", {"a": Bug("bug", (
    "`//` arredonda para baixo e perde a última página quando sobra resto: 25 itens com 10 por página dão 2, não 3. Use `math.ceil(len($col) / por_pagina)`.",
), KW_PAG)}, imports_fix=("import math",))
tpl("paginacao", "lotes", "divide $col em lotes", """
def numero_de_lotes(n, tamanho):
@@+a!|    return n // tamanho
@@-a|    return (n + tamanho - 1) // tamanho
""", {"a": Bug("bug", (
    "A divisão inteira descarta o lote final incompleto: 101 itens em lotes de 50 dão 2, e um item fica de fora. Use `(n + tamanho - 1) // tamanho`.",
), KW_PAG + ("lote",))})
tpl("paginacao", "int", "calcula quantas páginas cabem", """
def paginas_necessarias(itens, por_pagina):
@@+a!|    return int(len(itens) / por_pagina)
@@-a|    return -(-len(itens) // por_pagina)
""", {"a": Bug("bug", (
    "`int(...)` trunca: com resto, a última página parcial some. Arredonde para cima (`math.ceil` ou `-(-a // b)`).",
), KW_PAG)})
tpl("paginacao", "header", "monta o cabeçalho de paginação", """
def cabecalho_paginacao(total_registros, limite):
    return {
        "X-Total": total_registros,
@@+a!|        "X-Paginas": total_registros // limite,
@@-a|        "X-Paginas": math.ceil(total_registros / limite),
    }
""", {"a": Bug("bug", (
    "`total_registros // limite` arredonda para baixo: com resto, o cliente nunca pede a última página. Use `math.ceil(total_registros / limite)`.",
), KW_PAG)}, imports_fix=("import math",), teste=True)

# --------------------------------------------------------------------------- código limpo (sem bug)

LIMPOS: list[tuple[str, str, tuple[str, ...], bool]] = [
    ("formata um valor em reais", """
def formatar_reais(valor):
    inteiro, centavos = f"{valor:,.2f}".split(".")
    return f"R$$ {inteiro.replace(',', '.')},{centavos}"
""", (), False),
    ("normaliza um texto para busca", """
def normalizar(texto):
    sem_acento = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).casefold().strip()
""", ("import unicodedata",), False),
    ("agrupa $col por `$chave`", """
def agrupar_por_$chave($col):
    grupos = defaultdict(list)
    for x in $col:
        grupos[x["$chave"]].append(x)
    return dict(grupos)
""", ("from collections import defaultdict",), False),
    ("lê um arquivo JSON", """
def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)
""", ("import json",), False),
    ("ordena $col por `$campo`", """
def ordenar_por_$campo($col, decrescente=False):
    return sorted($col, key=lambda x: x["$campo"], reverse=decrescente)
""", (), False),
    ("remove duplicatas mantendo a ordem", """
def sem_duplicatas(itens):
    vistos = set()
    saida = []
    for x in itens:
        if x not in vistos:
            vistos.add(x)
            saida.append(x)
    return saida
""", (), False),
    ("limita um número a um intervalo", """
def limitar(valor, minimo, maximo):
    if minimo > maximo:
        raise ValueError("minimo maior que maximo")
    return max(minimo, min(valor, maximo))
""", (), False),
    ("divide uma lista em pedaços", """
def em_pedacos(itens, tamanho):
    if tamanho <= 0:
        raise ValueError("tamanho deve ser positivo")
    return [itens[i:i + tamanho] for i in range(0, len(itens), tamanho)]
""", (), False),
    ("conta as palavras de um texto", """
def palavras_mais_comuns(texto, n=10):
    return Counter(texto.lower().split()).most_common(n)
""", ("from collections import Counter",), False),
    ("converte uma data em texto", """
def ler_data(texto):
    return datetime.strptime(texto, "%d/%m/%Y").date()
""", ("from datetime import datetime",), False),
    ("soma `$campo` dos $col", """
def total_$campo($col):
    return sum(x["$campo"] for x in $col)
""", (), False),
    ("valida um e-mail", """
EMAIL = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[a-z]{2,}$$", re.IGNORECASE)


def email_valido(email):
    return bool(EMAIL.match(email.strip()))
""", ("import re",), False),
    ("mescla configurações", """
def mesclar(padrao, extra):
    saida = dict(padrao)
    saida.update({k: v for k, v in extra.items() if v is not None})
    return saida
""", (), True),
    ("busca o primeiro $item de um `$chave`", """
def primeiro_de($col, valor):
    return next((x for x in $col if x["$chave"] == valor), None)
""", (), True),
    ("conta $col por status", """
def contar_status($col):
    return Counter(x["status"] for x in $col)
""", ("from collections import Counter",), True),
    ("arredonda um valor monetário", """
def arredondar(valor):
    return Decimal(str(valor)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
""", ("from decimal import ROUND_HALF_UP, Decimal",), True),
]

# --------------------------------------------------------------------------- render


@dataclass
class Trecho:
    linhas: list[str]
    desc: str
    imports: list[str]
    anotacoes: dict[int, tuple[str, Bug]]  # índice da linha no trecho -> (classe, bug)
    classe: str | None = None
    template: str | None = None


_TAG = re.compile(r"^@@([+\-!])(\w)(!?)\|(.*)$")


def render_tpl(t: Tpl, dom: dict[str, str], ativos: set[str]) -> Trecho:
    linhas: list[str] = []
    anot: dict[int, tuple[str, Bug]] = {}
    for raw in Template(t.codigo).substitute(dom).split("\n"):
        m = _TAG.match(raw)
        if not m:
            linhas.append(raw)
            continue
        op, k, anchor, code = m.groups()
        if op == "+":
            if k not in ativos:
                continue
            if anchor:
                anot[len(linhas)] = (t.classe, t.bugs[k])
        elif op == "-":
            if k in ativos:
                continue
        elif op == "!" and k in ativos:
            anot[len(linhas)] = (t.classe, t.bugs[k])
        linhas.append(code)
    # "pass" que sobra num for sem corpo na versão corrigida (remove_iterando/remove)
    linhas = _limpa_for_vazio(linhas, anot)
    imports = list(t.imports) + ([] if ativos else list(t.imports_fix))
    return Trecho(linhas, Template(t.desc).substitute(dom), imports, anot, t.classe, f"{t.classe}/{t.nome}")


def _limpa_for_vazio(linhas: list[str], anot: dict[int, tuple[str, Bug]]) -> list[str]:
    """Remove `for x in y:` seguido só de `pass` (artefato da versão corrigida de um template)."""
    out, i, mapa = [], 0, {}
    while i < len(linhas):
        if (
            i + 1 < len(linhas)
            and linhas[i].lstrip().startswith("for ")
            and linhas[i + 1].strip() == "pass"
            and i not in anot
        ):
            i += 2
            continue
        mapa[i] = len(out)
        out.append(linhas[i])
        i += 1
    if anot:
        novo = {mapa[k]: v for k, v in anot.items()}
        anot.clear()
        anot.update(novo)
    return out


def render_limpo(desc: str, codigo: str, imports: tuple[str, ...], dom: dict[str, str]) -> Trecho:
    return Trecho(Template(codigo.strip("\n")).substitute(dom).split("\n"), Template(desc).substitute(dom), list(imports), {})


def _ordena_imports(imports: Iterable[str]) -> list[str]:
    imps = sorted({i for i in imports if i.startswith(("import ", "from "))}, key=lambda s: (s.startswith("from"), s))
    extra = sorted({i for i in imports if not i.startswith(("import ", "from "))})
    out = imps[:]
    if extra:
        out += ["", *extra]
    return out


@dataclass
class Arquivo:
    path: str
    antigo: list[str] | None  # None = arquivo novo
    novo: list[str]
    bugs: list[dict[str, Any]]  # {"linha", "classe", "severidade", "comentario", "palavras_chave", "template"}
    resumo: str
    templates: list[str] = field(default_factory=list)

    def diff(self) -> str:
        head = [f"diff --git a/{self.path} b/{self.path}"]
        if self.antigo is None:
            head += ["new file mode 100644", "--- /dev/null", f"+++ b/{self.path}"]
            corpo = [f"@@ -0,0 +1,{len(self.novo)} @@"] + [f"+{x}" for x in self.novo]
        else:
            head += [f"--- a/{self.path}", f"+++ b/{self.path}"]
            corpo = list(difflib.unified_diff(self.antigo, self.novo, lineterm="", n=3))[2:]
        return "\n".join(head + corpo) + "\n"


def _junta(trechos: list[Trecho], prefixo: list[str] | None = None) -> tuple[list[str], list[tuple[int, str, Bug, str]]]:
    imports = [i for t in trechos for i in t.imports]
    linhas = _ordena_imports(imports)
    if linhas:
        linhas += ["", ""]
    if prefixo:
        linhas += prefixo + ["", ""]
    bugs = []
    for j, t in enumerate(trechos):
        base = len(linhas)
        for idx, (classe, bug) in t.anotacoes.items():
            bugs.append((base + idx + 1, classe, bug, t.template or ""))
        linhas += t.linhas
        if j < len(trechos) - 1:
            linhas += ["", ""]
    return linhas, bugs


def _frase(bug: Bug, dom: dict[str, str], rng: random.Random) -> str:
    return Template(rng.choice(bug.frases)).safe_substitute(dom)


def _nome_arquivo(rng: random.Random, dom: dict[str, str]) -> str:
    return f"{rng.choice(PASTAS)}/{dom['col']}{rng.choice(['', '_service', '_utils', '_repo', '_jobs'])}.py"


def _resumo(trechos: list[Trecho], modificado: bool) -> str:
    partes = [t.desc for t in trechos]
    onde = " ao arquivo" if modificado else ""
    if len(partes) == 1:
        return f"Adiciona{onde} uma função que {partes[0]}."
    return f"Adiciona{onde} {len(partes)} funções: " + "; ".join(f"uma que {x}" for x in partes) + "."


def gera_arquivo(
    rng: random.Random, *, teste: bool, p_bug: float = 0.65, max_bugs: int = 3
) -> Arquivo:
    dom = rng.choice(DOMINIOS)
    tpls = [t for t in T if t.teste == teste]
    limpos = [c for c in LIMPOS if c[3] == teste]
    n = rng.choices([1, 2, 3], weights=[4, 4, 2])[0]
    trechos: list[Trecho] = []
    usados: set[str] = set()
    n_bugs = 0
    for _ in range(n):
        if rng.random() < 0.7:
            t = rng.choice([x for x in tpls if x.classe + x.nome not in usados] or tpls)
            usados.add(t.classe + t.nome)
            ativos: set[str] = set()
            if n_bugs < max_bugs and rng.random() < p_bug:
                ks = list(t.bugs)
                ativos = set(rng.sample(ks, rng.randint(1, len(ks))))
                if n_bugs + len(ativos) > max_bugs:
                    ativos = set(list(ativos)[: max_bugs - n_bugs])
                n_bugs += len(ativos)
            trechos.append(render_tpl(t, dom, ativos))
        else:
            desc, cod, imps, _ = rng.choice([c for c in limpos if c[0] not in usados] or limpos)
            usados.add(desc)
            trechos.append(render_limpo(desc, cod, imps, dom))
    # nomes de função repetidos no mesmo arquivo (dois templates do mesmo tipo) não são problema de sintaxe
    path = _nome_arquivo(rng, dom)
    modo = rng.random()
    if modo < 0.6 or len(trechos) == 1 and modo < 0.8:
        novo, bugs = _junta(trechos)
        arq = Arquivo(path, None, novo, [], _resumo(trechos, False))
    else:
        # arquivo existente: um trecho limpo já estava lá; os outros entram no PR
        desc, cod, imps, _ = rng.choice([c for c in limpos if c[0] not in usados] or limpos)
        velho = render_limpo(desc, cod, imps, dom)
        antigo, _ = _junta([velho])
        novo, bugs = _junta([velho, *trechos])
        arq = Arquivo(path, antigo, novo, [], _resumo(trechos, True))
    for linha, classe, bug, template in sorted(bugs, key=lambda b: b[0]):
        arq.bugs.append(
            {
                "linha": linha,
                "classe": classe,
                "severidade": bug.severidade,
                "comentario": _frase(bug, dom, rng),
                "palavras_chave": list(bug.palavras_chave),
                "template": template,
            }
        )
    arq.templates = [t.template for t in trechos if t.template]
    return arq


def resposta(arq: Arquivo) -> str:
    return json.dumps(
        {
            "resumo": arq.resumo,
            "comentarios": [
                {"linha": b["linha"], "severidade": b["severidade"], "comentario": b["comentario"]} for b in arq.bugs
            ],
        },
        ensure_ascii=False,
    )


def valida(arq: Arquivo) -> None:
    """O código gerado compila e cada bug cai numa linha adicionada do diff."""
    ast.parse("\n".join(arq.novo))
    if arq.antigo is not None:
        ast.parse("\n".join(arq.antigo))
    from forja_review.diff import parse_unified

    (f,) = parse_unified(arq.diff())
    for b in arq.bugs:
        if b["linha"] not in f.commentable() or b["linha"] not in f.added_lines():
            raise AssertionError(f"{arq.path}: linha {b['linha']} do bug não é uma linha adicionada")


def _sem_acento(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.casefold())
    return "".join(c for c in t if not unicodedata.combining(c))


@dataclass
class Nota:
    """Uma resposta do modelo contra o gabarito, com a mesma régua do `examples/recall.py` do forja-review."""

    achados: int
    bugs: int
    extras: int
    problemas: list[str]

    @property
    def perfeita(self) -> bool:
        return self.achados == self.bugs and self.extras == 0 and not self.problemas

    @property
    def valor(self) -> float:
        """Maior é melhor: cada bug achado vale 1, cada extra e cada problema de formato custa 1."""
        return self.achados - self.extras - len(self.problemas)


def pontua(arq: Arquivo, comentarios: Iterable[dict[str, Any]], problemas: Iterable[str] = (), tol: int = 2) -> Nota:
    """Bug achado = comentário a até `tol` linhas que cita uma palavra-chave da classe; o resto é extra."""
    cs = list(comentarios)

    def bate(b: dict[str, Any], c: dict[str, Any]) -> bool:
        return abs(int(c["linha"]) - b["linha"]) <= tol and any(
            _sem_acento(k) in _sem_acento(str(c.get("comentario", ""))) for k in b["palavras_chave"]
        )

    achados = sum(1 for b in arq.bugs if any(bate(b, c) for c in cs))
    extras = sum(1 for c in cs if not any(bate(b, c) for b in arq.bugs))
    return Nota(achados=achados, bugs=len(arq.bugs), extras=extras, problemas=list(problemas))
