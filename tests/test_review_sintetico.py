import ast
import itertools
import json
import random
import unicodedata

import pytest

from forja_posttrain.review_sintetico import (
    DOMINIOS,
    LIMPOS,
    T,
    gera_arquivo,
    render_limpo,
    render_tpl,
    resposta,
)

forja_review = pytest.importorskip("forja_review", reason="gerar dados de review exige `uv sync --group dados`")


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.casefold())
    return "".join(c for c in t if not unicodedata.combining(c))


def _variantes(t):
    ks = list(t.bugs)
    for r in range(len(ks) + 1):
        yield from (set(c) for c in itertools.combinations(ks, r))


@pytest.mark.parametrize("t", T, ids=lambda t: f"{t.classe}/{t.nome}")
def test_template_compila_e_ancora_um_comentario_por_bug(t):
    for dom in DOMINIOS:
        for ativos in _variantes(t):
            tr = render_tpl(t, dom, ativos)
            ast.parse("\n".join(tr.linhas))
            assert len(tr.anotacoes) == len(ativos), (t.nome, ativos)
            assert all(tr.linhas[i].strip() for i in tr.anotacoes)  # a linha comentada tem código


@pytest.mark.parametrize("t", T, ids=lambda t: f"{t.classe}/{t.nome}")
def test_versao_corrigida_difere_da_com_bug(t):
    dom = DOMINIOS[0]
    assert render_tpl(t, dom, set()).linhas != render_tpl(t, dom, set(t.bugs)).linhas


def test_limpos_compilam():
    for desc, cod, imps, _ in LIMPOS:
        for dom in DOMINIOS:
            ast.parse("\n".join(render_limpo(desc, cod, imps, dom).linhas))


@pytest.mark.parametrize("t", T, ids=lambda t: f"{t.classe}/{t.nome}")
def test_gabarito_passa_no_proprio_checador(t):
    """Cada frase de comentário contém uma palavra-chave da classe: o checador de recall aceitaria o gabarito."""
    for b in t.bugs.values():
        for f in b.frases:
            assert any(_norm(k) in _norm(f) for k in b.palavras_chave), f


def test_cada_classe_tem_template_de_treino_e_de_teste():
    classes = {t.classe for t in T}
    for c in classes:
        assert {t.teste for t in T if t.classe == c} == {False, True}, c


@pytest.mark.parametrize("teste", [False, True])
def test_arquivo_gerado_e_valido_e_respeita_o_split(teste):
    from forja_posttrain.review_sintetico import valida

    rng = random.Random(123)
    nomes_teste = {f"{t.classe}/{t.nome}" for t in T if t.teste}
    for _ in range(150):
        arq = gera_arquivo(rng, teste=teste)
        valida(arq)  # compila; cada bug cai numa linha adicionada do diff
        assert len(arq.bugs) <= 3
        assert all((tp in nomes_teste) == teste for tp in arq.templates)
        r = json.loads(resposta(arq))
        assert [c["linha"] for c in r["comentarios"]] == sorted(b["linha"] for b in arq.bugs)


def test_linha_do_gabarito_e_a_linha_do_bug():
    from forja_review.diff import parse_unified

    rng = random.Random(7)
    vistos = 0
    while vistos < 30:
        arq = gera_arquivo(rng, teste=False)
        if not arq.bugs:
            continue
        (f,) = parse_unified(arq.diff())
        for b in arq.bugs:
            assert f.line_text(b["linha"]) == arq.novo[b["linha"] - 1]
        vistos += 1


def test_pontua_gabarito_e_perfeito_e_alucinacao_custa():
    from forja_posttrain.review_sintetico import pontua

    rng = random.Random(3)
    com_bug = next(a for a in iter(lambda: gera_arquivo(rng, teste=False), None) if a.bugs)
    limpo = next(a for a in iter(lambda: gera_arquivo(rng, teste=False), None) if not a.bugs)
    gab = json.loads(resposta(com_bug))["comentarios"]
    assert pontua(com_bug, gab).perfeita
    assert pontua(limpo, []).perfeita
    inventado = [{"linha": 1, "comentario": "Isto pode vazar memória."}]
    n = pontua(limpo, inventado)
    assert not n.perfeita and n.extras == 1 and n.valor == -1
    n = pontua(com_bug, [])
    assert n.achados == 0 and n.valor == 0 and not n.perfeita
    assert not pontua(com_bug, gab, ["json_invalido"]).perfeita
