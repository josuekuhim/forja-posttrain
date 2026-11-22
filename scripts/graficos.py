"""Gera a tabela e os gráficos (SVG, sem dependências) do experimento de review em CPU.

Lê os `--json-out` do forja-review e os gabaritos, pontua com a mesma régua do `examples/recall.py`
(linha ±2 e palavra-chave) e escreve:

    results/review-cpu-v0.json          números de cada revisor, por PR e por classe
    docs/img/review-teste.svg           bugs achados × comentários errados no PR de teste
    docs/img/review-classes.svg         por classe de bug: ruff × SFT (onde cada um enxerga)
    docs/img/review-loss.svg            perda de treino e de validação do SFT

    uv run python scripts/graficos.py
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from html import escape
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REVIEW = RAIZ.parent / "forja-review"
TESTE = RAIZ / "data/review-sintetico-v0/teste/gabarito-teste.json"
PRS = {
    "teste": TESTE,
    "holdout": REVIEW / "examples/gabarito-holdout.json",
    "plantados": REVIEW / "examples/gabarito-bugs-plantados.json",
}
ARQ = {"teste": "teste", "holdout": "holdout", "plantados": "bugs"}  # prefixo em forja-review/results
REVISORES = [  # (rótulo, sufixo do arquivo em forja-review/results)
    ("Qwen 0.5B base · full prompt", "qwen0.5b-base-completo"),
    ("Qwen 0.5B base · short prompt", "qwen0.5b-base-curto"),
    ("Qwen 0.5B + SFT", "qwen0.5b-sft-curto"),
    ("Qwen 0.5B + SFT + DPO", "qwen0.5b-dpo-curto"),
    ("bandit", "bandit"),
    ("ruff", "ruff"),
    ("ruff + SFT", "ruff-sft"),
    ("ruff + DPO", "ruff-dpo"),
]

# Paleta de referência (validada para daltonismo nos três primeiros slots), superfície clara.
SUP, TXT, TXT2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
CLASSES = {
    "rede": "network: timeout/status", "sql_injecao": "SQL injection", "condicao_invertida": "inverted condition",
    "off_by_one": "off-by-one", "comando_injecao": "command injection", "divisao_vazia": "division by empty",
    "paginacao": "pagination", "padrao_mutavel": "mutable default", "remove_iterando": "remove while iterating",
    "excecao_engolida": "swallowed exception", "recurso_aberto": "unclosed resource", "identidade": "is × ==",
}
FONTE = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.casefold())
    return "".join(c for c in t if not unicodedata.combining(c))


def pontua(gab: dict, res: dict) -> dict:
    tol, bugs, cs = gab["tolerancia_linhas"], gab["bugs"], res["comentarios"]

    def bate(b, c):
        return (c["path"] == b["arquivo"] and abs(c["linha"] - b["linha"]) <= tol
                and any(_norm(k) in _norm(c["comentario"]) for k in b["palavras_chave"]))

    achados = [i for i, b in enumerate(bugs) if any(bate(b, c) for c in cs)]
    extras = sum(1 for c in cs if not any(bate(b, c) for b in bugs))
    probs = sum(len(a["problemas"]) for a in res.get("arquivos", []))
    return {"achados": len(achados), "bugs": len(bugs), "extras": extras, "comentarios": len(cs),
            "problemas_de_formato": probs, "segundos_no_modelo": res.get("segundos_no_modelo"),
            "_idx": achados}


def coleta() -> dict:
    out: dict = {"revisores": {}, "por_classe": {}}
    gabs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in PRS.items()}
    idx = {}
    for rotulo, suf in REVISORES:
        linha = {}
        for pr, gab in gabs.items():
            f = REVIEW / "results" / f"{ARQ[pr]}-{suf}.json"
            if f.exists():
                linha[pr] = pontua(gab, json.loads(f.read_text(encoding="utf-8")))
        if linha:
            idx[rotulo] = set(linha.get("teste", {}).get("_idx", []))
            out["revisores"][rotulo] = {pr: {k: v for k, v in d.items() if k != "_idx"} for pr, d in linha.items()}
    classes = [b["descricao"].split(" ")[0] for b in gabs["teste"]["bugs"]]
    for c, n in Counter(classes).most_common():
        ids = {i for i, x in enumerate(classes) if x == c}
        out["por_classe"][c] = {"bugs": n, **{r: len(ids & s) for r, s in idx.items()}}
    return out


# ---------- SVG ----------

def _txt(x, y, s, *, cor=TXT, tam=13, ancora="start", peso="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' fill='{cor}' font-size='{tam}' text-anchor='{ancora}' "
            f"font-weight='{peso}' {FONTE}>{escape(str(s))}</text>")


def _barra(x, y, w, h, cor):
    """Barra horizontal com a ponta de dados arredondada (4px) e a base reta no eixo."""
    if w <= 0:
        return ""
    r = min(4, w, h / 2)
    return (f"<path d='M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} "
            f"a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z' fill='{cor}'/>")


def _legenda(x, y, itens):
    s, dx = [], 0
    for rot, cor in itens:
        s.append(f"<rect x='{x + dx}' y='{y - 10}' width='12' height='12' rx='3' fill='{cor}'/>")
        s.append(_txt(x + dx + 18, y, rot, cor=TXT2, tam=12))
        dx += 24 + 7 * len(rot)
    return "".join(s)


def _svg(w, h, corpo, titulo):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' "
            f"aria-label='{escape(titulo)}'><rect width='{w}' height='{h}' fill='{SUP}'/>{corpo}</svg>\n")


def barras_agrupadas(titulo, sub, linhas, series, maximo, passo, rotulo_eixo, w=760):
    """linhas: [(rótulo, [valor por série])]; series: [(nome, cor)]."""
    esq, dir_, topo, bh, gap, grupo_gap = 230, 40, 78, 12, 2, 14
    gh = len(series) * bh + (len(series) - 1) * gap
    h = topo + len(linhas) * (gh + grupo_gap) + 46
    pw = w - esq - dir_
    esc = pw / maximo
    s = [_txt(16, 26, titulo, tam=15, peso="600"), _txt(16, 46, sub, cor=TXT2, tam=12),
         _legenda(esq, 66, series)]
    base_y = topo + len(linhas) * (gh + grupo_gap) - grupo_gap
    for v in range(0, maximo + 1, passo):
        x = esq + v * esc
        s.append(f"<line x1='{x:.1f}' y1='{topo - 4}' x2='{x:.1f}' y2='{base_y + 4}' stroke='{GRADE}' stroke-width='1'/>")
        s.append(_txt(x, base_y + 20, v, cor=TXT2, tam=11, ancora="middle"))
    s.append(_txt(esq + pw / 2, base_y + 38, rotulo_eixo, cor=TXT2, tam=11, ancora="middle"))
    for i, (rot, vals) in enumerate(linhas):
        y0 = topo + i * (gh + grupo_gap)
        s.append(_txt(esq - 10, y0 + gh / 2 + 4, rot, tam=12, ancora="end"))
        for j, ((_, cor), v) in enumerate(zip(series, vals, strict=True)):
            if v is None:
                continue
            y = y0 + j * (bh + gap)
            s.append(_barra(esq, y, v * esc, bh, cor))
            s.append(_txt(esq + v * esc + 5, y + bh - 2, v, cor=TXT2, tam=11))
    s.append(f"<line x1='{esq}' y1='{topo - 4}' x2='{esq}' y2='{base_y + 4}' stroke='{TXT2}' stroke-width='1'/>")
    return _svg(w, h, "".join(s), titulo)


def linhas_svg(titulo, sub, series, xmax, ymax, w=760, h=330):
    """series: [(nome, cor, [(x, y)], marcadores)]."""
    esq, dir_, topo, base = 56, 24, 78, 48
    pw, ph = w - esq - dir_, h - topo - base
    X = lambda x: esq + x / xmax * pw  # noqa: E731
    Y = lambda y: topo + ph - y / ymax * ph  # noqa: E731
    s = [_txt(16, 26, titulo, tam=15, peso="600"), _txt(16, 46, sub, cor=TXT2, tam=12),
         _legenda(esq, 66, [(n, c) for n, c, _, _ in series])]
    for i in range(5):
        v = ymax * i / 4
        s.append(f"<line x1='{esq}' y1='{Y(v):.1f}' x2='{w - dir_}' y2='{Y(v):.1f}' stroke='{GRADE}'/>")
        s.append(_txt(esq - 8, Y(v) + 4, f"{v:.1f}".replace(".", ","), cor=TXT2, tam=11, ancora="end"))
    for v in range(0, xmax + 1, 16):
        s.append(_txt(X(v), topo + ph + 18, v, cor=TXT2, tam=11, ancora="middle"))
    s.append(_txt(esq + pw / 2, topo + ph + 36, "optimization step", cor=TXT2, tam=11, ancora="middle"))
    for _, cor, pts, marc in series:
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (x, y) in enumerate(pts))
        s.append(f"<path d='{d}' fill='none' stroke='{cor}' stroke-width='2' stroke-linejoin='round'/>")
        if marc:
            for x, y in pts:
                s.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='4.5' fill='{cor}' stroke='{SUP}' stroke-width='2'/>")
                fim = x > xmax * 0.9  # perto da borda: rótulo à esquerda do ponto
                s.append(_txt(X(x) + (-8 if fim else 8), Y(y) - 10, f"{y:.3f}".replace(".", ","), cor=TXT2, tam=11,
                              ancora="end" if fim else "start"))
    return _svg(w, h, "".join(s), titulo)


def main() -> None:
    dados = coleta()
    (RAIZ / "results").mkdir(exist_ok=True)
    (RAIZ / "results/review-cpu-v0.json").write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    img = RAIZ / "docs/img"
    img.mkdir(parents=True, exist_ok=True)

    rev = dados["revisores"]
    linhas = [(r, [d["teste"]["achados"], d["teste"]["extras"]]) for r, d in rev.items() if "teste" in d]
    n = next(iter(rev.values()))["teste"]["bugs"]
    (img / "review-teste.svg").write_text(barras_agrupadas(
        f"Test PR: {n} bugs in 60 files, from templates never seen in training",
        "Found = a comment within 2 lines of the bug that names the problem. Extra = a comment that matches no bug.",
        linhas, [("bugs found", AZUL), ("extra comments (noise)", LARANJA)], 50, 10, "count"),
        encoding="utf-8")

    pc = dados["por_classe"]
    linhas = [(f"{CLASSES.get(c, c)} ({v['bugs']})", [v.get("ruff"), v.get("Qwen 0.5B + SFT")]) for c, v in pc.items()]
    (img / "review-classes.svg").write_text(barras_agrupadas(
        "By bug class: the linter catches patterns, the model catches semantics",
        "Bugs found in the test PR; in parentheses, how many bugs the class has.",
        linhas, [("ruff", AZUL), ("Qwen 0.5B + SFT", LARANJA)], 9, 1, "bugs found"), encoding="utf-8")

    estado = sorted((RAIZ / "runs/review-lora-cpu/checkpoints").glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if estado:
        h = json.loads(estado[-1].read_text(encoding="utf-8"))["log_history"]
        tr = [(x["step"], x["loss"]) for x in h if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
        (img / "review-loss.svg").write_text(linhas_svg(
            "SFT LoRA: training and validation loss (96 steps, 3h36 on CPU)",
            "Validation = 20 held-out examples from the same generator: it measures fit to the format, not generalization.",
            [("train", AZUL, tr, False), ("validation", LARANJA, ev, True)], 96, 2.0), encoding="utf-8")
    print(json.dumps(dados["revisores"], ensure_ascii=False, indent=1)[:1500])


if __name__ == "__main__":
    main()
