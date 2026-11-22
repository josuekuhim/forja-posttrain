"""Pontua os revisores da rodada T4 (D1 sint, D2 real, D3 misto) e as linhas de base nos quatro PRs.

Mesma régua do `scripts/graficos.py` e do `forja-review/examples/recall.py` (linha ±tolerância e palavra-chave),
com uma diferença necessária no PR real (`review-eval-v0`): os bugs reais são da classe `outro`, sem
palavra-chave, então contam só pela linha (como pede o `dataset.yaml`). Também mede o que importa no dado
real: quantos dos 199 arquivos **sem bug** receberam comentário (alarme falso por arquivo).

    forja-review/.venv312/Scripts/python.exe forja-posttrain/scripts/pontua_t4.py   # escreve results/review-t4-v0.json
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REVIEW = RAIZ.parent / "forja-review"
GAB = {
    "teste": RAIZ / "data/review-sintetico-v0/teste/gabarito-teste.json",
    "holdout": REVIEW / "examples/gabarito-holdout.json",
    "bugs": REVIEW / "examples/gabarito-bugs-plantados.json",
    "real": RAIZ.parent / "forja-evals/datasets/review-eval-v0/gabarito-eval.json",
}
REVISORES = [
    ("Qwen 0.5B base · prompt curto", "qwen0.5b-base-curto"),
    ("SFT CPU (sintético, R1)", "qwen0.5b-sft-curto"),
    ("D1 · T4 sintético", "qwen0.5b-t4-sint-curto"),
    ("D2 · T4 real", "qwen0.5b-t4-real-curto"),
    ("D3 · T4 misto", "qwen0.5b-t4-misto-curto"),
    ("ruff", "ruff"),
]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.casefold())
    return "".join(c for c in t if not unicodedata.combining(c))


def pontua(gab: dict, res: dict) -> dict:
    tol, bugs, cs = gab["tolerancia_linhas"], gab["bugs"], res["comentarios"]

    def bate(b, c):
        if c["path"] != b["arquivo"] or abs(c["linha"] - b["linha"]) > tol:
            return False
        kws = b.get("palavras_chave") or []
        return not kws or any(_norm(k) in _norm(c["comentario"]) for k in kws)

    achados = sum(any(bate(b, c) for c in cs) for b in bugs)
    extras = sum(1 for c in cs if not any(bate(b, c) for b in bugs))
    com_bug = {b["arquivo"] for b in bugs}
    comentados = {c["path"] for c in cs}
    n_arq = gab.get("arquivos")
    limpos = (n_arq - len(com_bug)) if n_arq else None
    out = {
        "achados": achados, "bugs": len(bugs), "comentarios": len(cs), "extras": extras,
        "problemas_de_formato": sum(len(a.get("problemas", [])) for a in res.get("arquivos", [])),
        "segundos_no_modelo": res.get("segundos_no_modelo"),
    }
    if limpos:
        out["arquivos_limpos"] = limpos
        out["limpos_com_comentario"] = len(comentados - com_bug)
    return out


def main() -> int:
    gabs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in GAB.items()}
    out: dict = {"revisores": {}}
    for rotulo, suf in REVISORES:
        linha = {}
        for pr, gab in gabs.items():
            f = REVIEW / "results" / f"{pr}-{suf}.json"
            if f.exists():
                linha[pr] = pontua(gab, json.loads(f.read_text(encoding="utf-8")))
        if linha:
            out["revisores"][rotulo] = linha
    destino = RAIZ / "results/review-t4-v0.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for rotulo, linha in out["revisores"].items():
        partes = []
        for pr, d in linha.items():
            s = f"{pr} {d['achados']}/{d['bugs']} (extras {d['extras']}, formato {d['problemas_de_formato']}"
            if "limpos_com_comentario" in d:
                s += f", limpos comentados {d['limpos_com_comentario']}/{d['arquivos_limpos']}"
            partes.append(s + ")")
        print(f"{rotulo}: " + " | ".join(partes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
