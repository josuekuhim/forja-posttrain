"""Gera o dataset sintético de review (treino) e o PR de teste com gabarito (templates nunca vistos).

    uv run --group dados python scripts/gerar_review.py --n-treino 400 --n-teste 60

O prompt é montado pelo próprio forja-review (`prompt.messages`, estilo "curto"): treino e inferência
usam exatamente o mesmo texto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml
from forja_review.diff import parse_unified
from forja_review.prompt import chunks, messages

from forja_posttrain.data import contencao, ngramas
from forja_posttrain.review_sintetico import T, gera_arquivo, resposta, valida

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--n-treino", type=int, default=400)
ap.add_argument("--n-teste", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/review-sintetico-v0")
ap.add_argument("--max-comentarios", type=int, default=3)
ap.add_argument("--planted", nargs="*", default=["../forja-review/examples/pr-bugs-plantados.diff", "../forja-review/examples/pr-holdout.diff"])
a = ap.parse_args()
out = Path(a.out)
(out / "teste").mkdir(parents=True, exist_ok=True)


def user_msgs(arq):
    (f,) = parse_unified(arq.diff())
    (hunks,) = chunks(f, 8000)
    return messages(f, hunks, max_comentarios=a.max_comentarios, estilo="curto")


rng = random.Random(a.seed)
vistos: set[str] = set()
linhas, classes, tpls = [], Counter(), Counter()
while len(linhas) < a.n_treino:
    arq = gera_arquivo(rng, teste=False, max_bugs=a.max_comentarios)
    valida(arq)
    h = hashlib.sha256(arq.diff().encode()).hexdigest()
    if h in vistos:
        continue
    vistos.add(h)
    msgs = user_msgs(arq) + [{"role": "assistant", "content": resposta(arq)}]
    classes.update(b["classe"] for b in arq.bugs)
    tpls.update(arq.templates)
    linhas.append({
        "id": f"rsv0-{len(linhas):04d}",
        "categoria": "deteccao_bug",
        "messages": msgs,
        "origem": "sintetico:forja-posttrain/review_sintetico@v0",
        "licenca": "MIT",
    })
with (out / "sft.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for r in linhas:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# PR de teste: só templates de teste, caminhos únicos
diffs, bugs, limpos, paths = [], [], 0, set()
classes_teste = Counter()
while len(diffs) < a.n_teste:
    arq = gera_arquivo(rng, teste=True, max_bugs=a.max_comentarios)
    if arq.path in paths:
        stem = arq.path[:-3]
        arq.path = next(f"{stem}_{i}.py" for i in range(2, 999) if f"{stem}_{i}.py" not in paths)
    valida(arq)
    paths.add(arq.path)
    diffs.append(arq.diff())
    limpos += not arq.bugs
    for b in arq.bugs:
        classes_teste[b["classe"]] += 1
        bugs.append({"arquivo": arq.path, "linha": b["linha"], "descricao": f"{b['classe']} ({b['template']})",
                     "palavras_chave": b["palavras_chave"]})
(out / "teste" / "pr-teste.diff").write_text("".join(diffs), encoding="utf-8", newline="\n")
gab = {"diff": "pr-teste.diff", "tolerancia_linhas": 2, "arquivos": len(diffs), "arquivos_sem_bug": limpos,
       "nota": "Templates de teste: nunca usados no treino (split por template).", "bugs": bugs}
(out / "teste" / "gabarito-teste.json").write_text(json.dumps(gab, ensure_ascii=False, indent=1), encoding="utf-8")

# Contaminação contra os PRs de bugs plantados do forja-review (código nunca visto no treino)
corpus = set()
for r in linhas:
    corpus |= ngramas(r["messages"][1]["content"], 5)
contam = {}
for p in a.planted:
    add = "\n".join(l[1:] for l in Path(p).read_text(encoding="utf-8").splitlines() if l.startswith("+") and not l.startswith("+++"))
    contam[Path(p).name] = round(contencao(corpus, ngramas(add, 5)), 3)

meta = {
    "nome": "review-sintetico-v0",
    "uso": "treino",
    "versao": 0,
    "descricao": "Diffs Python com bugs injetados por template (e as versões corrigidas como negativos), no formato do forja-review.",
    "origem": "sintetico: forja_posttrain/review_sintetico.py",
    "licenca": "MIT",
    "seed": a.seed,
    "exemplos": len(linhas),
    "sem_bug": sum(1 for r in linhas if '"comentarios": []' in r["messages"][-1]["content"]),
    "bugs_por_classe": dict(sorted(classes.items())),
    "templates_treino": sum(1 for t in T if not t.teste),
    "templates_teste": sum(1 for t in T if t.teste),
    "teste": {"arquivos": len(diffs), "sem_bug": limpos, "bugs": len(bugs), "por_classe": dict(sorted(classes_teste.items()))},
    "contencao_5gramas_vs_prs_plantados": contam,
}
(out / "dataset.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
