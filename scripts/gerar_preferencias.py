"""Gera pares de preferência *on-policy* para o DPO de review.

O modelo SFT revisa arquivos novos (só templates de treino, outra seed). Cada resposta é pontuada
contra o gabarito com a mesma régua do `recall.py` do forja-review. Quando a pior das amostras erra
(inventa bug, perde bug, quebra o formato), vira par: escolhida = gabarito, rejeitada = o erro do próprio
modelo. Resposta perfeita não vira par: não há o que corrigir.

    # forja-serve com o SFT mesclado na porta 8000
    uv run --group dados python scripts/gerar_preferencias.py --n 240 --model review-sft
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import yaml
from forja_contracts import OrigemPar, PreferencePair, write_jsonl
from forja_review.diff import parse_unified
from forja_review.parse import parse
from forja_review.prompt import chunks, messages

from forja_posttrain.review_sintetico import gera_arquivo, pontua, resposta, valida

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=240, help="arquivos revisados pelo modelo")
ap.add_argument("--seed", type=int, default=1, help="diferente da seed do SFT: arquivos novos")
ap.add_argument("--amostras", type=int, default=2)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--max-comentarios", type=int, default=3)
ap.add_argument("--concorrencia", type=int, default=4)
ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--model", default="review-sft")
ap.add_argument("--sft", default="data/review-sintetico-v0/sft.jsonl", help="diffs já vistos no SFT ficam de fora")
ap.add_argument("--out", default="data/review-preferencias-v0")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

vistos = set()
with open(a.sft, encoding="utf-8") as fh:
    for line in fh:
        user = json.loads(line)["messages"][1]["content"]
        vistos.add(hashlib.sha256(user.encode()).hexdigest())

rng = random.Random(a.seed)
casos = []
while len(casos) < a.n:
    arq = gera_arquivo(rng, teste=False, max_bugs=a.max_comentarios)
    valida(arq)
    (f,) = parse_unified(arq.diff())
    (hunks,) = chunks(f, 8000)
    msgs = messages(f, hunks, max_comentarios=a.max_comentarios, estilo="curto")
    h = hashlib.sha256(msgs[1]["content"].encode()).hexdigest()
    if h in vistos:
        continue
    vistos.add(h)
    casos.append((arq, f, msgs))


async def amostra(cli: httpx.AsyncClient, sem: asyncio.Semaphore, msgs: list[dict], seed: int) -> str:
    async with sem:
        r = await cli.post(
            f"{a.base_url}/chat/completions",
            json={"model": a.model, "messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens,
                  "seed": seed},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def main():
    sem = asyncio.Semaphore(a.concorrencia)
    async with httpx.AsyncClient(timeout=900) as cli:
        # O forja-serve responde a qualquer nome de modelo: confere quem está servindo antes de gastar horas.
        servidos = [m["id"] for m in (await cli.get(f"{a.base_url}/models")).json()["data"]]
        if a.model not in servidos:
            sys.exit(f"o endpoint serve {servidos}, não {a.model!r}: pares on-policy exigem a política certa")
        tarefas = [amostra(cli, sem, msgs, s) for _, _, msgs in casos for s in range(a.amostras)]
        return await asyncio.gather(*tarefas)


t0 = time.time()
textos = asyncio.run(main())
seg = time.time() - t0

pares, tipos, perfeitas = [], Counter(), 0
total_amostras = 0
for i, (arq, f, msgs) in enumerate(casos):
    notas = []
    for txt in textos[i * a.amostras : (i + 1) * a.amostras]:
        p = parse(txt, f, max_comentarios=a.max_comentarios)
        cs = [{"linha": c.linha, "comentario": c.comentario} for c in p.comentarios]
        notas.append((pontua(arq, cs, p.problemas), txt))
    total_amostras += len(notas)
    perfeitas += sum(n.perfeita for n, _ in notas)
    pior, txt = min(notas, key=lambda x: x[0].valor)
    if pior.perfeita:
        continue
    if pior.problemas:
        tipos["formato"] += 1
    elif not arq.bugs:
        tipos["inventou_bug_em_arquivo_limpo"] += 1
    elif pior.extras:
        tipos["comentario_extra_em_arquivo_com_bug"] += 1
    else:
        tipos["perdeu_bug"] += 1
    pares.append(PreferencePair(prompt=msgs[1]["content"], escolhida=resposta(arq), rejeitada=txt.strip(),
                                origem=OrigemPar.JUIZ))

write_jsonl(out / "pairs.jsonl", pares)
meta = {
    "nome": out.name,
    "uso": "treino",
    "gerado_de": ["../review-sintetico-v0"],
    "descricao": "Pares on-policy: escolhida = gabarito do gerador sintético; rejeitada = pior amostra do SFT.",
    "juiz": "programático: gabarito do review_sintetico + régua do recall.py (linha ±2 e palavra-chave)",
    "politica": {"model": a.model, "temperature": a.temperature, "amostras_por_arquivo": a.amostras},
    "seed": a.seed,
    "arquivos": len(casos),
    "amostras": total_amostras,
    "amostras_perfeitas": perfeitas,
    "pares": len(pares),
    "erros_por_tipo": dict(tipos),
    "segundos_gerando": round(seg, 1),
    "system_prompt_de_inferencia": casos[0][2][0]["content"],
}
(out / "dataset.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(yaml.safe_dump({k: v for k, v in meta.items() if k != "system_prompt_de_inferencia"}, allow_unicode=True))
