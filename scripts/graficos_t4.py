"""Gráficos da rodada T4 (D1 sintético, D2 real, D3 misto), a partir de `results/review-t4-v0.json`.

Rode antes o `scripts/pontua_t4.py`. Usa os mesmos helpers SVG (e a mesma paleta) do `scripts/graficos.py`:

    docs/img/review-t4-real.svg        PR real: bugs achados × arquivos limpos com comentário (ambos de 199)
    docs/img/review-t4-recall.svg      recall no PR sintético × no PR real: o que cada dado ensina

    forja-review/.venv312/Scripts/python.exe forja-posttrain/scripts/graficos_t4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graficos import AZUL, LARANJA, RAIZ, barras_agrupadas  # noqa: E402

CURTO = {  # rótulos em inglês para o eixo (as chaves vêm do results/review-t4-v0.json)
    "Qwen 0.5B base · prompt curto": "Qwen 0.5B base",
    "SFT CPU (sintético, R1)": "SFT CPU, synthetic (R1)",
    "D1 · T4 sintético": "D1 · T4 synthetic",
    "D2 · T4 real": "D2 · T4 real",
    "D3 · T4 misto": "D3 · T4 mixed",
}


def main() -> None:
    rev = json.loads((RAIZ / "results/review-t4-v0.json").read_text(encoding="utf-8"))["revisores"]
    img = RAIZ / "docs/img"

    linhas = [(CURTO.get(r, r), [d["real"]["achados"], d["real"].get("limpos_com_comentario")])
              for r, d in rev.items() if "real" in d]
    (img / "review-t4-real.svg").write_text(barras_agrupadas(
        "Real PR: 199 bugs from fix commits + 199 already-fixed files",
        "Public repositories, commits from 2025–2026 (real training data stops at Sep 2024). More blue and less orange is better.",
        linhas, [("real bugs found", AZUL), ("bug-free files that got a comment", LARANJA)],
        200, 25, "files (out of 199)"), encoding="utf-8")

    pct = lambda d: round(100 * d["achados"] / d["bugs"])  # noqa: E731
    linhas = [(CURTO.get(r, r), [pct(d["teste"]) if "teste" in d else None, pct(d["real"]) if "real" in d else None])
              for r, d in rev.items()]
    (img / "review-t4-recall.svg").write_text(barras_agrupadas(
        "Recall: synthetic PR (held-out templates) × real PR",
        "% of bugs found. Synthetic data teaches the templates; only real data moves the number on real code.",
        linhas, [("synthetic test PR (53 bugs)", AZUL), ("real PR (199 bugs)", LARANJA)],
        50, 10, "% of bugs found"), encoding="utf-8")
    print("ok:", ", ".join(p.name for p in img.glob("review-t4-*.svg")))


if __name__ == "__main__":
    main()
