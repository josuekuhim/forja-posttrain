#!/usr/bin/env bash
# Monta o zip que vira Dataset do Kaggle: código do forja-posttrain, forja-contracts, o seed-v0 (checagem
# de contaminação) e os dados de treino de review. Sem .venv, runs/ nem o PR de teste do sintético.
#
#   bash scripts/empacota_kaggle.sh [saida.zip]     # padrão: D:/forja-kaggle/forja-kaggle.zip
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/d/forja-kaggle/forja-kaggle.zip}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

P=$T/forja-kaggle
mkdir -p "$P/forja" "$P/forja-evals/datasets" "$P/forja-posttrain/data"
cp -r "$R/forja/contracts" "$P/forja/contracts"
rm -rf "$P/forja/contracts/.venv"
cp -r "$R/forja-evals/datasets/forja-seed-v0" "$P/forja-evals/datasets/"
cp -r "$R/forja-posttrain/src" "$R/forja-posttrain/configs" "$R/forja-posttrain/scripts" \
      "$R/forja-posttrain/pyproject.toml" "$R/forja-posttrain/README.md" "$P/forja-posttrain/"
for d in review-sintetico-v0 review-real-v0; do
  if [ -f "$R/forja-posttrain/data/$d/sft.jsonl" ]; then
    mkdir -p "$P/forja-posttrain/data/$d"
    cp "$R/forja-posttrain/data/$d/sft.jsonl" "$R/forja-posttrain/data/$d/dataset.yaml" "$P/forja-posttrain/data/$d/"
  else
    echo "aviso: $d sem sft.jsonl, fora do pacote"
  fi
done
find "$P" -name __pycache__ -prune -exec rm -rf {} +

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
BASE=$(cygpath -m "${OUT%.zip}" 2>/dev/null || echo "${OUT%.zip}")
(cd "$T" && python -c "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', '.', 'forja-kaggle')" "$BASE")
ls -la "$OUT"
