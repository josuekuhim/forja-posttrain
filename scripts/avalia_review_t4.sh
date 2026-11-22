#!/usr/bin/env bash
# Avalia na CPU local os SFTs de review treinados na T4 (D1 sint, D2 real, D3 misto) e as linhas de base no
# dado real. Para cada modelo: sobe o forja-serve (fp32), confere em /v1/models quem está servindo, roda o
# forja-review nos PRs (teste sintético, holdout, plantados, review-eval-v0 real) e o forja-seed-v0.
# Retomável: pula o que já tem JSON em forja-review/results.
#
#   bash forja-posttrain/scripts/avalia_review_t4.sh          # log em D:/forja-kaggle/eval-t4
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
SERVE=$R/forja-serve/.venv/Scripts/forja-serve.exe
REV=$R/forja-review/.venv312/Scripts/forja-review.exe
PY=$R/forja-review/.venv312/Scripts/python.exe
EVALS=$R/forja-evals/.venv312/Scripts/forja-evals.exe
LOG=${LOG:-/d/forja-kaggle/eval-t4}
M=D:/ai-ml-models
mkdir -p "$LOG"
export HF_HOME=$M/hf HF_HUB_OFFLINE=1 FORJA_REVIEW_MAX_ARQUIVOS=1000 FORJA_REVIEW_CONCORRENCIA=8 PYTHONIOENCODING=utf-8
URL=http://127.0.0.1:8000/v1
declare -A DIFF=(
  [teste]=$R/forja-posttrain/data/review-sintetico-v0/teste/pr-teste.diff
  [holdout]=$R/forja-review/examples/pr-holdout.diff
  [bugs]=$R/forja-review/examples/pr-bugs-plantados.diff
  [real]=$R/forja-evals/datasets/review-eval-v0/pr-eval.diff
)

sobe() {  # caminho nome
  if curl -s -m 3 $URL/models >/dev/null; then echo "porta 8000 ocupada: pare o servidor antes"; exit 1; fi
  "$SERVE" serve "$1" --name "$2" --port 8000 --dtype fp32 > "$LOG/serve-$2.log" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do
    curl -s -m 3 $URL/models | grep -q "\"$2\"" && return 0
    kill -0 $SPID 2>/dev/null || { echo "forja-serve morreu"; tail "$LOG/serve-$2.log"; exit 1; }
    sleep 2
  done
  echo "forja-serve não subiu"; exit 1
}
desce() {
  local w; w=$(cat /proc/$SPID/winpid 2>/dev/null)
  kill $SPID 2>/dev/null; [ -n "$w" ] && taskkill //F //T //PID "$w" >/dev/null 2>&1
  wait $SPID 2>/dev/null; sleep 3
}
revisa() {  # sufixo nome-do-modelo prs...
  local suf=$1 nome=$2; shift 2
  for pr in "$@"; do
    out=$R/forja-review/results/$pr-$suf.json
    [ -s "$out" ] && { echo "== $suf/$pr: já existe"; continue; }
    echo "== $suf/$pr: $(date +%H:%M)"
    s=$(date +%s)
    # PR real: prompts de ~1.300 tokens. Com 8 em paralelo cada pedido passava de 10 min e estourava o timeout.
    local conc=8 tmo=600; [ "$pr" = real ] && conc=4 tmo=1800
    (cd "$R/forja-review" && FORJA_REVIEW_CONCORRENCIA=$conc FORJA_REVIEW_LLM_TIMEOUT_S=$tmo "$REV" local --diff "${DIFF[$pr]}" --base-url $URL --model "$nome" --prompt curto \
       --json-out "$out" > "$LOG/review-$pr-$suf.log" 2>&1) || { echo "falhou"; tail "$LOG/review-$pr-$suf.log"; }
    echo "   $(( $(date +%s) - s )) s"
  done
}
seed() {  # x
  local cfg=configs/forja-serve-qwen05-review-$1-t4.yaml
  ls -d "$R"/forja-evals/runs/*-review-$1-t4-fp32 >/dev/null 2>&1 && { echo "== seed $1: já existe"; return; }
  echo "== seed $1: $(date +%H:%M)"
  (cd "$R/forja-evals" && "$EVALS" run "$cfg" > "$LOG/seed-$1.log" 2>&1) || { echo "falhou"; tail "$LOG/seed-$1.log"; }
}
precisa() { for pr in "${@:2}"; do [ -s "$R/forja-review/results/$pr-$1.json" ] || return 0; done; return 1; }

# 1) Os três SFTs da T4: PRs rápidos primeiro, depois o seed-v0, por último o PR real (o mais longo).
for x in sint real misto; do
  suf=qwen0.5b-t4-$x-curto
  if precisa $suf teste holdout bugs real || ! ls -d "$R"/forja-evals/runs/*-review-$x-t4-fp32 >/dev/null 2>&1; then
    sobe "$M/review-$x-t4-merged" "review-$x-t4"
    revisa $suf "review-$x-t4" teste holdout bugs
    seed $x
    revisa $suf "review-$x-t4" real
    desce
  fi
done

# 2) Linhas de base no dado real: o base com o prompt curto e o SFT de CPU (sintético, R1).
if precisa qwen0.5b-base-curto real; then
  sobe Qwen/Qwen2.5-Coder-0.5B-Instruct qwen0.5b-base; revisa qwen0.5b-base-curto qwen0.5b-base real; desce
fi
if precisa qwen0.5b-sft-curto real; then
  sobe "$M/review-lora-cpu-merged" review-sft; revisa qwen0.5b-sft-curto review-sft real; desce
fi

# 3) ruff no dado real (mesma régua, sem modelo).
[ -s "$R/forja-review/results/real-ruff.json" ] || (cd "$R/forja-review" && "$PY" examples/linters.py "${DIFF[real]}" \
  --ferramenta ruff --json-out results/real-ruff.json > "$LOG/ruff-real.log" 2>&1)
echo "== fim: $(date +%H:%M)"
