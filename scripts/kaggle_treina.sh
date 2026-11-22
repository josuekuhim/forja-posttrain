#!/usr/bin/env bash
# Roda no Kaggle (GPU T4): treina os SFTs de review D1–D3 e empacota os adapters para baixar.
#
# Entrada: o zip de scripts/empacota_kaggle.sh subido como Dataset do Kaggle (o Kaggle descompacta).
# Saída:   /kaggle/working/review-t4-runs.tar.gz (adapters + summary + trainer_state de cada run).
#
#   EXPERIMENTOS="sint real misto" bash .../forja-posttrain/scripts/kaggle_treina.sh
set -euo pipefail

IN=$(find /kaggle/input -maxdepth 5 -type d -name forja-posttrain | head -1)
[ -n "$IN" ] || { echo "forja-posttrain não encontrado em /kaggle/input: suba o forja-kaggle.zip como Dataset"; exit 1; }
W=/kaggle/working/forja
rm -rf "$W" && mkdir -p "$W" && cp -r "$(dirname "$IN")"/. "$W"/
cd "$W/forja-posttrain"

nvidia-smi --query-gpu=name,memory.total --format=csv
python --version
# O torch com CUDA do Kaggle fica; só as bibliotecas de treino são fixadas nas versões usadas na CPU.
# --ignore-requires-python: o pyproject pede 3.12 (lock do Windows); o código não usa nada além do 3.11.
python -c "import urllib.request as u; u.urlopen('https://pypi.org/simple/hatchling/', timeout=15)" 2>/dev/null \
  || { echo "sem acesso ao PyPI: no editor, Settings > Internet > On, e rode de novo"; exit 1; }
# hatchling instalado antes e --no-build-isolation: o pip não monta um venv isolado por pacote local
# (é onde o erro saía sem mensagem) e, se algo falhar, o log mostra qual pacote foi.
pip install -q hatchling editables
pip install -q --no-build-isolation --ignore-requires-python -e ../forja/contracts -e . \
  "transformers==5.23.0" "trl==1.18.1" "peft==0.21.3" "accelerate>=1.0" "datasets>=3" \
  || { echo "pip falhou; repetindo com log completo"; pip install --no-build-isolation --ignore-requires-python -v -e ../forja/contracts 2>&1 | tail -60; exit 1; }
# O torchao do Kaggle (0.10) é velho demais para o peft/transformers fixados e quebra o import; não usamos.
pip uninstall -y -q torchao || true
python -c "import torch; assert torch.cuda.is_available(), 'sem GPU: Settings > Accelerator > GPU T4'; print(torch.__version__, torch.cuda.get_device_name(0))"

if [ -f data/review-real-v0/sft.jsonl ]; then
  mkdir -p data/review-misto-v0
  cat data/review-sintetico-v0/sft.jsonl data/review-real-v0/sft.jsonl > data/review-misto-v0/sft.jsonl
  printf 'nome: review-misto-v0\nuso: treino\ngerado_de: [../review-sintetico-v0, ../review-real-v0]\ndescricao: concatenação (D3)\n' \
    > data/review-misto-v0/dataset.yaml
fi

mkdir -p logs
for x in ${EXPERIMENTOS:-sint real misto}; do
  cfg=configs/review_${x}_lora_t4.yaml
  if ! grep -q . "$(python -c "from forja_posttrain.config import load_config as l; print(l('$cfg').data.train_path)")" 2>/dev/null; then
    echo "== $x: sem dados, pulando"; continue
  fi
  echo "== $x: check-data"
  forja-posttrain check-data "$cfg" > "logs/check-$x.json"
  echo "== $x: treino"
  s=$(date +%s)
  forja-posttrain train "$cfg" > "logs/train-$x.log" 2>&1 &
  pid=$!
  tail -n +1 -F "logs/train-$x.log" --pid=$pid 2>/dev/null | grep --line-buffered -E "'loss'|eval_loss|Error|Traceback" || true
  if ! wait $pid; then
    echo "== $x: treino falhou; fim do log:"; tail -40 "logs/train-$x.log"; exit 1
  fi
  echo "== $x: $(( $(date +%s) - s )) s"
done

# Só o necessário para mesclar e avaliar na máquina local: adapters, metadados e a curva de perda.
estados=$(for d in runs/*/checkpoints; do ls -d "$d"/checkpoint-* | sort -V | tail -1; done | sed 's#$#/trainer_state.json#')
tar czf /kaggle/working/review-t4-runs.tar.gz runs/*/final runs/*/summary.json runs/*/dados.json runs/*/config.yaml $estados logs
ls -la /kaggle/working/review-t4-runs.tar.gz
