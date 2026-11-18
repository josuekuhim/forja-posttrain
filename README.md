# forja-posttrain

**Pós-treino do modelo base do [Forja](../forja): SFT completo, LoRA, QLoRA e DPO no mesmo modelo, todos medidos pela mesma régua ([forja-evals](../forja-evals)).**

> Resultado: _pendente_. A qualidade por categoria (forja-evals), o custo, o tempo, a VRAM e a regressão em benchmarks gerais de cada método entram aqui.

## O ponto de partida

O primeiro baseline real do forja-evals: Qwen2.5-Coder-1.5B-Instruct (GGUF Q4_K_M, CPU, prompt de sistema padrão, sem juiz) no `forja-seed-v0`:

| Categoria | Score | O que aconteceu |
| --- | --- | --- |
| correcao | 3/4 | No palíndromo, afirmou que o código já estava correto |
| deteccao_bug | 0/4 | Diagnóstico errado nos quatro |
| recusa | 0/3 | Inventou o retorno de uma função que não viu e um parâmetro que não existe |

**Score geral 0,273.** O modelo base alucina com confiança e não acha bugs. É isso que o pós-treino precisa corrigir: explicar código, achar bugs, sugerir correções, responder sobre APIs e **dizer "não sei" quando não sabe**, tudo em português.

## Papel no sistema

```
forja-pretrain ──► forja-posttrain ──► forja-serve ──► forja-review
                        │    ▲
          checkpoints   ▼    │  pares de preferência (só de datasets uso: treino)
                     forja-evals (a régua)
```

- **Entrada:** o modelo base do [ADR 0001](../forja/docs/adr/0001-modelo-base.md) (Qwen2.5-Coder-1.5B-Instruct; 0.5B como plano B, mesmo tokenizer), um dataset de instrução e pares de preferência (`PreferencePair` do `forja-contracts`, gerados pelo `forja-evals pairs`).
- **Saída:** checkpoints em safetensors (adapter LoRA e modelo mesclado) para o `forja-serve`, cada um com `config.yaml` resolvida, seed, versões e `summary.json`.
- **Régua:** todo checkpoint é servido num endpoint OpenAI-compatível e avaliado pelo forja-evals num conjunto que **nunca** encosta no treino (checagem de contaminação explícita, abaixo).

## Arquitetura

| Etapa | O que faz | Arquivo |
| --- | --- | --- |
| Config | Dataclasses tipadas, YAML + overrides `secao.chave=valor` (mesma convenção do forja-pretrain) | `config.py` |
| Dados de instrução | Schema `InstructionExample` (`messages` ou `prompt`+`resposta`, `categoria`, `origem`, `licenca`), normalização (NFC, LF, espaços) | `data.py` |
| Dedup | Exata (SHA-256 da conversa normalizada) + near-dup (Jaccard de 5-gramas de palavra ≥ 0,8) | `data.py` |
| Contaminação | Contenção de n-gramas de cada exemplo contra cada `EvalCase` do forja-evals: entrada vs. `input` e resposta vs. `referencia`. **Aborta o treino** se passar do limiar | `data.py` |
| Preferências | `PreferencePair.to_trl()` → formato conversacional; recusa pares cujo `dataset.yaml` (ou o de origem, em `gerado_de`) diz `uso: avaliacao` | `data.py` |
| Treino | `SFTTrainer` / `DPOTrainer` do TRL, LoRA via PEFT, QLoRA via bitsandbytes NF4 | `train.py` |
| Mescla | Adapter + base → safetensors para o forja-serve e para ser a política inicial do DPO | `merge.py` |
| Smoke | Qwen2 minúsculo e aleatório + tokenizer BPE treinado na hora: o encanamento inteiro roda em CPU sem baixar nada | `tiny.py` |

```
data/sft.jsonl ─► normaliza ─► dedup ─► contaminação ✗ aborta ─► prompt/completion ─► SFTTrainer ─► final/ (adapter)
                                            ▲                                                          │ merge
                         forja-evals/datasets/*/cases.jsonl                                            ▼
data/preferencias/ ─► proveniência (uso: treino?) ─► contaminação ─► DPOTrainer (política = SFT mesclado) ─► final/
```

## Decisões e trade-offs

- **Loss só na resposta.** O SFT usa o formato *prompt/completion* conversacional do TRL: a loss cai só na última fala do assistant (o prompt de sistema e a pergunta viram `-100`). Não depende de `{% generation %}` no chat template, que o template original do Qwen2.5 não tem.
- **O mesmo prompt de sistema do forja-evals.** O exemplo de treino recebe exatamente o `SYSTEM_PROMPT` que a régua usa na inferência; um teste confere a igualdade com o repo vizinho. Treinar com um prompt e avaliar com outro mediria a diferença de prompt, não o treino. (Pendência: subir o texto para o `forja-contracts`.)
- **Contaminação é erro, não aviso.** Usa *contenção* (fração dos n-gramas do caso de eval presentes no exemplo), não Jaccard: um caso de eval colado dentro de um exemplo maior tem Jaccard baixo, mas contenção 1. Checa também o gabarito (`referencia`) contra as respostas de treino.
- **Proveniência obrigatória para o DPO.** Sem `dataset.yaml` com `uso: treino` e `gerado_de`, não há DPO. É a mesma regra anti-circularidade do `forja-evals pairs`, checada de novo do lado de quem consome.
- **Alvo de hardware: T4 de 16 GB** (Kaggle/Colab gratuito). Consequências: fp16 em vez de bf16 (Turing não tem bf16 nativo), SDPA em vez de flash-attn, activation checkpointing ligado, batch 2 × accum 8. O vocabulário de 152k torna os **logits** o maior consumidor de ativação.
- **SFT completo do 1,5B não cabe no T4** (~24,6 GB só de pesos + gradientes + Adam em fp32). Por isso a comparação justa full vs. LoRA vs. QLoRA roda no **0.5B** (E1), e no 1,5B comparamos LoRA vs. QLoRA (E2). Estimativas de VRAM estão comentadas em cada `configs/*.yaml`; os números medidos vão para `summary.json`.
- **DPO sobre o SFT mesclado, com LoRA.** A referência é o próprio modelo com o adapter desligado: nenhuma segunda cópia na GPU.
- **Mesma régua, mesmo serving.** O baseline 0,273 é de um GGUF Q4_K_M em CPU. Comparar com um checkpoint fp16 servido por vLLM mistura duas variáveis (quantização e pós-treino). Cada linha da tabela de resultados declara o serving; ou o base é reavaliado no mesmo serving, ou o checkpoint é convertido para GGUF Q4_K_M.
- **Esqueleto primeiro.** Este repo é o terreno preparado: o pipeline roda de ponta a ponta em CPU com dados de exemplo; o treino real roda no Kaggle/Colab.

## Como reproduzir

### Local (CPU, só o encanamento)

```bash
uv sync                                   # sem bitsandbytes/wandb: ficam nos extras gpu/track
uv run pytest -q                          # 27 testes, ~30 s em CPU, 100% offline
uv run forja-posttrain check-data configs/smoke.yaml      # dedup + contaminação (exit 1 se contaminado)
uv run forja-posttrain check-prefs configs/smoke.yaml     # proveniência + contaminação dos pares
uv run forja-posttrain train configs/smoke.yaml           # SFT LoRA, 3 passos, modelo minúsculo
uv run forja-posttrain train configs/smoke.yaml method=dpo out_dir=runs/smoke-dpo
```

Os repositórios ficam lado a lado: o `forja-contracts` vem de `../forja/contracts` e as configs apontam a checagem de contaminação para `../forja-evals/datasets/`.

### Kaggle / Colab (T4), passo a passo

1. **Notebook com GPU.** Kaggle: *Settings → Accelerator → GPU T4* e *Internet on*. Colab: *Ambiente de execução → T4*.
2. **Clonar os três repositórios lado a lado** (o contracts e os datasets de avaliação são dependências):
   ```bash
   git clone https://github.com/<usuario>/forja
   git clone https://github.com/<usuario>/forja-evals
   git clone https://github.com/<usuario>/forja-posttrain
   cd forja-posttrain
   ```
3. **Instalar reaproveitando o torch com CUDA que já vem no ambiente** (o `uv.lock` foi resolvido no Windows/CPU; no notebook o pip é mais rápido):
   ```bash
   pip install -q -e ../forja/contracts -e ".[gpu]"     # gpu = bitsandbytes (QLoRA, paged_adamw_8bit)
   ```
4. **Dados.** Suba `sft.jsonl` (e `preferencias/` com `dataset.yaml`) como dataset do Kaggle / Drive e copie para `data/`.
5. **Checar antes de gastar GPU:**
   ```bash
   forja-posttrain check-data configs/sft_lora.yaml
   ```
6. **Treinar** (um método por sessão; o `summary.json` registra VRAM de pico, tempo e versões):
   ```bash
   forja-posttrain train configs/sft_lora.yaml
   forja-posttrain train configs/sft_qlora.yaml
   forja-posttrain train configs/sft_full.yaml           # 0.5B; 1,5B só com 24 GB+
   ```
7. **Mesclar** o adapter para servir (e para ser a política do DPO):
   ```bash
   forja-posttrain merge runs/sft-lora/final runs/sft-lora/merged
   forja-posttrain train configs/dpo.yaml                # dpo.policy_path=runs/sft-lora/merged
   ```
8. **Avaliar com o forja-evals.** Sirva o modelo mesclado num endpoint OpenAI-compatível (ex.: `vllm serve runs/sft-lora/merged --dtype half --port 8000`, ou converta para GGUF e use o `llama-server`, como no baseline) e rode `forja-evals run` com uma config apontando para esse endpoint. Compare com `forja-evals compare` contra o baseline **servido do mesmo jeito**.
9. **Guardar os artefatos.** `runs/<nome>/` (config resolvida, `dados.json`, `summary.json`, `final/`) vai para o output do Kaggle ou para o Hub, com model card linkando o ID da execução do forja-evals.

Cada execução salva em `runs/<nome>/`: `config.yaml` resolvida, `dados.json` (dedup, contaminação, contagem por categoria), `summary.json` (loss, passos, tempo, VRAM, parâmetros treináveis, versões) e `final/` (adapter ou modelo completo, em safetensors).

## Experimentos planejados (uma variável por vez)

Todos com a mesma seed, o mesmo dataset (hash registrado) e avaliados pelo forja-evals num conjunto mantido fora do treino.

| # | Comparação | Modelo | Qualidade por categoria (forja-evals) | Custo (GPU·h) | Tempo | VRAM pico | Regressão geral (lm-evaluation-harness) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Base + prompt (baseline) | 1,5B | 0,273 (seed-v0, sem juiz) | — | — | — | referência |
| E1 | SFT completo vs. LoRA vs. QLoRA | 0,5B | — | — | — | — | — |
| E2 | LoRA vs. QLoRA | 1,5B | — | — | — | — | — |
| E3 | SFT (LoRA) vs. SFT + DPO | 1,5B | — | — | — | — | — |
| E4 | Rank do LoRA: 8 · 16 · 64 | 1,5B | — | — | — | — | — |

- **Qualidade por categoria:** `explicacao`, `deteccao_bug`, `correcao`, `api`, `recusa`, com IC do bootstrap pareado do forja-evals. A meta mais visível: tirar `recusa` e `deteccao_bug` do zero.
- **Regressão geral:** o pós-treino não pode destruir o que o modelo já sabia. Rodar o `lm-evaluation-harness` antes e depois em tarefas gerais (ex.: ARC, HellaSwag) e de código (HumanEval, MBPP), mais uma tarefa em português.
- **Custo:** GPU·h × preço de referência da GPU; no Kaggle/Colab gratuito o custo é a cota.

## Surpresas de API (transformers 5.23, TRL 1.18, PEFT 0.21)

Verificadas no código instalado, não na documentação:

- `SFTConfig`/`DPOConfig` ligam **bf16 por padrão** (se `fp16` não estiver setado). No T4 isso quebra; aqui a precisão é sempre explícita.
- O TRL liga **activation checkpointing por padrão**, e desde o transformers 5.23 o nome é `activation_checkpointing` (`gradient_checkpointing` virou alias deprecado).
- `warmup_ratio` **não existe mais** no transformers 5: `warmup_steps` aceita fração (< 1) ou número de passos.
- `torch_dtype=` virou `dtype=` no `from_pretrained`; `save_pretrained` não tem mais `safe_serialization` (sempre safetensors).
- Os trainers recebem `processing_class` (não `tokenizer`) e `quantization_config` direto no construtor.
- `DPOConfig` não tem mais `max_prompt_length`; `loss_type` é uma **lista**.
- Com QLoRA, o TRL converte os parâmetros do adapter para **bf16**. No T4 com fp16 isso conflita com o GradScaler; `train.py` devolve os treináveis para fp32 nesse caso.
- O TRL **descarta em silêncio** exemplos cuja resposta some no truncamento; `train.py` falha se o dataset ficar vazio.
- Windows: o Smart App Control bloqueou a DLL do `regex` 2026.11.19 (binário recém-publicado). O `pyproject.toml` restringe `regex<2026`.

## Testes

27 testes, ~30 s em CPU, sem rede (`HF_HUB_OFFLINE=1` no `conftest.py`):

- Config: overrides, chaves desconhecidas, as quatro configs reais carregam, mapeamento de precisão/checkpointing para o `SFTConfig`.
- Dados: schema, normalização, dedup exata e near-dup, contaminação (caso de eval plantado dentro de um exemplo maior, gabarito vazado, dados limpos passam, exemplos não contaminam o `forja-seed-v0`), recusa de pares de `uso: avaliacao` e sem proveniência.
- Smoke de treino com o Qwen2 minúsculo: SFT LoRA (só o adapter muda, base congelado, loss só na resposta), SFT completo (todos os pesos mudam), DPO LoRA (sem cópia da referência, loss ≈ ln 2 no início), salvar + mesclar (logits do modelo mesclado = logits base + adapter) e treino abortado por contaminação.

## Limitações

- **Ainda não treinou nada de verdade.** É o terreno preparado; os números vêm do Kaggle/Colab.
- Os dados em `data/exemplo/` são 9 exemplos e 3 pares escritos à mão: provam o encanamento, não ensinam nada.
- Near-dup O(N²) por Jaccard exato; MinHash + LSH entra na v2, quando o dataset passar de alguns milhares de exemplos.
- A contaminação é léxica (n-gramas). Uma paráfrase de um caso de eval passa; checagem semântica (embeddings) fica para a v2.
- QLoRA não tem caminho em CPU (bitsandbytes 4-bit exige CUDA): no smoke só o LoRA, o SFT completo e o DPO rodam.
- O `uv.lock` foi resolvido no Windows/CPU; no notebook a instalação é por pip, então as versões exatas ficam no `summary.json` de cada execução.
- O prompt de sistema é uma cópia do do forja-evals (um teste guarda a igualdade); o certo é morar no `forja-contracts`.
