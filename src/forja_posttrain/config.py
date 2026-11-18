"""Configuração tipada do pós-treino: YAML + overrides `secao.chave=valor` (mesma convenção do forja-pretrain)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

Metodo = Literal["sft_full", "lora", "qlora", "dpo"]
Precisao = Literal["fp32", "fp16", "bf16"]

# Valor especial de `model.base_id`: modelo Qwen2 minúsculo, aleatório, com tokenizer construído localmente.
# Usado no smoke test e nos testes; nunca baixa nada.
TINY_MODEL_ID = "tiny-random"


@dataclass
class ModelConfig:
    base_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # ADR 0001; plano B: Qwen/Qwen2.5-Coder-0.5B-Instruct
    # Dtype dos pesos ao carregar. No T4 use float16 (o T4 não tem bf16 nativo); "auto" lê o config.json.
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "float16"
    attn_implementation: str = "sdpa"  # flash-attn não roda no T4 (exige Ampere+)
    revision: str | None = None  # fixe um commit do Hub para reprodutibilidade


@dataclass
class LoraParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Todas as projeções lineares dos blocos do Qwen2 (atenção + MLP), como no paper do QLoRA.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"


@dataclass
class QuantParams:
    """Só usado em `method: qlora` (exige o extra `gpu`, com bitsandbytes)."""

    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["float16", "bfloat16"] = "float16"  # float16 no T4


@dataclass
class TrainParams:
    learning_rate: float = 2e-4
    num_epochs: float = 3.0
    max_steps: int = -1  # > 0 sobrescreve num_epochs
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    warmup: float = (
        0.03  # < 1 = fração dos passos; >= 1 = número de passos (transformers 5 removeu warmup_ratio)
    )
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    precision: Precisao = "fp16"
    activation_checkpointing: bool = True
    optim: str = "adamw_torch"
    logging_steps: int = 10
    save_steps: int = 200
    eval_fraction: float = 0.05  # fatia do treino separada para loss de validação (não é o forja-evals)
    use_cpu: bool = False
    report_to: str = "none"  # "wandb" exige o extra `track`


@dataclass
class DPOParams:
    beta: float = 0.1
    loss_type: str = "sigmoid"
    # Política de partida: um checkpoint SFT já mesclado (saída do `forja-posttrain merge`). None = model.base_id.
    policy_path: str | None = None
    use_lora: bool = True  # com LoRA, a referência é o próprio modelo com o adapter desligado (sem 2ª cópia)


@dataclass
class DataConfig:
    train_path: str = "data/exemplo/sft.jsonl"
    preference_dir: str = "data/exemplo/preferencias"
    # Datasets do forja-evals contra os quais a contaminação é checada. Treino que vaza aqui aborta.
    eval_dataset_dirs: list[str] = field(default_factory=lambda: ["../forja-evals/datasets/forja-seed-v0"])
    contamination_ngram: int = 5
    contamination_threshold: float = 0.5
    near_dup_ngram: int = 5
    near_dup_threshold: float = 0.8
    system_prompt: str | None = None  # None = o mesmo prompt de sistema que o forja-evals usa na inferência


@dataclass
class ExperimentConfig:
    name: str = "default"
    method: Metodo = "lora"
    seed: int = 42
    out_dir: str = "runs/default"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraParams = field(default_factory=LoraParams)
    quant: QuantParams = field(default_factory=QuantParams)
    train: TrainParams = field(default_factory=TrainParams)
    dpo: DPOParams = field(default_factory=DPOParams)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if self.method not in ("sft_full", "lora", "qlora", "dpo"):
            raise ValueError(f"method inválido: {self.method!r}")
        if self.train.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"train.precision inválida: {self.train.precision!r}")

    @property
    def usa_lora(self) -> bool:
        return self.method in ("lora", "qlora") or (self.method == "dpo" and self.dpo.use_lora)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECOES = {
    "model": ModelConfig,
    "lora": LoraParams,
    "quant": QuantParams,
    "train": TrainParams,
    "dpo": DPOParams,
    "data": DataConfig,
}
_TOPO = {"name", "method", "seed", "out_dir"}


def _build(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Chaves desconhecidas em {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):  # PyYAML (YAML 1.1) lê "1e-4" como string
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed


def _coerce_floats(raw: dict[str, Any]) -> None:
    """Mesmo problema do "1e-4" dentro do YAML: converte strings numéricas em campos float."""
    for nome, cls in _SECOES.items():
        sec = raw.get(nome) or {}
        for f in fields(cls):
            if f.type == "float" and isinstance(sec.get(f.name), str):
                sec[f.name] = float(sec[f.name])


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Carrega YAML e aplica overrides no formato `secao.chave=valor` (valor em YAML)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for ov in overrides or []:
        key, sep, value = ov.partition("=")
        if not sep:
            raise ValueError(f"override sem '=': {ov!r}")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_scalar(value)
    unknown = set(raw) - _TOPO - set(_SECOES)
    if unknown:
        raise ValueError(f"Chaves desconhecidas no topo da config: {sorted(unknown)}")
    _coerce_floats(raw)
    return ExperimentConfig(
        name=raw.get("name", Path(path).stem),
        method=raw.get("method", "lora"),
        seed=int(raw.get("seed", 42)),
        out_dir=raw.get("out_dir", f"runs/{raw.get('name', Path(path).stem)}"),
        **{nome: _build(cls, raw.get(nome)) for nome, cls in _SECOES.items()},
    )


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
