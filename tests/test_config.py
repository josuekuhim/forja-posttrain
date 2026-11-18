from pathlib import Path

import pytest

from forja_posttrain.config import load_config
from forja_posttrain.train import build_args, build_peft_config, valida

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_overrides():
    cfg = load_config(
        CONFIGS / "smoke.yaml",
        [
            "lora.r=8",
            "train.learning_rate=1e-4",
            "method=dpo",
            "dpo.beta=0.3",
            "lora.target_modules=[q_proj]",
        ],
    )
    assert cfg.lora.r == 8
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert cfg.method == "dpo"
    assert cfg.dpo.beta == pytest.approx(0.3)
    assert cfg.lora.target_modules == ["q_proj"]


def test_chaves_e_valores_invalidos():
    with pytest.raises(ValueError, match="desconhecidas"):
        load_config(CONFIGS / "smoke.yaml", ["train.nao_existe=1"])
    with pytest.raises(ValueError, match="desconhecidas"):
        load_config(CONFIGS / "smoke.yaml", ["secao_nova.x=1"])
    with pytest.raises(ValueError, match="method"):
        load_config(CONFIGS / "smoke.yaml", ["method=rlhf"])
    with pytest.raises(ValueError, match="sem '='"):
        load_config(CONFIGS / "smoke.yaml", ["lora.r"])


@pytest.mark.parametrize(
    "nome,metodo,usa_lora",
    [
        ("sft_lora", "lora", True),
        ("sft_qlora", "qlora", True),
        ("sft_full", "sft_full", False),
        ("dpo", "dpo", True),
    ],
)
def test_configs_reais_carregam(nome, metodo, usa_lora):
    cfg = load_config(CONFIGS / f"{nome}.yaml")
    assert cfg.method == metodo
    assert cfg.usa_lora is usa_lora
    assert isinstance(cfg.train.learning_rate, float)
    assert cfg.train.precision == "fp16"  # alvo é o T4, sem bf16
    assert (build_peft_config(cfg) is not None) is usa_lora


def test_sft_full_fp16_exige_pesos_fp32():
    cfg = load_config(CONFIGS / "sft_full.yaml", ["model.dtype=float16"])
    with pytest.raises(ValueError, match="float32"):
        valida(cfg)


def test_build_args_mapeia_precisao_e_checkpointing():
    cfg = load_config(CONFIGS / "sft_lora.yaml")
    args = build_args(cfg, tem_eval=True)
    assert args.fp16 and not args.bf16  # o TRL liga bf16 por padrão; aqui é explícito
    assert args.activation_checkpointing
    assert args.model_init_kwargs["dtype"] == "float16"
    assert args.gradient_accumulation_steps == 8
    assert args.eval_strategy == "steps"

    smoke = build_args(load_config(CONFIGS / "smoke.yaml"), tem_eval=False)
    assert not smoke.fp16 and not smoke.bf16 and smoke.use_cpu
    assert not smoke.activation_checkpointing
    assert smoke.model_init_kwargs is None  # modelo tiny é instanciado localmente
