"""Monta modelo, PEFT e o trainer do TRL a partir da config, e treina.

Escrito contra as APIs instaladas (ver README, "Surpresas de API"): transformers 5.23, TRL 1.18, PEFT 0.21.
- `processing_class` (não `tokenizer`) nos trainers do TRL;
- `SFTConfig`/`DPOConfig` ligam bf16 e activation checkpointing por padrão: aqui tudo é explícito;
- `activation_checkpointing` é o nome novo de `gradient_checkpointing` desde o transformers 5.23;
- `quantization_config` vai direto para o trainer quando o modelo é passado como id;
- modelo é carregado com `dtype=` (o `torch_dtype=` foi aposentado no transformers 5).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from forja_posttrain.config import TINY_MODEL_ID, ExperimentConfig, save_config
from forja_posttrain.data import (
    SYSTEM_PROMPT,
    prepara_instrucoes,
    prepara_preferencias,
    to_trl_dpo,
    to_trl_sft,
)

# --------------------------------------------------------------------------- peças


def precision_kwargs(precision: str) -> dict[str, bool]:
    return {"fp16": precision == "fp16", "bf16": precision == "bf16"}


def is_tiny(cfg: ExperimentConfig) -> bool:
    return cfg.model.base_id == TINY_MODEL_ID


def model_id(cfg: ExperimentConfig) -> str:
    """No DPO a política parte do checkpoint SFT mesclado, quando informado."""
    if cfg.method == "dpo" and cfg.dpo.policy_path:
        return cfg.dpo.policy_path
    return cfg.model.base_id


def valida(cfg: ExperimentConfig) -> None:
    """Combinações que quebram só depois de minutos de download/treino: falham aqui, antes."""
    if cfg.method == "sft_full" and cfg.train.precision == "fp16" and cfg.model.dtype == "float16":
        raise ValueError(
            "SFT completo com AMP fp16 precisa de pesos mestres em float32 (model.dtype=float32); "
            "com pesos fp16 o GradScaler recusa desescalar gradientes fp16."
        )
    if cfg.method == "qlora" and not is_tiny(cfg) and importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("method=qlora exige bitsandbytes: `uv sync --extra gpu` (só em GPU NVIDIA).")
    if cfg.method == "qlora" and is_tiny(cfg):
        raise ValueError("qlora não tem caminho tiny/CPU: bitsandbytes 4-bit exige CUDA.")
    if cfg.train.precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU sem bf16 (ex.: T4): use train.precision=fp16.")


def build_tokenizer(cfg: ExperimentConfig, corpus: list[str] | None = None):
    if is_tiny(cfg):
        from forja_posttrain.tiny import build_tiny_tokenizer

        return build_tiny_tokenizer(corpus or [])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id(cfg), revision=cfg.model.revision)
    if tok.pad_token is None:  # o Qwen2.5 já define <|endoftext|> como pad; outros modelos podem não definir
        tok.pad_token = tok.eos_token
    return tok


def build_quantization_config(cfg: ExperimentConfig):
    if cfg.method != "qlora":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant.quant_type,
        bnb_4bit_use_double_quant=cfg.quant.double_quant,
        bnb_4bit_compute_dtype=getattr(torch, cfg.quant.compute_dtype),
    )


def build_peft_config(cfg: ExperimentConfig):
    if not cfg.usa_lora:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )


def model_init_kwargs(cfg: ExperimentConfig) -> dict[str, Any]:
    kw: dict[str, Any] = {"dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation}
    if cfg.model.revision:
        kw["revision"] = cfg.model.revision
    return kw


def build_args(cfg: ExperimentConfig, *, tem_eval: bool):
    t = cfg.train
    comum: dict[str, Any] = dict(
        output_dir=str(Path(cfg.out_dir) / "checkpoints"),
        seed=cfg.seed,
        data_seed=cfg.seed,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum,
        warmup_steps=t.warmup if t.warmup < 1 else int(t.warmup),
        lr_scheduler_type=t.scheduler,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        optim=t.optim,
        logging_steps=t.logging_steps,
        save_strategy="steps" if t.save_steps > 0 else "no",
        save_steps=max(t.save_steps, 1),
        save_total_limit=2,
        eval_strategy="steps" if tem_eval else "no",
        eval_steps=max(t.save_steps, 1) if tem_eval else None,
        activation_checkpointing=t.activation_checkpointing,
        use_cpu=t.use_cpu,
        report_to=t.report_to,
        run_name=cfg.name,
        max_length=t.max_length,
        dataloader_num_workers=0,
        **precision_kwargs(t.precision),
    )
    if t.activation_checkpointing:
        # Não reentrante: é o recomendado pelo PyTorch e funciona com PEFT sem `enable_input_require_grads`.
        comum["activation_checkpointing_kwargs"] = {"use_reentrant": False}
    # O modelo é passado como id (não instância) para o TRL aplicar dtype/quantização ao carregar.
    if not is_tiny(cfg):
        comum["model_init_kwargs"] = model_init_kwargs(cfg)

    if cfg.method == "dpo":
        from trl import DPOConfig

        return DPOConfig(beta=cfg.dpo.beta, loss_type=[cfg.dpo.loss_type], **comum)
    from trl import SFTConfig

    # completion_only_loss=None: o padrão do TRL para prompt/completion é calcular a loss só na resposta.
    return SFTConfig(packing=False, completion_only_loss=None, **comum)


def _split(rows: list[dict], frac: float, seed: int) -> tuple[Dataset, Dataset | None]:
    ds = Dataset.from_list(rows)
    n_eval = int(len(rows) * frac)
    if n_eval < 1:
        return ds, None
    sp = ds.train_test_split(test_size=n_eval, seed=seed)
    return sp["train"], sp["test"]


@dataclass
class DadosPreparados:
    train: Dataset
    eval: Dataset | None
    relatorio: dict[str, Any] = field(default_factory=dict)
    textos: list[str] = field(default_factory=list)  # usados só para treinar o tokenizer do caminho tiny


def build_datasets(cfg: ExperimentConfig) -> DadosPreparados:
    d = cfg.data
    system = d.system_prompt if d.system_prompt is not None else SYSTEM_PROMPT
    if cfg.method == "dpo":
        prefs = prepara_preferencias(
            d.preference_dir, d.eval_dataset_dirs, n=d.contamination_ngram, limiar=d.contamination_threshold
        )
        rows = [to_trl_dpo(p, system) for p in prefs.pares]
        textos = [p.prompt + p.escolhida + p.rejeitada for p in prefs.pares]
        rel = {"pares": len(rows), "manifest": prefs.manifest}
    else:
        rep = prepara_instrucoes(
            d.train_path,
            d.eval_dataset_dirs,
            dedup_n=d.near_dup_ngram,
            dedup_limiar=d.near_dup_threshold,
            contam_n=d.contamination_ngram,
            contam_limiar=d.contamination_threshold,
        )
        rows = [to_trl_sft(e, system) for e in rep.dedup.mantidos]
        textos = [m.content for e in rep.dedup.mantidos for m in e.conversa()]
        rel = rep.resumo()
    train, ev = _split(rows, cfg.train.eval_fraction, cfg.seed)
    rel["n_train"], rel["n_eval"] = len(train), (len(ev) if ev is not None else 0)
    return DadosPreparados(train=train, eval=ev, relatorio=rel, textos=[system, *textos])


def build_trainer(cfg: ExperimentConfig, dados: DadosPreparados | None = None):
    """Devolve (trainer, dados). Não treina."""
    from transformers import set_seed

    valida(cfg)
    set_seed(cfg.seed)
    dados = dados or build_datasets(cfg)
    tok = build_tokenizer(cfg, dados.textos)
    args = build_args(cfg, tem_eval=dados.eval is not None)
    peft_config = build_peft_config(cfg)

    if is_tiny(cfg):
        from forja_posttrain.tiny import build_tiny_model

        model: Any = build_tiny_model(tok, seed=cfg.seed)
    else:
        model = model_id(cfg)

    if cfg.method == "dpo":
        from trl import DPOTrainer

        ref_model = None
        if is_tiny(cfg) and peft_config is None:
            ref_model = copy.deepcopy(model)  # sem LoRA, a referência é uma cópia congelada
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=dados.train,
            eval_dataset=dados.eval,
            processing_class=tok,
            peft_config=peft_config,
            quantization_config=build_quantization_config(cfg),
        )
    else:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dados.train,
            eval_dataset=dados.eval,
            processing_class=tok,
            peft_config=peft_config,
            quantization_config=build_quantization_config(cfg),
        )

    if len(trainer.train_dataset) == 0:
        # O TRL descarta em silêncio exemplos cuja resposta some no truncamento (max_length).
        raise ValueError(
            f"nenhum exemplo sobrou após tokenizar/truncar (train.max_length={cfg.train.max_length}). "
            "Aumente max_length ou encurte os exemplos."
        )

    if cfg.method == "qlora" and cfg.train.precision == "fp16":
        # O TRL 1.18 converte os adapters de QLoRA para bf16 (recomendação do paper). No T4 não há bf16
        # nativo e o GradScaler do fp16 espera parâmetros fp32: voltamos os treináveis para fp32.
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    return trainer, dados


def conta_parametros(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "treinaveis": treinaveis}


def _versoes() -> dict[str, str]:
    import peft
    import transformers
    import trl

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
    }


def train(cfg: ExperimentConfig) -> dict[str, Any]:
    """Treina e salva em `out_dir`: config.yaml resolvida, dados.json, final/ (adapter ou modelo), summary.json."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    trainer, dados = build_trainer(cfg)
    (out / "dados.json").write_text(
        json.dumps(dados.relatorio, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    resultado = trainer.train()
    dur = time.perf_counter() - t0

    final = out / "final"
    trainer.save_model(str(final))  # com PEFT salva só o adapter; sem PEFT, o modelo inteiro em safetensors

    summary = {
        "name": cfg.name,
        "method": cfg.method,
        "base": model_id(cfg),
        "seed": cfg.seed,
        "train_loss": resultado.training_loss,
        "global_step": resultado.global_step,
        "segundos": round(dur, 2),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "parametros": conta_parametros(trainer.model),
        "n_train": dados.relatorio.get("n_train"),
        "n_eval": dados.relatorio.get("n_eval"),
        "versoes": _versoes(),
        "artefato": str(final),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
