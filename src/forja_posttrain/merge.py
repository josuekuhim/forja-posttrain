"""Mescla o adapter LoRA no modelo base e salva em safetensors, pronto para o forja-serve.

O modelo mesclado é um checkpoint comum (sem dependência de PEFT na inferência). Também é o ponto de
partida do DPO (`dpo.policy_path`).

QLoRA: o adapter foi treinado sobre pesos 4-bit, mas a mescla é feita sobre o base em fp16 (não dá para
somar o delta em pesos NF4 sem perda). Isso introduz uma diferença pequena entre o modelo treinado e o
mesclado: o forja-evals mede o mesclado, que é o que vai para produção.
"""

from __future__ import annotations

from pathlib import Path

import torch


def merge_adapter(
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    base_id: str | None = None,
    dtype: str = "float16",
    base_model=None,
    tokenizer=None,
) -> Path:
    """`base_model`/`tokenizer` permitem mesclar sem baixar nada (testes). Sem eles, carrega `base_id`
    (por padrão, o `base_model_name_or_path` gravado no adapter_config.json)."""
    from peft import PeftConfig, PeftModel

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    if base_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_id = base_id or PeftConfig.from_pretrained(str(adapter_dir)).base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype=getattr(torch, dtype))
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(str(adapter_dir))

    merged = PeftModel.from_pretrained(base_model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    # transformers 5: sempre safetensors (o argumento safe_serialization deixou de existir).
    merged.save_pretrained(str(out_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out_dir))
    return out_dir
