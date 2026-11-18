"""Smoke de treino em CPU com modelo Qwen2 minúsculo e tokenizer local (nada é baixado).

Exercita o caminho real: dados -> checagem de contaminação -> SFTTrainer/DPOTrainer do TRL -> PEFT -> salvar
adapter -> mesclar em safetensors. Não mede qualidade; prova que o encanamento funciona antes de gastar GPU.
"""

import json
from pathlib import Path

import pytest
import torch

from forja_posttrain.config import load_config
from forja_posttrain.train import build_trainer, train

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def _cfg(tmp_path, eval_dir, *extra):
    return load_config(
        CONFIG,
        [
            f"out_dir={(tmp_path / 'run').as_posix()}",
            f"data.eval_dataset_dirs=['{eval_dir.as_posix()}']",
            *extra,
        ],
    )


def _snapshot_treinaveis(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def test_smoke_sft_lora_atualiza_so_o_adapter(tmp_path, eval_dir):
    trainer, dados = build_trainer(_cfg(tmp_path, eval_dir, "train.max_steps=2"))
    antes = _snapshot_treinaveis(trainer.model)
    congelado = next(p.detach().clone() for n, p in trainer.model.named_parameters() if "embed_tokens" in n)

    assert antes and all("lora_" in n for n in antes)
    total = sum(p.numel() for p in trainer.model.parameters())
    assert sum(p.numel() for p in antes.values()) < 0.2 * total

    # Loss só na resposta: o TRL 1.18 grava `labels` com -100 no prompt (system + user).
    ex = trainer.train_dataset[0]
    com_loss = sum(1 for t in ex["labels"] if t != -100)
    assert 0 < com_loss < len(ex["input_ids"]) / 2  # o prompt de sistema sozinho é maior que a resposta

    resultado = trainer.train()
    assert resultado.global_step == 2 and torch.isfinite(torch.tensor(resultado.training_loss))
    depois = dict(trainer.model.named_parameters())
    assert any(not torch.equal(antes[n], depois[n]) for n in antes if "lora_B" in n)  # lora_B nasce zerado
    emb = next(p for n, p in trainer.model.named_parameters() if "embed_tokens" in n)
    assert torch.equal(congelado, emb)  # base congelado
    assert dados.relatorio["n_train"] == 9


def test_smoke_train_salva_artefatos_e_merge(tmp_path, eval_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from forja_posttrain.merge import merge_adapter
    from forja_posttrain.tiny import build_tiny_model

    cfg = _cfg(tmp_path, eval_dir)
    summary = train(cfg)
    run = Path(cfg.out_dir)
    for f in [
        "config.yaml",
        "dados.json",
        "summary.json",
        "final/adapter_model.safetensors",
        "final/adapter_config.json",
    ]:
        assert (run / f).exists(), f
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["method"] == "lora"
    assert summary["parametros"]["treinaveis"] < summary["parametros"]["total"]

    # Mescla o adapter no base (o mesmo modelo aleatório, recriado com a mesma seed) e compara saídas.
    tok = AutoTokenizer.from_pretrained(run / "final")
    base = build_tiny_model(tok, seed=cfg.seed)
    from peft import PeftModel

    com_adapter = PeftModel.from_pretrained(build_tiny_model(tok, seed=cfg.seed), str(run / "final")).eval()
    out = merge_adapter(run / "final", run / "merged", base_model=base, tokenizer=tok)
    assert (out / "model.safetensors").exists()
    mesclado = AutoModelForCausalLM.from_pretrained(out).eval()
    ids = torch.tensor([tok("def soma(a, b): return a + b")["input_ids"]])
    with torch.no_grad():
        torch.testing.assert_close(mesclado(ids).logits, com_adapter(ids).logits, atol=1e-4, rtol=1e-4)


def test_smoke_sft_full_treina_todos_os_pesos(tmp_path, eval_dir):
    trainer, _ = build_trainer(
        _cfg(tmp_path, eval_dir, "method=sft_full", "train.max_steps=1", "train.learning_rate=1e-4")
    )
    antes = _snapshot_treinaveis(trainer.model)
    assert sum(p.numel() for p in antes.values()) == sum(p.numel() for p in trainer.model.parameters())
    trainer.train()
    depois = dict(trainer.model.named_parameters())
    assert not torch.equal(
        antes["model.layers.0.mlp.down_proj.weight"], depois["model.layers.0.mlp.down_proj.weight"]
    )


def test_smoke_dpo_lora(tmp_path, eval_dir):
    trainer, dados = build_trainer(_cfg(tmp_path, eval_dir, "method=dpo", "train.max_steps=2"))
    assert dados.relatorio["pares"] == 3
    assert trainer.ref_model is None  # com LoRA, a referência é o modelo com o adapter desligado
    assert {"prompt_ids", "chosen_ids", "rejected_ids"} <= set(trainer.train_dataset.column_names)
    antes = _snapshot_treinaveis(trainer.model)
    resultado = trainer.train()
    assert resultado.global_step == 2
    depois = dict(trainer.model.named_parameters())
    assert any(not torch.equal(antes[n], depois[n]) for n in antes if "lora_B" in n)
    # ~ln 2 no começo: política == referência
    assert resultado.training_loss == pytest.approx(0.693, abs=0.05)


def test_treino_aborta_com_dado_contaminado(tmp_path, eval_dir):
    from conftest import CASOS_EVAL

    from forja_posttrain.data import ContaminacaoError

    sft = tmp_path / "sft.jsonl"
    linha = {
        "id": "x-1",
        "categoria": "recusa",
        "prompt": CASOS_EVAL[1]["input"],
        "resposta": "Não existe.",
        "origem": "t",
        "licenca": "t",
    }
    sft.write_text(json.dumps(linha, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ContaminacaoError):
        build_trainer(_cfg(tmp_path, eval_dir, f"data.train_path={sft.as_posix()}"))
    assert not (tmp_path / "run" / "final").exists()
