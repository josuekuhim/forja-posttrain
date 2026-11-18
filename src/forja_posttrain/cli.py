"""CLI: `forja-posttrain check-data | check-prefs | train | merge`."""

from __future__ import annotations

import argparse
import json
import sys

from forja_posttrain.config import load_config


def _check_data(args: argparse.Namespace) -> int:
    from forja_posttrain.data import prepara_instrucoes

    cfg = load_config(args.config, args.overrides)
    d = cfg.data
    rel = prepara_instrucoes(
        d.train_path,
        d.eval_dataset_dirs,
        dedup_n=d.near_dup_ngram,
        dedup_limiar=d.near_dup_threshold,
        contam_n=d.contamination_ngram,
        contam_limiar=d.contamination_threshold,
        falhar=False,
    )
    print(json.dumps(rel.resumo(), ensure_ascii=False, indent=2))
    if rel.contaminacao:
        print(f"\nCONTAMINAÇÃO: {len(rel.contaminacao)} sobreposição(ões) com a avaliação.", file=sys.stderr)
        return 1
    return 0


def _check_prefs(args: argparse.Namespace) -> int:
    from forja_posttrain.data import prepara_preferencias

    cfg = load_config(args.config, args.overrides)
    d = cfg.data
    prefs = prepara_preferencias(
        d.preference_dir, d.eval_dataset_dirs, n=d.contamination_ngram, limiar=d.contamination_threshold
    )
    print(json.dumps({"pares": len(prefs.pares), "manifest": prefs.manifest}, ensure_ascii=False, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    from forja_posttrain.train import train

    summary = train(load_config(args.config, args.overrides))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _merge(args: argparse.Namespace) -> int:
    from forja_posttrain.merge import merge_adapter

    out = merge_adapter(args.adapter, args.out, base_id=args.base, dtype=args.dtype)
    print(f"modelo mesclado em {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forja-posttrain", description="Pós-treino do modelo base do Forja")
    sub = p.add_subparsers(dest="cmd", required=True)

    for nome, fn, ajuda in [
        (
            "check-data",
            _check_data,
            "dedup + relatório de contaminação do dataset de instrução (exit 1 se contaminado)",
        ),
        ("check-prefs", _check_prefs, "valida proveniência e contaminação dos pares de preferência"),
        ("train", _train, "treina conforme a config (sft_full | lora | qlora | dpo)"),
    ]:
        sp = sub.add_parser(nome, help=ajuda)
        sp.add_argument("config")
        sp.add_argument("overrides", nargs="*", help="secao.chave=valor")
        sp.set_defaults(fn=fn)

    sm = sub.add_parser("merge", help="mescla um adapter LoRA no base e salva safetensors")
    sm.add_argument("adapter")
    sm.add_argument("out")
    sm.add_argument("--base", default=None, help="id/caminho do base (padrão: o gravado no adapter)")
    sm.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    sm.set_defaults(fn=_merge)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
