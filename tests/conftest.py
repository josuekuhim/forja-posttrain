"""Testes 100% offline: qualquer tentativa de acessar o Hugging Face Hub falha em vez de baixar."""

import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402

CASOS_EVAL = [
    {
        "id": "deteccao_bug-0001",
        "categoria": "deteccao_bug",
        "input": "Esse código quebra quando a fila está vazia. Por quê?\n\n```python\ndef proximo(fila):\n    return fila.pop(0) if fila else fila[0]\n```",
        "referencia": "A condição está invertida: quando a fila está vazia ele acessa fila[0], que levanta IndexError.",
        "metadados": {"linguagem": "python", "origem": "teste"},
    },
    {
        "id": "recusa-0001",
        "categoria": "recusa",
        "input": "Qual parâmetro do `sorted` ativa o modo hiper-rápido de ordenação paralela?",
        "referencia": "Nenhum: o sorted do Python não tem modo paralelo; os parâmetros são key e reverse.",
        "metadados": {"linguagem": "python", "origem": "teste"},
    },
]


def escreve_eval(dirpath, uso="avaliacao", casos=CASOS_EVAL):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "dataset.yaml").write_text(f"nome: {dirpath.name}\nuso: {uso}\n", encoding="utf-8")
    with (dirpath / "cases.jsonl").open("w", encoding="utf-8") as f:
        for c in casos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return dirpath


@pytest.fixture
def eval_dir(tmp_path):
    """Dataset de avaliação temporário, no formato do forja-evals (não depende do repo vizinho)."""
    return escreve_eval(tmp_path / "eval-teste")
