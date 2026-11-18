import ast
import json
from pathlib import Path

import pytest
from conftest import CASOS_EVAL, escreve_eval
from forja_contracts import PreferencePair

from forja_posttrain.data import (
    SYSTEM_PROMPT,
    ContaminacaoError,
    DatasetDeAvaliacaoError,
    InstructionExample,
    dedup,
    load_instructions,
    load_preferences,
    prepara_instrucoes,
    prepara_preferencias,
    to_trl_dpo,
    to_trl_sft,
)

RAIZ = Path(__file__).resolve().parents[1]
EXEMPLO = RAIZ / "data" / "exemplo"
SEED_V0 = RAIZ.parent / "forja-evals" / "datasets" / "forja-seed-v0"


def _ex(i, prompt, resposta="Resposta qualquer.", categoria="explicacao"):
    return {
        "id": f"t-{i}",
        "categoria": categoria,
        "prompt": prompt,
        "resposta": resposta,
        "origem": "teste",
        "licenca": "CC0",
    }


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- schema


def test_exemplos_carregam_e_normalizam(tmp_path):
    exemplos = load_instructions(EXEMPLO / "sft.jsonl")
    assert 5 <= len(exemplos) <= 10
    assert {e.categoria.value for e in exemplos} == {
        "explicacao",
        "deteccao_bug",
        "correcao",
        "api",
        "recusa",
    }

    p = _jsonl(tmp_path / "crlf.jsonl", [_ex(1, "  Linha 1   \r\nLinha 2\r\n", "ok  ")])
    (e,) = load_instructions(p)
    assert e.conversa()[0].content == "Linha 1\nLinha 2"
    assert e.prompt is None and len(e.messages) == 2


def test_schema_rejeita_formas_invalidas():
    with pytest.raises(ValueError, match="última mensagem"):
        InstructionExample.model_validate(
            {
                "id": "x",
                "categoria": "api",
                "messages": [{"role": "user", "content": "oi"}],
                "origem": "t",
                "licenca": "t",
            }
        )
    with pytest.raises(ValueError, match="faltou"):
        InstructionExample.model_validate(
            {"id": "x", "categoria": "api", "prompt": "oi", "origem": "t", "licenca": "t"}
        )
    with pytest.raises(ValueError):  # licença é obrigatória
        InstructionExample.model_validate(
            {"id": "x", "categoria": "api", "prompt": "a", "resposta": "b", "origem": "t"}
        )


def test_formato_trl_sft_multiturno():
    ex = next(e for e in load_instructions(EXEMPLO / "sft.jsonl") if e.id == "exemplo-api-01")
    row = to_trl_sft(ex)
    assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in row["prompt"]] == ["system", "user", "assistant", "user"]
    assert [m["role"] for m in row["completion"]] == ["assistant"]  # loss só na última resposta
    assert to_trl_sft(ex, system_prompt=None)["prompt"][0]["role"] == "user"


@pytest.mark.skipif(not (RAIZ.parent / "forja-evals").exists(), reason="forja-evals não está ao lado")
def test_system_prompt_igual_ao_do_forja_evals():
    """Treinar com um prompt de sistema e avaliar com outro mede a diferença de prompt, não o treino."""
    src = (RAIZ.parent / "forja-evals" / "src" / "forja_evals" / "prompts.py").read_text(encoding="utf-8")
    alvo = next(
        n.value.value
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "SYSTEM_PROMPT"
    )
    assert alvo == SYSTEM_PROMPT


# --------------------------------------------------------------------------- dedup


def test_dedup_exata_ignora_caixa_e_espacos():
    a = InstructionExample.model_validate(_ex(1, "O que faz `zip`?", "Pareia iteráveis."))
    b = InstructionExample.model_validate(_ex(2, "o  que faz `ZIP` ?", "pareia   iteráveis."))
    rep = dedup([a, b])
    assert [e.id for e in rep.mantidos] == ["t-1"]
    assert rep.duplicatas_exatas == [("t-2", "t-1")]


def test_near_dup_por_jaccard():
    base = "Explique por que essa função recursiva que calcula a sequência de Fibonacci fica lenta para valores grandes de n e como resolver com memoização"
    a = InstructionExample.model_validate(_ex(1, base, "Recalcula subproblemas; use lru_cache."))
    b = InstructionExample.model_validate(
        _ex(2, base + " por favor", "Recalcula subproblemas; use lru_cache.")
    )
    c = InstructionExample.model_validate(
        _ex(3, "Como abrir um arquivo CSV com o módulo csv?", "Use csv.reader.")
    )
    rep = dedup([a, b, c], n=3, limiar=0.8)
    assert [e.id for e in rep.mantidos] == ["t-1", "t-3"]
    assert rep.quase_duplicatas[0][:2] == ("t-2", "t-1")
    assert rep.quase_duplicatas[0][2] >= 0.8


# --------------------------------------------------------------------------- contaminação


def test_contaminacao_pega_caso_plantado(tmp_path, eval_dir):
    vazado = "Me ajuda com isso, é do trabalho: " + CASOS_EVAL[0]["input"] + "\nObrigado!"
    p = _jsonl(
        tmp_path / "sft.jsonl", [_ex(1, "Como abrir um arquivo CSV?", "Use csv.reader."), _ex(2, vazado)]
    )
    with pytest.raises(ContaminacaoError, match=r"t-2 ~ eval-teste/deteccao_bug-0001 \(input"):
        prepara_instrucoes(p, [eval_dir])

    rel = prepara_instrucoes(p, [eval_dir], falhar=False)  # modo relatório (check-data)
    assert [(a.treino_id, a.eval_id) for a in rel.contaminacao] == [("t-2", "deteccao_bug-0001")]


def test_contaminacao_pega_gabarito_vazado(tmp_path, eval_dir):
    p = _jsonl(
        tmp_path / "sft.jsonl", [_ex(1, "Pergunta diferente sobre ordenação.", CASOS_EVAL[1]["referencia"])]
    )
    with pytest.raises(ContaminacaoError, match="referencia"):
        prepara_instrucoes(p, [eval_dir])


def test_dados_limpos_passam(eval_dir):
    rel = prepara_instrucoes(EXEMPLO / "sft.jsonl", [eval_dir])
    assert rel.contaminacao == [] and len(rel.dedup.mantidos) == rel.total


@pytest.mark.skipif(not SEED_V0.exists(), reason="forja-evals não está ao lado")
def test_exemplos_nao_contaminam_o_forja_seed_v0():
    assert prepara_instrucoes(EXEMPLO / "sft.jsonl", [SEED_V0]).contaminacao == []
    assert len(prepara_preferencias(EXEMPLO / "preferencias", [SEED_V0]).pares) == 3


# --------------------------------------------------------------------------- preferências (DPO)


def _prefs_dir(tmp_path, manifest: str):
    d = tmp_path / "prefs"
    d.mkdir(parents=True)
    (d / "dataset.yaml").write_text(manifest, encoding="utf-8")
    par = PreferencePair(prompt="Tem bug?", escolhida="Sim, IndexError.", rejeitada="Não.", origem="humano")
    (d / "pairs.jsonl").write_text(par.model_dump_json() + "\n", encoding="utf-8")
    return d


def test_dpo_recusa_pares_de_dataset_de_avaliacao(tmp_path, eval_dir):
    d = _prefs_dir(tmp_path, f"nome: p\nuso: treino\ngerado_de: ['{eval_dir.as_posix()}']\n")
    with pytest.raises(DatasetDeAvaliacaoError, match="uso: avaliacao"):
        load_preferences(d)


def test_dpo_recusa_manifest_de_avaliacao_ou_sem_proveniencia(tmp_path):
    with pytest.raises(DatasetDeAvaliacaoError, match="uso='avaliacao'"):
        load_preferences(_prefs_dir(tmp_path / "a", "nome: p\nuso: avaliacao\n"))
    with pytest.raises(DatasetDeAvaliacaoError, match="gerado_de"):
        load_preferences(_prefs_dir(tmp_path / "b", "nome: p\nuso: treino\n"))
    with pytest.raises(FileNotFoundError):
        load_preferences(tmp_path)  # sem dataset.yaml


def test_dpo_aceita_dataset_de_treino(tmp_path):
    origem = escreve_eval(tmp_path / "treino-v1", uso="treino")
    d = _prefs_dir(tmp_path, "nome: p\nuso: treino\ngerado_de: ['../treino-v1']\n")  # relativo ao diretório
    prefs = load_preferences(d)
    assert len(prefs.pares) == 1 and origem.exists()

    row = to_trl_dpo(prefs.pares[0])
    t = prefs.pares[0].to_trl()
    assert row["prompt"][-1] == {"role": "user", "content": t["prompt"]}
    assert row["chosen"] == [{"role": "assistant", "content": t["chosen"]}]
    assert row["rejected"] == [{"role": "assistant", "content": t["rejected"]}]


def test_dpo_checa_contaminacao_dos_pares(tmp_path, eval_dir):
    d = tmp_path / "prefs"
    d.mkdir()
    (d / "dataset.yaml").write_text("nome: p\nuso: treino\ngerado_de: [escrito-a-mao]\n", encoding="utf-8")
    par = PreferencePair(
        prompt=CASOS_EVAL[1]["input"], escolhida="Não existe.", rejeitada="Use turbo=True.", origem="humano"
    )
    (d / "pairs.jsonl").write_text(par.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ContaminacaoError):
        prepara_preferencias(d, [eval_dir])
