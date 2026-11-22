"""Pipeline de dados do pós-treino.

Instrução (SFT): JSONL de `InstructionExample` -> normalização -> dedup exata (hash) -> near-dup
(Jaccard de n-gramas) -> checagem de contaminação contra os datasets do forja-evals -> formato
prompt/completion conversacional do TRL.

Preferência (DPO): `PreferencePair` (forja-contracts) + `dataset.yaml` de proveniência. Pares
gerados a partir de um dataset com `uso: avaliacao` são recusados (regra anti-circularidade).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from forja_contracts import Categoria, EvalCase, PreferencePair, read_jsonl
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Mantido idêntico a `forja_evals.prompts.SYSTEM_PROMPT`: treinar com o mesmo prompt de sistema que a
# régua usa na inferência. `tests/test_data.py` confere a igualdade quando o forja-evals está ao lado.
# Pendência: subir esse texto para o forja-contracts para não haver cópia.
SYSTEM_PROMPT = """Você é um engenheiro de software sênior que revisa código.
Responda sempre em português do Brasil, de forma direta.
- Ao explicar código, diga o que ele faz e por quê, sem repetir o código linha a linha.
- Ao apontar um bug, diga onde está, por que é um bug e como corrigir.
- Ao corrigir código, devolva a função corrigida completa num bloco ```python.
- Se a pergunta não puder ser respondida com o contexto dado, ou se algo não existir, diga isso claramente em vez de inventar."""


# --------------------------------------------------------------------------- schema


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class InstructionExample(BaseModel):
    """Um exemplo de instrução. Aceita `messages` (multi-turno) ou o atalho `prompt` + `resposta`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    categoria: Categoria
    messages: list[Message] = Field(default_factory=list)
    prompt: str | None = None
    resposta: str | None = None
    origem: str = Field(min_length=1)  # ex.: "escrito-a-mao", "destilado:<modelo>", "<dataset>@<commit>"
    licenca: str = Field(min_length=1)  # ex.: "CC-BY-4.0", "Apache-2.0"

    @model_validator(mode="after")
    def _forma(self) -> InstructionExample:
        if self.messages and (self.prompt or self.resposta):
            raise ValueError(f"{self.id}: use `messages` ou `prompt`+`resposta`, não os dois")
        if not self.messages and not (self.prompt and self.resposta):
            raise ValueError(f"{self.id}: faltou `messages` ou `prompt`+`resposta`")
        msgs = self.conversa()
        if msgs[-1].role != "assistant":
            raise ValueError(f"{self.id}: a última mensagem precisa ser do assistant")
        if not any(m.role == "user" for m in msgs):
            raise ValueError(f"{self.id}: precisa de pelo menos uma mensagem do user")
        return self

    def conversa(self) -> list[Message]:
        if self.messages:
            return list(self.messages)
        return [
            Message(role="user", content=self.prompt or ""),
            Message(role="assistant", content=self.resposta or ""),
        ]

    def texto_entrada(self) -> str:
        """O que o usuário manda (todas as falas do user). É isso que se compara com `EvalCase.input`."""
        return "\n".join(m.content for m in self.conversa() if m.role == "user")

    def texto_resposta(self) -> str:
        return "\n".join(m.content for m in self.conversa() if m.role == "assistant")


# --------------------------------------------------------------------------- normalização


def normaliza_texto(s: str) -> str:
    """NFC, fim de linha LF, sem espaço no fim das linhas e nas pontas. Indentação de código é preservada."""
    s = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(linha.rstrip() for linha in s.split("\n")).strip()


def normaliza(ex: InstructionExample) -> InstructionExample:
    msgs = [Message(role=m.role, content=normaliza_texto(m.content)) for m in ex.conversa()]
    return ex.model_copy(update={"messages": msgs, "prompt": None, "resposta": None})


def load_instructions(path: str | Path) -> list[InstructionExample]:
    exemplos = [normaliza(e) for e in read_jsonl(path, InstructionExample)]
    ids = [e.id for e in exemplos]
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if repetidos:
        raise ValueError(f"ids repetidos em {path}: {repetidos}")
    return exemplos


# --------------------------------------------------------------------------- n-gramas

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokens(s: str) -> list[str]:
    """Tokenização simples por palavra/pontuação, minúscula. Serve tanto para prosa quanto para código."""
    return _TOKEN.findall(normaliza_texto(s).lower())


def ngramas(s: str, n: int) -> set[tuple[str, ...]]:
    toks = tokens(s)
    if not toks:
        return set()
    if len(toks) < n:  # texto curto: o texto inteiro vira um único "n-grama"
        return {tuple(toks)}
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contencao(a: set, b: set) -> float:
    """Fração de `b` contida em `a`. Assimétrica de propósito: um caso de eval colado dentro de um exemplo
    de treino maior tem Jaccard baixo, mas contenção 1."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(b)


# --------------------------------------------------------------------------- dedup


def hash_exato(ex: InstructionExample) -> str:
    canon = json.dumps(
        [(m.role, " ".join(tokens(m.content))) for m in ex.conversa()],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


@dataclass
class DedupReport:
    mantidos: list[InstructionExample]
    duplicatas_exatas: list[tuple[str, str]] = field(default_factory=list)  # (removido, original)
    # (removido, original, jaccard)
    quase_duplicatas: list[tuple[str, str, float]] = field(default_factory=list)


def dedup(exemplos: Iterable[InstructionExample], *, n: int = 5, limiar: float = 0.8) -> DedupReport:
    """Dedup exata por hash e near-dup por Jaccard de n-gramas de palavra.

    O(N²) na near-dup: ok para os milhares de exemplos da v1. Na v2 entra MinHash + LSH (ex.: `datasketch`),
    que estima o mesmo Jaccard em O(N).

    Pares contrastivos não contam como near-dup: o mesmo arquivo antes e depois da correção (`origem` com a
    mesma base antes do `#`, ao menos um dos dois com sufixo, ex.: `github:repo@sha:arquivo` e
    `github:repo@sha:arquivo#corrigido`) é quase igual por construção e de propósito. Duplicata exata
    continua saindo.
    """
    rep = DedupReport(mantidos=[])
    vistos: dict[str, str] = {}
    shingles: list[tuple[str, set, str, bool]] = []
    for ex in exemplos:
        h = hash_exato(ex)
        if h in vistos:
            rep.duplicatas_exatas.append((ex.id, vistos[h]))
            continue
        vistos[h] = ex.id
        base, sufixo = ex.origem.split("#", 1)[0], "#" in ex.origem
        sh = ngramas(" ".join(m.content for m in ex.conversa()), n)
        parecido = max(
            ((jaccard(sh, s), eid) for eid, s, b, suf in shingles if not (b == base and (suf or sufixo))),
            default=(0.0, ""),
        )
        if parecido[0] >= limiar:
            rep.quase_duplicatas.append((ex.id, parecido[1], round(parecido[0], 3)))
            continue
        shingles.append((ex.id, sh, base, sufixo))
        rep.mantidos.append(ex)
    return rep


# --------------------------------------------------------------------------- contaminação


class ContaminacaoError(RuntimeError):
    """Dado de treino se sobrepõe ao conjunto de avaliação. Nunca é aviso: treinar assim invalida a régua."""


@dataclass
class EvalDataset:
    nome: str
    uso: str
    casos: list[EvalCase]
    caminho: Path


def _read_manifest(dirpath: Path) -> dict[str, Any]:
    manifest = dirpath / "dataset.yaml"
    if not manifest.exists():
        raise FileNotFoundError(f"{dirpath} não tem dataset.yaml (proveniência é obrigatória)")
    return yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}


def load_eval_dataset(dirpath: str | Path) -> EvalDataset:
    """Lê um dataset no formato do forja-evals: `dataset.yaml` + `cases.jsonl` de `EvalCase`."""
    d = Path(dirpath)
    meta = _read_manifest(d)
    return EvalDataset(
        nome=meta.get("nome", d.name),
        uso=meta.get("uso", "avaliacao"),
        casos=read_jsonl(d / "cases.jsonl", EvalCase),
        caminho=d,
    )


@dataclass(frozen=True)
class Sobreposicao:
    treino_id: str
    eval_dataset: str
    eval_id: str
    campo: Literal["input", "referencia"]
    score: float


def checa_contaminacao(
    itens: Iterable[tuple[str, str, str]],
    eval_dirs: Iterable[str | Path],
    *,
    n: int = 5,
    limiar: float = 0.5,
) -> list[Sobreposicao]:
    """Compara cada item de treino com cada caso de avaliação.

    `itens` são tuplas (id, texto_entrada, texto_resposta). Duas checagens por par:
    - entrada do treino vs. `EvalCase.input` (pergunta vazada);
    - resposta do treino vs. `EvalCase.referencia` (gabarito vazado).
    O score é a contenção: fração dos n-gramas do caso de eval presentes no texto de treino.
    """
    casos: list[tuple[str, EvalCase, set, set]] = []
    for d in eval_dirs:
        ds = load_eval_dataset(d)
        for c in ds.casos:
            casos.append((ds.nome, c, ngramas(c.input, n), ngramas(c.referencia or "", n)))

    achados = []
    for tid, entrada, resposta in itens:
        ng_in, ng_out = ngramas(entrada, n), ngramas(resposta, n)
        for nome, caso, ng_caso_in, ng_caso_ref in casos:
            s_in = contencao(ng_in, ng_caso_in)
            if s_in >= limiar:
                achados.append(Sobreposicao(tid, nome, caso.id, "input", round(s_in, 3)))
            s_ref = contencao(ng_out, ng_caso_ref)
            if s_ref >= limiar:
                achados.append(Sobreposicao(tid, nome, caso.id, "referencia", round(s_ref, 3)))
    return achados


def exige_sem_contaminacao(achados: list[Sobreposicao]) -> None:
    if achados:
        linhas = "\n".join(
            f"  - {a.treino_id} ~ {a.eval_dataset}/{a.eval_id} ({a.campo}, contenção {a.score})"
            for a in achados
        )
        raise ContaminacaoError(
            f"{len(achados)} sobreposição(ões) entre treino e avaliação. Remova esses exemplos do treino:\n{linhas}"
        )


@dataclass
class RelatorioDados:
    total: int
    dedup: DedupReport
    contaminacao: list[Sobreposicao]

    def resumo(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "mantidos": len(self.dedup.mantidos),
            "duplicatas_exatas": self.dedup.duplicatas_exatas,
            "quase_duplicatas": self.dedup.quase_duplicatas,
            "contaminacao": [a.__dict__ for a in self.contaminacao],
            "por_categoria": _conta_categorias(self.dedup.mantidos),
        }


def _conta_categorias(exemplos: list[InstructionExample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in exemplos:
        out[e.categoria.value] = out.get(e.categoria.value, 0) + 1
    return dict(sorted(out.items()))


def prepara_instrucoes(
    path: str | Path,
    eval_dirs: Iterable[str | Path],
    *,
    dedup_n: int = 5,
    dedup_limiar: float = 0.8,
    contam_n: int = 5,
    contam_limiar: float = 0.5,
    falhar: bool = True,
) -> RelatorioDados:
    """Carrega, normaliza, deduplica e checa contaminação. Com `falhar=True`, contaminação levanta erro."""
    exemplos = load_instructions(path)
    rep = dedup(exemplos, n=dedup_n, limiar=dedup_limiar)
    achados = checa_contaminacao(
        ((e.id, e.texto_entrada(), e.texto_resposta()) for e in rep.mantidos),
        eval_dirs,
        n=contam_n,
        limiar=contam_limiar,
    )
    if falhar:
        exige_sem_contaminacao(achados)
    return RelatorioDados(total=len(exemplos), dedup=rep, contaminacao=achados)


# --------------------------------------------------------------------------- formatos do TRL


def to_trl_sft(ex: InstructionExample, system_prompt: str | None = SYSTEM_PROMPT) -> dict[str, list[dict]]:
    """Formato prompt/completion conversacional: a loss cai só na última resposta do assistant
    (`completion_only_loss` é o padrão do SFTTrainer para esse formato; não depende de `{% generation %}`
    no chat template)."""
    msgs = [m.model_dump() for m in ex.conversa()]
    if system_prompt and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": system_prompt}, *msgs]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


class DatasetDeAvaliacaoError(ValueError):
    """Pares de preferência vindos de um dataset de avaliação (circularidade)."""


@dataclass
class Preferencias:
    pares: list[PreferencePair]
    manifest: dict[str, Any]


def load_preferences(dirpath: str | Path) -> Preferencias:
    """Lê `pairs.jsonl` + `dataset.yaml` de um diretório de preferências.

    O `dataset.yaml` declara `uso` e `gerado_de` (diretórios dos datasets de origem, relativos ao próprio
    diretório ou absolutos). Recusa se o próprio diretório ou qualquer origem tiver `uso: avaliacao`, ou se
    a proveniência estiver faltando.
    """
    d = Path(dirpath)
    meta = _read_manifest(d)
    if meta.get("uso") != "treino":
        raise DatasetDeAvaliacaoError(f"{d}: uso={meta.get('uso')!r}; pares para DPO precisam de uso: treino")
    origens = meta.get("gerado_de")
    if not origens:
        raise DatasetDeAvaliacaoError(f"{d}: dataset.yaml sem `gerado_de`; sem proveniência, sem DPO")
    for origem in origens:
        p = Path(origem)
        p = p if p.is_absolute() else (d / p)
        if str(origem).startswith(("escrito-a-mao", "humano:")):
            continue  # pares escritos à mão não têm dataset de origem
        uso = _read_manifest(p).get("uso", "avaliacao")
        if uso == "avaliacao":
            raise DatasetDeAvaliacaoError(
                f"{d}: pares gerados a partir de {p} (uso: avaliacao). O conjunto de avaliação nunca vira "
                "par de preferência."
            )
    return Preferencias(pares=read_jsonl(d / "pairs.jsonl", PreferencePair), manifest=meta)


def to_trl_dpo(par: PreferencePair, system_prompt: str | None = SYSTEM_PROMPT) -> dict[str, list[dict]]:
    """`PreferencePair.to_trl()` (strings) -> formato conversacional, para o DPOTrainer aplicar o chat template."""
    t = par.to_trl()
    prompt = [{"role": "user", "content": normaliza_texto(t["prompt"])}]
    if system_prompt:
        prompt = [{"role": "system", "content": system_prompt}, *prompt]
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": normaliza_texto(t["chosen"])}],
        "rejected": [{"role": "assistant", "content": normaliza_texto(t["rejected"])}],
    }


def prepara_preferencias(
    dirpath: str | Path, eval_dirs: Iterable[str | Path], *, n: int = 5, limiar: float = 0.5
) -> Preferencias:
    prefs = load_preferences(dirpath)
    achados = checa_contaminacao(
        ((f"par-{i:04d}", p.prompt, p.escolhida) for i, p in enumerate(prefs.pares)),
        eval_dirs,
        n=n,
        limiar=limiar,
    )
    exige_sem_contaminacao(achados)
    return prefs
