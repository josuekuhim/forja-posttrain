"""Modelo e tokenizer minúsculos, construídos localmente, para smoke test e testes.

Nada aqui baixa arquivos: o tokenizer BPE byte-level é treinado em memória (biblioteca `tokenizers`) e o
modelo é um `Qwen2ForCausalLM` com pesos aleatórios. A arquitetura é a mesma do modelo base (Qwen2), então
os nomes de módulo (`q_proj`, `gate_proj`, ...) que o LoRA mira são os mesmos de produção.
"""

from __future__ import annotations

from collections.abc import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

# ChatML, o mesmo formato do Qwen2.5 (sem o prompt de sistema padrão e sem ferramentas).
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
ESPECIAIS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

_CORPUS_PADRAO = [
    "Você é um engenheiro de software sênior que revisa código.",
    "def soma(a, b):\n    return a + b",
    "O código percorre a lista e devolve o maior elemento.",
    "Não sei: essa função não aparece no contexto enviado.",
]


def build_tiny_tokenizer(corpus: Iterable[str] = (), vocab_size: int = 2048) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=ESPECIAIS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator([*_CORPUS_PADRAO, *corpus], trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        chat_template=CHAT_TEMPLATE,
    )


def build_tiny_model(
    tokenizer: PreTrainedTokenizerFast, *, hidden_size: int = 32, n_layers: int = 2, seed: int = 0
):
    import torch

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA, como no Qwen2.5
        max_position_embeddings=512,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,
    )
    return Qwen2ForCausalLM(cfg)
