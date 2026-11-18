# Dados de exemplo

**Só exemplos.** Escritos à mão para o pipeline rodar de ponta a ponta (check-data, smoke de SFT e de DPO).
Nove exemplos e três pares não ensinam nada a um modelo de 1,5B; o dataset real é montado à parte.

- `sft.jsonl`: `InstructionExample` (ver `src/forja_posttrain/data.py`). Aceita `messages` (multi-turno) ou o
  atalho `prompt` + `resposta`, sempre com `categoria`, `origem` e `licenca`.
- `preferencias/`: `pairs.jsonl` de `PreferencePair` (forja-contracts) + `dataset.yaml` com `uso: treino` e
  `gerado_de` (proveniência).

Nenhum caso do `forja-evals` foi copiado para cá; o `forja-posttrain check-data` confere isso a cada execução.
