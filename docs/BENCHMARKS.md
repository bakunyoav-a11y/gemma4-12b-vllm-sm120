# Benchmarks

Hardware: **RTX PRO 2000 Blackwell**, 16 GB GDDR7, **288 GB/s**, SM120, PCIe (no NVLink).
Runtime: `vllm/vllm-openai:nightly` (`0.22.1rc1.dev231`, transformers 5.10.2, torch
2.11+cu130). Harness: [`bench/bench_tps.py`](../bench/bench_tps.py). Date: 2026‑06‑07.

All runs use `ignore_eos` so each request emits exactly the target token count
(clean decode measurement). Models are the `AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-*`
variants; spec‑decode drafter is `google/gemma-4-12B-it-assistant`.

## Single‑stream decode — `BENCH_NREQ=1 BENCH_OUTTOK=512`

| Model | quant | TP | spec (k) | tok/s |
|---|---|---|---|---|
| FP8 | FP8 W8A8, 13 GB | 1 | — | (OOM/KV‑starved on 16 GB) |
| FP8 | FP8 W8A8, 13 GB | 2 | — | 36.1 |
| FP8 | FP8 W8A8, 13 GB | 4 | — | 54.1 |
| Mixed | NVFP4 MLP + FP8 attn, 9.3 GB | 1 | — | 26.6 |
| Mixed | 9.3 GB | 1 | 4 | **53.5** |
| Mixed | 9.3 GB | 4 | 3 | **118.2** |

**TP=4 + spec `num_speculative_tokens` sweep:** k3 **118.2** · k4 114.6 · k5 116.0 ·
k6 109.1 · k8 84.5.

Theory check: single‑GPU decode ≈ `288 GB/s × ~0.85 / footprint`. Mixed 9.3 GB → ~26 t/s
(matches 26.6). Spec‑decode multiplies by the accept rate (~2.0× here). TP sharding
multiplies usable bandwidth but pays the host‑memory all‑reduce tax
([NO‑P2P‑MULTIGPU](NO-P2P-MULTIGPU.md)).

## Aggregate throughput — `BENCH_NREQ=64 BENCH_OUTTOK=256 -c 65536`

| Model | TP | aggregate tok/s | per‑req | concurrency |
|---|---|---|---|---|
| FP8 | 2 | **1418.4** | 22.2 | 2.32× |
| FP8 | 4 | 1242.4 | 19.4 | 4.95× |
| FP8 | 1 | — | — | can't fit 65536 KV on 16 GB |

## Max context, single GPU (Mixed)
`--max-num-seqs 4 --gpu-memory-utilization 0.95` → KV budget **37,465 tokens**, serves
**`-c 32768`** at ~50 t/s single‑stream. (gemma‑4 sliding‑window attention keeps KV
near context‑independent.)

## Reproduce
```bash
# single-stream, Mixed TP=1 + spec k=3
docker run --rm --gpus '"device=0"' --ipc=host --shm-size 16gb --entrypoint python3 \
  -e BENCH_TP=1 -e BENCH_NREQ=1 -e BENCH_OUTTOK=512 -e BENCH_MAXLEN=4096 -e BENCH_GPU_UTIL=0.93 \
  -e BENCH_MODEL=/m -e BENCH_SPEC_MODEL=/draft -e BENCH_SPEC_NUM=3 -e BENCH_KV_DTYPE=fp8 \
  -e NCCL_P2P_DISABLE=1 \
  -v $PWD/models/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8:/m:ro \
  -v $PWD/models/gemma-4-12B-it-assistant:/draft:ro \
  -v $PWD/bench/bench_tps.py:/work/bench_tps.py:ro \
  vllm/vllm-openai:nightly /work/bench_tps.py
```
`bench_tps.py` env knobs: `BENCH_TP`, `BENCH_NREQ`, `BENCH_OUTTOK`, `BENCH_MAXLEN`,
`BENCH_GPU_UTIL`, `BENCH_MAX_NUM_SEQS`, `BENCH_KV_DTYPE`, `BENCH_SPEC_MODEL`, `BENCH_SPEC_NUM`.
