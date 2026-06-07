# gemma4-12b-vllm-sm120

**Serve an abliterated Gemma‑4‑12B at 50–118 tok/s single‑stream on consumer‑class
Blackwell GPUs (SM120) that have _no NVLink / no GPU P2P_ — with vLLM + NVIDIA
ModelOpt FP8 + a 0.4 B MTP speculative drafter.**

A complete, reproducible recipe by [Lna‑Lab](https://github.com/lna-lab). Everything
here was measured end‑to‑end on **7× RTX PRO 2000 Blackwell (16 GB each, SM120,
PCIe, no NVLink)**. It is not a model release — it wires together other people's
excellent models (AEON‑7's abliteration + quant, Google's MTP drafter) and
documents the runtime tricks that make them actually fly on this hardware.

> **The headline finding:** Gemma‑4‑12B is the new `gemma4_unified` architecture.
> It needs **vLLM nightly** (released ≤0.22.1 can't load it). On a no‑P2P box, any
> tensor‑parallel run **hangs** unless you disable both NCCL P2P *and* vLLM's custom
> all‑reduce. Once you do, the single best interactive config is the **Mixed
> NVFP4+FP8** variant on **one GPU + MTP spec‑decode → ~53 tok/s**, or **TP=4 →
> 118 tok/s** if you want to spend 4 GPUs.

---

## Results (this hardware)

Model: `AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-*` · vLLM `nightly`
(`0.22.1rc1.dev`, native `Gemma4UnifiedForConditionalGeneration`, transformers 5.10.2)
· RTX PRO 2000 Blackwell (16 GB, **288 GB/s**, SM120, no NVLink).

### Single‑stream decode (1 request × 512 tok, interactive latency)

| Config | GPUs | tok/s | Notes |
|---|---|---|---|
| BF16 (reference) | — | 7.7 | from AEON card (GB10) |
| Mixed NVFP4+FP8, TP=1, no spec | 1 | 26.6 | memory‑BW bound (288 ÷ 9.3 GB) |
| **Mixed NVFP4+FP8, TP=1 + MTP spec (k=4)** | **1** | **53.5** | 2.0× from spec; frees 6 GPUs |
| **Mixed NVFP4+FP8, TP=4 + MTP spec (k=3)** | **4** | **118.2** | ~15× BF16; peak here |

**`num_speculative_tokens` (TP=4) sweep:** k3 **118.2** · k4 114.6 · k5 116.0 · k6 109.1 · k8 84.5
(too many spec tokens collapses the MTP acceptance rate).

### Aggregate throughput (64 concurrent × 256 tok, `-c 65536`)

| Config | GPUs | aggregate tok/s |
|---|---|---|
| FP8, **TP=2** | 2 | **1418** |
| FP8, TP=4 | 4 | 1242 |

> **The inversion that surprises everyone:** for *aggregate* batched throughput
> **TP=2 beats TP=4** (host‑memory all‑reduce overhead grows faster than the compute
> win), but for *single‑stream* decode **TP=4 beats TP=2** (decode is memory‑bandwidth
> bound, so sharding the weights buys you bandwidth). **Pick TP by workload, not VRAM.**

### Max context (single GPU)

13 GB FP8 won't leave KV room on one 16 GB card. The **9.3 GB Mixed** variant does:
with `--max-num-seqs 4 --gpu-memory-utilization 0.95` it serves **`-c 32768`** on one
GPU at ~50 tok/s (gemma‑4's sliding‑window attention keeps the KV read small).

---

## Quickstart

Requires: a Blackwell GPU (SM120 / RTX 50‑series / GB10 / B100/B200), Docker with the
NVIDIA runtime, `hf` CLI.

```bash
git clone https://github.com/lna-lab/gemma4-12b-vllm-sm120
cd gemma4-12b-vllm-sm120

# 1) Download the serving model (9.3 GB) + the MTP drafter (0.84 GB)
mkdir -p models
hf download AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8 \
    --local-dir models/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8
hf download google/gemma-4-12B-it-assistant \
    --local-dir models/gemma-4-12B-it-assistant

# 2) Serve (OpenAI API on :8000). solo = 1 GPU ~50 t/s; fast = 4 GPU ~118 t/s
MODELS=./models MODE=solo ./serve/serve_gemma4-12b-aeon.sh
#                MODE=fast GPUS=0,1,4,5  ./serve/serve_gemma4-12b-aeon.sh

# 3) Test
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"model":"gemma-4-12b-aeon","messages":[{"role":"user","content":"Explain the CAP theorem in one sentence."}]}'
```

The launcher auto‑detects ModelOpt (`--quantization modelopt`), pins KV cache to
`fp8` (required — nvfp4 KV collapses draft acceptance), and in `fast` mode adds the
no‑P2P flags below.

---

## The three things that actually matter

### 1. Use vLLM **nightly** — `gemma4_unified` is too new for releases
`google/gemma-4-12B-it` is `model_type: gemma4_unified` (text+vision+audio, dual
attention `head_dim` 256 sliding / 512 global). Only `vllm/vllm-openai:nightly`
registers `Gemma4UnifiedForConditionalGeneration`; released vLLM ≤0.22.1 falls back to
the transformers backend and crashes in the global‑attention `o_proj`. See
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### 2. No NVLink/P2P → disable BOTH NCCL P2P and vLLM custom all‑reduce
On a PCIe box with no working GPU P2P, **any TP>1 vLLM run hangs**: first at NCCL init,
then (after you fix that) the *forward pass* deadlocks. You need **both**:
```
NCCL_P2P_DISABLE=1            # env — fixes the NCCL-init hang
--disable-custom-all-reduce   # flag — fixes the forward-pass deadlock (vLLM's custom
                              #        all-reduce kernel also assumes P2P)
```
plus Docker `--ipc=host --shm-size 16gb`. Full write‑up:
[docs/NO-P2P-MULTIGPU.md](docs/NO-P2P-MULTIGPU.md).

### 3. Speculative decode with Google's native MTP drafter (lossless, ~2×)
`google/gemma-4-12B-it-assistant` (0.4 B, `gemma4_unified_assistant`) is the
first‑party MTP drafter. vLLM nightly drives it via
`--speculative-config '{"method":"mtp","model":<drafter>,"num_speculative_tokens":3}'`.
It is **lossless** (the target verifies every token), so reasoning quality is identical
— we confirmed it on the mirror‑axis, clock‑angle, and 0.999…=1 probes
([bench/quality_test.py](bench/quality_test.py)). Peak at **k=3** here.

---

## Which AEON variant?

| Variant | quant | size | best at | repo |
|---|---|---|---|---|
| FP8 | FP8 W8A8 | 13 GB | aggregate (TP=2), max quality (MMLU 80.4) | `…-K4-FP8` |
| NVFP4 (MLP‑only) | NVFP4 MLP + BF16 attn | 11.7 GB | (superseded) | `…-K4-NVFP4` |
| **Mixed** | NVFP4 MLP + FP8 attn | **9.3 GB** | **single‑stream, fits 1 GPU** | `…-K4-NVFP4-FP8` |

This repo defaults to **Mixed** — smallest footprint = fastest decode, and 9.3 GB fits
one 16 GB GPU so you dodge the no‑P2P penalty entirely. Note: "attention as NVFP4 hurts
reasoning" — the working 4‑bit variants keep attention at BF16 or FP8, never NVFP4.

---

## Repo layout

```
serve/serve_gemma4-12b-aeon.sh   # OpenAI-API launcher, solo|fast modes
bench/bench_tps.py               # aggregate + single-stream + spec-decode aware
bench/diag_generate.py           # coherence / weight-loading diagnosis
bench/quality_test.py            # reasoning-quality probe (insight/precision/rigor)
quantize/                        # bonus: quantize *any* gemma4_unified to NVFP4 yourself
docs/NO-P2P-MULTIGPU.md          # the TP>1 hang fix
docs/BENCHMARKS.md               # full numbers + how they were taken
docs/QUANTIZE.md                 # gemma4_unified NVFP4 lessons (W4A4 collapse, ct pin)
docs/TROUBLESHOOTING.md          # every wall we hit and the fix
```

## ⚠️ Known‑bad base model

**Do not use `huihui-ai/Huihui-gemma-4-12B-it-abliterated`.** Its abliterated base is
functionally broken — it emits `<pad>` under greedy decoding and word‑salad when
sampled, in **bf16 Transformers as well as every quantization**. The weights are
numerically clean and its config/tokenizer match Google's, so the abliteration itself
damaged the model. AEON‑7's K=4 biprojection abliteration is the working one we use here.

## Credits

- **Abliteration + quant:** [AEON‑7](https://huggingface.co/AEON-7) (K=4 biprojection, ModelOpt FP8/NVFP4)
- **Base + MTP drafter:** [Google DeepMind](https://huggingface.co/google/gemma-4-12B-it) (Gemma 4)
- **Runtime:** [vLLM](https://github.com/vllm-project/vllm) · **Quant tooling:** [llm‑compressor](https://github.com/vllm-project/llm-compressor) / [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- Packaging, benchmarks, and the no‑P2P/spec‑decode recipe: **Lna‑Lab**

## License

Apache‑2.0 for the scripts/docs in this repo. The models retain their own licenses
(Gemma terms for the Gemma‑4 derivatives; check each model card).
