# Troubleshooting — every wall we hit, and the fix

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: ... model type 'gemma4_unified' ... not recognize this architecture` | transformers too old | transformers ≥ 5.10 (vLLM nightly has it) |
| vLLM loads via "Transformers modeling backend", crashes in `o_proj` (`flashinfer ... assert a.shape[1]==b.shape[1]` / `cutlass ... shapes cannot be multiplied (…x2048 and 3840x4096)`) | released vLLM has no native `Gemma4Unified`; the fallback mishandles the **dual head_dim** (256 sliding / 512 global) | use **`vllm/vllm-openai:nightly`** (registers `Gemma4UnifiedForConditionalGeneration`, forces TRITON_ATTN for the heterogeneous heads) |
| TP>1 hangs at `pynccl ... nccl==2.28.9`, GPUs 100 % / weights never load | no GPU P2P (PCIe, no NVLink) | `NCCL_P2P_DISABLE=1` |
| TP>1 hangs in forward: `shm_broadcast: No available shared memory broadcast block found in 60 seconds` | vLLM custom all‑reduce also needs P2P | `--disable-custom-all-reduce` (+ keep `NCCL_P2P_DISABLE=1`) |
| `RuntimeError: ... start a new process before ... bootstrapping` (offline script, TP>1) | TP spawn re‑imports your module | guard the script with `if __name__ == "__main__":` |
| `no module or parameter named 'vision_embedder.patch_dense.weight'` | `config.json` `quantization_config.ignore` doesn't match the checkpoint's BF16 modules | patch ignore to cover all BF16 Linears, or re‑bake with `ignore:[lm_head,'re:.*embedding_projection.*']` |
| Output is all `<pad>` (token 0) under greedy; word‑salad when sampled | broken **base model** (e.g. huihui abliterated) *or* a W4A4 NVFP4 activation collapse | use a known‑good base (AEON K=4); for self‑quant use W4A16/FP8, not W4A4 |
| garbage output, model loads fine, ct version `0.17.1a` | compressed‑tensors alpha pack format vLLM mis‑decodes | bake with `compressed-tensors==0.17.0` (match the runtime) |
| `KV cache is needed ... larger than available ... estimated maximum model length is N` | 13 GB weights leave no KV on 16 GB | use the 9.3 GB Mixed variant; `--max-num-seqs 4 --gpu-memory-utilization 0.95`; or add a GPU (TP=2) |
| `Free memory on device cuda:N ... less than desired GPU memory utilization` | another process resident on that GPU | lower `--gpu-memory-utilization` to 0.85 or pick clean GPUs |
| spec‑decode drops acceptance / slows down | `num_speculative_tokens` too high (MTP layer reused too many times) | k=3–5 is the sweet spot here (k=3 peak); never pair spec‑decode with **nvfp4** KV — use `fp8` KV |

## Diagnosis tools
- `bench/diag_generate.py` — forces tokens (`ignore_eos`, greedy), prints raw token ids
  and the chat‑template prompt. All‑`0` ids = `<pad>` collapse; varied‑but‑incoherent =
  garbage; coherent = good. Set `DIAG_TP`, `DIAG_GPU_UTIL`, `VLLM_MODEL_IMPL`.
- `bench/quality_test.py` — three reasoning probes (mirror axis, clock angle, 0.999…=1)
  that pattern‑matching fails, to confirm a quant didn't lobotomize the model.
