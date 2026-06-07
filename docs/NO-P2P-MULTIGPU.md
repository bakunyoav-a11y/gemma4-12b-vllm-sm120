# Multi‑GPU vLLM on a no‑NVLink / no‑P2P Blackwell box

Consumer / entry‑pro Blackwell cards (e.g. **RTX PRO 2000**, RTX 50‑series) on plain
PCIe usually have **no working GPU peer‑to‑peer (P2P)** and no NVLink. vLLM tensor
parallelism assumes fast P2P, so out of the box **every TP>1 run hangs**. Here is the
exact failure ladder and the fix.

## Symptom 1 — hang at NCCL init
Stuck right after:
```
INFO [pynccl.py] vLLM is using nccl==2.28.9
```
GPUs pinned at ~369 MiB / 100 % util, weights never load. NCCL spins forever trying
P2P that doesn't exist.

**Fix:** `NCCL_P2P_DISABLE=1` (env). NCCL falls back to shared‑memory / host staging.

## Symptom 2 — hang in the forward pass
After fixing #1, weights load and KV cache is sized, then it deadlocks during the first
forward with, every 60 s:
```
INFO [shm_broadcast.py] No available shared memory broadcast block found in 60 seconds...
```
GPUs at 100 %, no tokens emitted. Cause: vLLM's **custom all‑reduce** CUDA kernel also
assumes P2P.

**Fix:** `--disable-custom-all-reduce` (serve flag) or `disable_custom_all_reduce=True`
(offline `LLM(...)`). All‑reduce goes through NCCL's host path instead.

## The working invocation

```bash
docker run --rm --gpus '"device=0,1,4,5"' --ipc=host --shm-size 16gb \
  -e NCCL_P2P_DISABLE=1 \
  vllm/vllm-openai:nightly \
  --model /model --tensor-parallel-size 4 --disable-custom-all-reduce ...
```

`--ipc=host --shm-size 16gb` are also required (NCCL/host‑path comm needs the shared
memory).

## Performance consequence — TP=2 can beat TP=4

Because all‑reduce now crosses **host memory** (slow, latency‑bound), comm cost grows
with participant count. Measured on this box:

| workload | TP=2 | TP=4 | winner |
|---|---|---|---|
| aggregate (64 conc × 256) | **1418** | 1242 | TP=2 — comm overhead dominates batched compute |
| single‑stream (1 × 512) | 36.1 | **54.1** | TP=4 — decode is BW‑bound, sharding adds bandwidth |

**Rule of thumb on a no‑P2P box:** more GPUs help *latency* (single‑stream, BW‑bound)
but hurt *batched throughput*. The only way past the single‑stream comm floor (~118 t/s
for a 12 B here) is real NVLink/P2P — or a model small enough to run TP=1.

## GPU selection notes (this node)
- GPU2 drives the desktop (Wayland) — avoid for heavy jobs (browser stutter).
- One GPU may carry a small resident process; if TP init fails with
  `Free memory ... less than desired GPU memory utilization`, drop `--gpu-memory-utilization`
  to 0.85 or pick clean GPUs.
