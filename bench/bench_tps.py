#!/usr/bin/env python3
"""Aggregate parallel-request TPS benchmark (vLLM offline, continuous batching).
Submits NREQ prompts at once; vLLM batches them. Aggregate TPS = total output
tokens / wall time. ignore_eos forces exactly OUTTOK tokens per request for a
clean decode-throughput measurement. Guarded under __main__ for TP>1 spawn."""
import os, time
from vllm import LLM, SamplingParams


def main():
    TP     = int(os.environ.get("BENCH_TP", "1"))
    NREQ   = int(os.environ.get("BENCH_NREQ", "64"))
    OUTTOK = int(os.environ.get("BENCH_OUTTOK", "256"))
    MAXLEN = int(os.environ.get("BENCH_MAXLEN", "65536"))
    MODEL  = os.environ.get("BENCH_MODEL", "/model")

    UTIL = float(os.environ.get("BENCH_GPU_UTIL", "0.90"))
    kw = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        max_model_len=MAXLEN,
        gpu_memory_utilization=UTIL,
        trust_remote_code=True,
        # No NVLink/P2P on this box: vLLM custom all-reduce (P2P-based) deadlocks
        # the forward at TP>1. Pair with NCCL_P2P_DISABLE=1 in the env.
        disable_custom_all_reduce=(TP > 1),
    )
    kvdt = os.environ.get("BENCH_KV_DTYPE")  # e.g. fp8 (required with spec-decode)
    if kvdt:
        kw["kv_cache_dtype"] = kvdt
    mns = os.environ.get("BENCH_MAX_NUM_SEQS")  # low (e.g. 4) frees graph mem -> longer -c
    if mns:
        kw["max_num_seqs"] = int(mns)
    spec = os.environ.get("BENCH_SPEC_MODEL")  # MTP drafter path -> speculative decode
    if spec:
        kw["speculative_config"] = {
            "method": "mtp",
            "model": spec,
            "num_speculative_tokens": int(os.environ.get("BENCH_SPEC_NUM", "4")),
        }
        print(f"[bench] speculative MTP draft={spec} k={kw['speculative_config']['num_speculative_tokens']}", flush=True)
    print(f"[bench] loading TP={TP} max_model_len={MAXLEN} util={UTIL} kv={kvdt or 'auto'} ...", flush=True)
    llm = LLM(**kw)

    TOPICS = [
        "the CAP theorem", "quantum entanglement", "the French Revolution",
        "how transformers work", "photosynthesis", "the theory of relativity",
        "Bayesian inference", "the Roman Empire", "neural network backprop",
        "black holes", "the stock market", "DNA replication", "climate change",
        "the internet protocol stack", "evolution by natural selection", "monetary policy",
    ]
    prompts = [[{"role": "user",
                 "content": f"(req {i}) Explain {TOPICS[i % len(TOPICS)]} in thorough detail."}]
               for i in range(NREQ)]

    # warmup (compile CUDA graphs, etc.)
    llm.chat([[{"role": "user", "content": "hello"}]],
             SamplingParams(max_tokens=8, temperature=0), use_tqdm=False)

    sp = SamplingParams(max_tokens=OUTTOK, temperature=0.7, top_p=0.9, ignore_eos=True)
    t0 = time.time()
    outs = llm.chat(prompts, sp, use_tqdm=False)
    dt = time.time() - t0

    tot_out = sum(len(o.outputs[0].token_ids) for o in outs)
    agg = tot_out / dt
    print(f"\n##### RESULT TP={TP} #####", flush=True)
    print(f"requests={NREQ}  out_tokens/req={OUTTOK}  max_model_len={MAXLEN}", flush=True)
    print(f"total_output_tokens={tot_out}  wall={dt:.2f}s", flush=True)
    print(f"AGGREGATE_TPS={agg:.1f} tok/s   per_request_avg={agg/NREQ:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
