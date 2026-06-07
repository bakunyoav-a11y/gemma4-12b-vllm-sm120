#!/usr/bin/env python3
"""Diagnose empty output: is it immediate-EOS (template/stop issue) or garbage
(weights/calibration)? Inspect finish_reason, token ids, and force generation
with ignore_eos to see the raw decode."""
import os
from vllm import LLM, SamplingParams


def main():
    MODEL = os.environ.get("SANITY_MODEL", "/output")
    TP = int(os.environ.get("DIAG_TP", "1"))
    UTIL = float(os.environ.get("DIAG_GPU_UTIL", "0.85"))
    # No NVLink/P2P on this box -> vLLM custom all-reduce (needs P2P) deadlocks the
    # forward; disable it (pair with NCCL_P2P_DISABLE=1 in the env).
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=UTIL,
              tensor_parallel_size=TP, enforce_eager=True, trust_remote_code=True,
              disable_custom_all_reduce=(TP > 1))
    tok = llm.get_tokenizer()

    msg = [[{"role": "user", "content": "What is the CAP theorem? Explain in detail."}]]

    # 1) default sampling (let it stop naturally)
    o1 = llm.chat(msg, SamplingParams(max_tokens=128, temperature=0.7), use_tqdm=False)[0].outputs[0]
    print(f"[default]  finish={o1.finish_reason} ntok={len(o1.token_ids)} ids={list(o1.token_ids)[:15]}")
    print(f"[default]  text={o1.text[:200]!r}")

    # 2) force tokens (ignore_eos) -> reveals whether decode is coherent
    o2 = llm.chat(msg, SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True, min_tokens=64),
                  use_tqdm=False)[0].outputs[0]
    print(f"\n[forced]   finish={o2.finish_reason} ntok={len(o2.token_ids)} ids={list(o2.token_ids)[:15]}")
    print(f"[forced]   text={o2.text[:300]!r}")

    # 3) raw prompt with explicit gemma turn template (bypass chat-template quirks)
    raw = "<start_of_turn>user\nWhat is the CAP theorem?<end_of_turn>\n<start_of_turn>model\n"
    o3 = llm.generate([raw], SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True, min_tokens=64),
                      use_tqdm=False)[0].outputs[0]
    print(f"\n[raw+force] ntok={len(o3.token_ids)} ids={list(o3.token_ids)[:15]}")
    print(f"[raw+force] text={o3.text[:300]!r}")

    # what is the prompt token stream the chat template produced?
    pt = tok.apply_chat_template(msg[0], add_generation_prompt=True, tokenize=True)
    print(f"\n[tmpl] prompt_ids tail={pt[-10:]}  decoded={tok.decode(pt)[-120:]!r}")
    print("[diag] done")


if __name__ == "__main__":
    main()
