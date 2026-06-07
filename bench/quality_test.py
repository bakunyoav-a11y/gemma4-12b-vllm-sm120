#!/usr/bin/env python3
"""Reasoning-quality probe for the served quant: pattern-matching fails these,
so they reveal whether real reasoning survived NVFP4+FP8 + abliteration.
(Spec-decode is lossless — quality == non-spec — so we run plain TP=1.)"""
import os
from vllm import LLM, SamplingParams


def main():
    MODEL = os.environ.get("SANITY_MODEL", "/m")
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.93,
              enforce_eager=True, trust_remote_code=True)
    qs = [
        "A common claim: a mirror flips left and right but not up and down. "
        "Is that actually what a mirror does? State precisely which axis a mirror "
        "reverses, and explain exactly why people misdescribe it as 'left-right'.",

        "Exactly what is the angle between the hour and minute hands of an analog "
        "clock at 3:15? Reason step by step and give the precise number in degrees.",

        "Make the single strongest one-paragraph argument that 0.999... (repeating) "
        "is NOT equal to 1. Then, in a second paragraph, identify precisely where that "
        "argument is wrong.",
    ]
    prompts = [[{"role": "user", "content": q}] for q in qs]
    sp = SamplingParams(max_tokens=700, temperature=0.6, top_p=0.95)
    outs = llm.chat(prompts, sp)
    for i, o in enumerate(outs):
        print(f"\n{'='*70}\nQ{i+1}: {qs[i][:80]}...\n{'-'*70}")
        print(o.outputs[0].text.strip())
    print(f"\n{'='*70}\n[quality] done")


if __name__ == "__main__":
    main()
