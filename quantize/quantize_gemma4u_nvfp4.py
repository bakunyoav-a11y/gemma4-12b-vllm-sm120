#!/usr/bin/env python3
"""NVFP4-quantize a Gemma 4 *unified* (text+vision+audio) model.

Faithful to Lna-Lab's proven Gemma-4 recipe
(sakamakismile/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4), adapted for:
  - gemma4_unified arch  -> AutoModelForImageTextToText + transformers 5.10
  - many small GPUs (16GB RTX PRO 2000) -> device_map="balanced" + max_memory

THE RECIPE (RedHatAI/Lna-Lab proven path):
  scheme NVFP4 (W4A4), ignore vision/audio/embed/lm_head. The multimodal
  encoders + embeddings stay BF16; only the language_model Linear layers go FP4.
  Calibration drives the activation FP4 input_global_scale -> MUST use the
  processor's chat template (text-only neuralmagic/calibration), or activations
  collapse and the model emits <pad>.

Env: QUANT_MODEL_ID, QUANT_OUTPUT_DIR, QUANT_GPU_MEM (default 13GiB),
     QUANT_SEQ_LEN (2048), QUANT_NUM_SAMPLES (32).
"""
import os
import shutil

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = os.environ["QUANT_MODEL_ID"]
OUTPUT_DIR = os.environ["QUANT_OUTPUT_DIR"]
SEQ = int(os.environ.get("QUANT_SEQ_LEN", "2048"))
NSAMP = int(os.environ.get("QUANT_NUM_SAMPLES", "32"))
SCHEME = os.environ.get("QUANT_SCHEME", "NVFP4A16")  # W4A16 for gemma4_unified
# (W4A4/NVFP4 collapses this arch's activations -> <pad>; community serves W4A16)

# --- Gemma-4-unified recipe (W4A16 weights-FP4, activations BF16) -------------
# Ignore exactly like the known-good coolthor/gemma-4-12B-it-NVFP4A16: only the
# embedding_projection adapters + lm_head stay BF16; everything else (incl. the
# vision/audio embedders) is quantized. This keeps the saved config.ignore in
# sync with the actual quantized modules (no vision_embedder mismatch in vLLM).
import json as _json
_ig = _json.loads(os.environ.get("QUANT_IGNORE", '["lm_head", "re:.*embedding_projection.*"]'))
recipe = QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=_ig)

# --- load across the visible 16GB GPUs ---------------------------------------
n = torch.cuda.device_count()
gpu_mem = os.environ.get("QUANT_GPU_MEM", "13GiB")
max_memory = {i: gpu_mem for i in range(n)}
print(f"Loading {MODEL_ID}: device_map=balanced, {n} GPUs x {gpu_mem}", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- calibration: neuralmagic/calibration (LLM split), via chat template ------
print(f"Preparing calibration: neuralmagic/calibration LLM [:{NSAMP}] x {SEQ} tok", flush=True)
ds = load_dataset("neuralmagic/calibration", name="LLM", split=f"train[:{NSAMP}]")

def preprocess(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages, return_tensors="pt", padding=False, truncation=True,
        max_length=SEQ, tokenize=True, add_special_tokens=False,
        return_dict=True, add_generation_prompt=False,
    )

ds = ds.map(preprocess, batched=False, remove_columns=ds.column_names)

def data_collator(batch):
    assert len(batch) == 1
    return {
        k: (torch.tensor(v) if k != "pixel_values"
            else torch.tensor(v, dtype=torch.bfloat16).squeeze(0))
        for k, v in batch[0].items()
    }

# --- one-shot NVFP4 -----------------------------------------------------------
print("NVFP4 oneshot...", flush=True)
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=SEQ,
    num_calibration_samples=NSAMP,
    data_collator=data_collator,
)

print(f"Saving compressed NVFP4 checkpoint -> {OUTPUT_DIR}", flush=True)
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
processor.save_pretrained(OUTPUT_DIR)

for fn in ("chat_template.jinja", "processor_config.json", "preprocessor_config.json",
           "video_preprocessor_config.json", "generation_config.json",
           "special_tokens_map.json"):
    s = os.path.join(MODEL_ID, fn)
    d = os.path.join(OUTPUT_DIR, fn)
    if os.path.exists(s) and not os.path.exists(d):
        shutil.copy2(s, d)

print("Done!", flush=True)
