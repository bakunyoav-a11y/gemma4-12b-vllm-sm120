# Quantizing a `gemma4_unified` model to NVFP4 yourself (bonus)

You don't need this to serve — the AEON checkpoints are pre‑quantized. This documents the
`quantize/` pipeline (llm‑compressor → compressed‑tensors NVFP4) and the **non‑obvious
traps** for the new `gemma4_unified` arch, in case you want to quantize a different
Gemma‑4 derivative.

## Container
`quantize/Dockerfile` builds on `vllm/vllm-openai:nightly` (gives torch 2.11+cu130,
transformers 5.10.2, Blackwell), adds `git`, llm‑compressor from **git main**, and
`datasets`. Build: `docker build -t gemma4u-quant quantize/`.

## Recipe (`quantize/quantize_gemma4u_nvfp4.py`)
`QuantizationModifier(targets="Linear", scheme=<SCHEME>, ignore=[...])`, calibrated with
the **multimodal processor's chat template** over `neuralmagic/calibration` (32×2048).
Loaded with `device_map="balanced"` so a 24 GB BF16 model spreads across several 16 GB
GPUs.

## Traps (learned the hard way)

1. **`gemma4_unified` needs transformers ≥ 5.10.** The released `llm-compressor` (0.11)
   pins `transformers <= 4.57`, which can't even load the model → **use llm‑compressor
   git main**, then force `transformers==5.10.2` back.

2. **W4A4 NVFP4 collapses this arch.** Full activation FP4 makes the model emit `<pad>`.
   The community (and this repo) use **W4A16 (`scheme="NVFP4A16"`)** or **ModelOpt FP8 /
   mixed** instead. "Attention as NVFP4 hurts reasoning" — keep attention BF16 or FP8.

3. **Pin `compressed-tensors==0.17.0`.** llm‑compressor main pulls `0.17.1a` (an alpha
   AHEAD of vLLM's 0.17.0) whose NVFP4 **pack format vLLM mis‑decodes → garbage output**.
   Bake with the *same* ct version the runtime ships (0.17.0).

4. **`ignore` list must match the real module names.** llm‑compressor's regex
   (`re:.*vision.*`, `re:.*audio.*`, `re:.*embed.*`, `lm_head`) is correct at bake time,
   but the saved `config.json` ignore list can be written with names that don't match the
   checkpoint, so vLLM tries to quantize a BF16 module and fails to load
   (`no parameter named vision_embedder.patch_dense.weight`). Use the coolthor‑style
   `ignore: [lm_head, 're:.*embedding_projection.*']`, or patch `config.json` after baking
   so its `quantization_config.ignore` covers every BF16 Linear.

## Verify
Always sanity‑check generation after baking (`bench/diag_generate.py` forces tokens and
prints raw ids). Greedy `<pad>` output = a broken bake (or a broken base — see the huihui
warning in the main README). A coherent paragraph = good.
