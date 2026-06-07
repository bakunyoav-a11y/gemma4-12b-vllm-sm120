# NVFP4 quantization container for Gemma 4 *unified* (text+vision+audio) models.
# Base gives us torch 2.11+cu130 + Blackwell (sm_120) ready to go.
# We add: transformers 5.10.2 (gemma4_unified support) + llm-compressor main
# (transformers 5.x compatible; released 0.11 still pins <=4.57) + datasets.
FROM vllm/vllm-openai:cu130-nightly

ENV PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# git is needed to pip-install llm-compressor from source (not in base image).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# llm-compressor from git main pulls a compatible compressed-tensors.
RUN pip install "git+https://github.com/vllm-project/llm-compressor.git" "datasets>=3.0"

# Force transformers to 5.10.2 last so gemma4_unified is recognized
# (overrides whatever pin llm-compressor brings in).
RUN pip install -U "transformers==5.10.2"

# Sanity: confirm the unified arch is registered and the quant stack imports.
RUN python3 -c "import transformers, llmcompressor, compressed_tensors, datasets, torch; \
print('torch', torch.__version__); print('transformers', transformers.__version__); \
print('llmcompressor', llmcompressor.__version__); print('compressed_tensors', compressed_tensors.__version__); \
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C; \
assert 'gemma4_unified' in C, 'gemma4_unified NOT registered'; print('gemma4_unified: OK')"

WORKDIR /work
ENTRYPOINT ["python3"]
