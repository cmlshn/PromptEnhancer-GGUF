<div align="center">

# PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting

</div>

<p align="center">
  <!-- <a href="https://www.arxiv.org/abs/2509.04545"><img src="https://img.shields.io/badge/Paper-arXiv:2509.04545-red?logo=arxiv" alt="arXiv"></a> -->
  <a href="https://huggingface.co/tencent/HunyuanImage-2.1/tree/main/reprompt"><img src="https://img.shields.io/badge/HuggingFace-Model_V1-blue?logo=huggingface" alt="HuggingFace Model"></a>
  <a href="https://hunyuan-promptenhancer.github.io/"><img src="https://img.shields.io/badge/Homepage-PromptEnhancer-1abc9c" alt="Homepage"></a>
  <a href="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1"><img src="https://img.shields.io/badge/Code-HunyuanImage2.1-2ecc71?logo=github" alt="HunyuanImage2.1 Code"></a>
  <a href="https://huggingface.co/tencent/HunyuanImage-2.1"><img src="https://img.shields.io/badge/Model-HunyuanImage2.1-3498db?logo=huggingface" alt="HunyuanImage2.1 Model"></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Transformers-4.56%2B-FFD21E?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</p>

---

<p align="center">
  <img src="assets/teaser-1.png" alt="PromptEnhancer Teaser"/>
</p>

## Overview

Hunyuan-PromptEnhancer is a prompt rewriting utility built on top of Tencent's Hunyuan models. It restructures an input prompt while preserving the original intent, producing clearer, layered, and logically consistent prompts suitable for downstream image generation or similar tasks.

- Preserves intent across key elements (subject/action/quantity/style/layout/relations/attributes/text, etc.).
- Encourages a "global–details–summary" narrative, describing primary elements first, then secondary/background elements, ending with a concise style/type summary.
- Robust output parsing with graceful fallback: prioritizes `<answer>...</answer>`; if missing, removes `<think>...</think>` and extracts clean text; otherwise falls back to the original input.
- Configurable inference parameters (temperature, top_p, max_new_tokens) for balancing determinism and diversity.

## Installation

```bash
pip install -r requirements.txt
```

## Model Download

```bash
huggingface-cli download tencent/HunyuanImage-2.1/reprompt --local-dir ./models/
```

## Quickstart

```python
from inference.prompt_enhancer import HunyuanPromptEnhancer

# 1) Provide a local path or a repo id, e.g., "tencent/Hunyuan-7B-Instruct"
models_root_path = "./models/"

# 2) Initialize (device_map respects your configuration)
enhancer = HunyuanPromptEnhancer(models_root_path=models_root_path, device_map="auto")

# 3) Enhance a prompt (Chinese or English)
user_prompt = "Third-person view, a race car speeding on a city track..."
new_prompt = enhancer.predict(
    prompt_cot=user_prompt,
    # Default system prompt is tailored for image prompt rewriting; override if needed
    temperature=0.7,   # >0 enables sampling; 0 uses deterministic generation
    top_p=0.9,
    max_new_tokens=256,
)

print("Enhanced:", new_prompt)
```

## Parameters

- `models_root_path`: Local path or repo id; supports `trust_remote_code` models.
- `device_map`: Device mapping (default `auto`).
- `predict(...)`:
  - `prompt_cot` (str): Input prompt to rewrite.
  - `sys_prompt` (str): Optional system prompt; a default is provided for image prompt rewriting.
  - `temperature` (float): `>0` enables sampling; `0` for deterministic generation.
  - `top_p` (float): Nucleus sampling threshold (effective when sampling).
  - `max_new_tokens` (int): Maximum number of new tokens to generate.

## TODO

- [ ] Add large parameter PromptEnhancer version (PromptEnhancer V2)

## License

This project is distributed under the terms specified in `LICENSE`.

## Acknowledgements

We would like to thank the following open-source projects and communities for their contributions to open research and exploration: [Transformers](https://huggingface.co/transformers) and [HuggingFace](https://huggingface.co).

## Contact

If you would like to leave a message for our R&D and product teams, Welcome to contact our open-source team . You can also contact us via email (hunyuan_opensource@tencent.com).