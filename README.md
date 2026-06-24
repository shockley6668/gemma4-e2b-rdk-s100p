# Gemma4-E2B RDK S100P Quantization & Deployment

[中文](./README_zh.md) | **English**

Google Gemma4-E2B (Vision + Text) PTQ quantization and board deployment for D-Robotics RDK S100P (`march=nash-m`).

## Directory Structure

```
gemma4-e2b-quant/
├── leap_llm_gemma4/          # Gemma4 adaptation code (integrates into OE-LLM leap_llm)
│   ├── models/gemma4/        #   Model definitions (Vision ViT + Text LLM)
│   ├── apis/model/           #   API layer (Gemma4VisionApi / Gemma4TextApi)
│   └── install.sh            #   One-command integration into leap_llm
│
├── scripts/
│   ├── compile/              # Quantization compilation scripts
│   │   ├── run_vision_compile.sh
│   │   ├── run_text_compile.sh
│   │   ├── run_text_decode_resume.sh
│   │   ├── compile_vision.py
│   │   ├── compile_text_decode.py
│   │   └── setup_swap.sh
│   ├── calibration/          # Calibration data preparation
│   │   ├── download_coco_images.py
│   │   └── generate_calib_images.py
│   └── verify/               # Accuracy verification scripts
│       ├── quick_text_verify.py
│       └── run_remote_hbm_verify.sh
│
├── board_runtime/            # Board-side inference code (maintained by board agent)
│
└── docs/
    └── QUANTIZATION_TUTORIAL.md   # Full quantization tutorial
```

## Quick Start

```bash
# 1. Install OE-LLM SDK (see docs/QUANTIZATION_TUTORIAL.md)
conda activate oellm

# 2. Integrate Gemma4 code into leap_llm
cd leap_llm_gemma4 && bash install.sh

# 3. Compile HBM models
cd scripts/compile
bash run_vision_compile.sh    # Vision HBM
bash run_text_compile.sh      # Text HBM

# 4. Verify accuracy
cd scripts/verify
python quick_text_verify.py
```

## Model Files

Pre-compiled HBM files are available on HuggingFace. See `docs/QUANTIZATION_TUTORIAL.md` Appendix.
