# Gemma4-E2B RDK S100P Quantization & Deployment / 量化部署

> **English** | [中文](#中文)

---

## English

Google Gemma4-E2B (Vision + Text) PTQ quantization and board deployment for D-Robotics RDK S100P (`march=nash-m`).

### Directory Structure

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

### Quick Start

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

### Model Files

Pre-compiled HBM files are available on HuggingFace. See `docs/QUANTIZATION_TUTORIAL.md` Appendix.

---

## 中文

Google Gemma4-E2B（Vision + Text）在地瓜 RDK S100P（`march=nash-m`）上的 PTQ 量化编译与板端部署。

### 目录结构

```
gemma4-e2b-quant/
├── leap_llm_gemma4/          # Gemma4 适配代码（集成进 OE-LLM leap_llm）
│   ├── models/gemma4/        #   模型定义（Vision ViT + Text LLM）
│   ├── apis/model/           #   API 层（Gemma4VisionApi / Gemma4TextApi）
│   └── install.sh            #   一键集成进 leap_llm
│
├── scripts/
│   ├── compile/              # 量化编译脚本
│   │   ├── run_vision_compile.sh
│   │   ├── run_text_compile.sh
│   │   ├── run_text_decode_resume.sh
│   │   ├── compile_vision.py
│   │   ├── compile_text_decode.py
│   │   └── setup_swap.sh
│   ├── calibration/          # 校准数据准备
│   │   ├── download_coco_images.py
│   │   └── generate_calib_images.py
│   └── verify/               # 精度验证脚本
│       ├── quick_text_verify.py
│       └── run_remote_hbm_verify.sh
│
├── board_runtime/            # 板端推理代码（板端 Agent 维护）
│
└── docs/
    └── QUANTIZATION_TUTORIAL.md   # 完整量化教程
```

### 快速开始

```bash
# 1. 安装 OE-LLM SDK（见 docs/QUANTIZATION_TUTORIAL.md）
conda activate oellm

# 2. 将 Gemma4 代码集成进 leap_llm
cd leap_llm_gemma4 && bash install.sh

# 3. 编译 HBM 模型
cd scripts/compile
bash run_vision_compile.sh    # Vision HBM
bash run_text_compile.sh      # Text HBM

# 4. 精度验证
cd scripts/verify
python quick_text_verify.py
```

### 模型文件

预编译的 HBM 文件已上传 HuggingFace，见 `docs/QUANTIZATION_TUTORIAL.md` 附录。
