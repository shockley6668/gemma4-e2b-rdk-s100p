# Gemma4-E2B RDK S100P 量化部署

Google Gemma4-E2B（Vision + Text）在地瓜 RDK S100P 上的 PTQ 量化编译 + 板端推理代码仓库。

## 目录结构

```
gemma4-e2b-quant/
├── leap_llm_gemma4/          # Gemma4 适配代码（集成进 OE-LLM leap_llm）
│   ├── models/gemma4/        #   模型定义（Vision ViT + Text LLM）
│   │   ├── model.py
│   │   └── blocks/           #   attention / encoder_layer / mlp
│   ├── apis/model/
│   │   ├── gemma4.py         #   Gemma4VisionApi / Gemma4TextApi
│   │   └── model_factory.patch  # 注册 gemma4-* 到 model_factory.py
│   └── install.sh            # 一键集成到已安装的 leap_llm
│
├── scripts/
│   ├── compile/              # 量化编译脚本
│   │   ├── run_vision_compile.sh
│   │   ├── run_text_compile.sh
│   │   ├── compile_vision.py
│   │   ├── compile_text_decode.py
│   │   └── setup_swap.sh
│   ├── calibration/          # 校准数据准备
│   │   ├── download_coco_images.py
│   │   └── generate_calib_images.py
│   ├── verify/               # 精度验证脚本
│   │   ├── quick_text_verify.py
│   │   ├── e2e_vlm_verify.py
│   │   ├── e2e_vlm_verify_fast.py
│   │   └── run_remote_hbm_verify.sh
│   └── oss/                  # OSS 上传 / 部署打包
│       └── upload_oss_deploy.py
│
├── board_runtime/            # 板端推理代码（板端 Agent 维护）
│   └── README.md
│
└── docs/                     # 文档
    ├── QUANTIZATION_TUTORIAL.md   # 量化部署教程
    └── BOARD_VLM_FIX.md           # VLM 融合修复指南
```

## 快速开始

### 1. 环境准备

```bash
# 安装 OE-LLM SDK（见 docs/QUANTIZATION_TUTORIAL.md §3）
conda activate oellm
```

### 2. 集成 Gemma4 适配代码

```bash
cd leap_llm_gemma4
bash install.sh    # 把 gemma4 代码复制进 leap_llm 并注册
```

### 3. 量化编译

```bash
cd scripts/compile
bash run_vision_compile.sh   # Vision HBM
bash run_text_compile.sh     # Text HBM
```

### 4. 精度验证

```bash
cd scripts/verify
python quick_text_verify.py
```

## 模型文件下载

编译产出的 HBM 模型文件从 OSS 获取，见 `docs/QUANTIZATION_TUTORIAL.md` 附录。
