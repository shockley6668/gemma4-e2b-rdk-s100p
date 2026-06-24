# Gemma4-E2B on RDK S100P

**中文** | [English](./README.md)

Google **Gemma4-E2B**（视觉 + 语言多模态）在 **地瓜 RDK S100P**（`march=nash-m`）板端的实时 VLM 推理。完全在 BPU 上跑，无需联网，推理时不依赖 Python。

![VLM](https://img.shields.io/badge/VLM-可用-green) ![平台](https://img.shields.io/badge/平台-RDK%20S100P-blue) ![许可证](https://img.shields.io/badge/许可证-MIT-lightgrey)

## 快速开始

### 1. 下载预编译模型

```bash
pip install huggingface_hub
hf download ShockleyWong/gemma4-e2b-rdk-s100p --local-dir ~/gemma4_e2b
```

### 2. 编译 C++ runtime（在板端）

需要 OE-LLM 板端镜像，以及 `cmake`、`g++`、`libopencv-dev`、`cargo`。

```bash
cd board_runtime/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

首次编译会构建自带的 `tokenizers-cpp`（原生 HF tokenizers），耗时数分钟；之后增量编译很快。

### 3. 运行

```bash
export GEMMA4_HOME=~/gemma4_e2b
./gemma4_chat
```

进入交互后：

```
gemma4> /image photo.jpg          # 为下一条消息加载图片
gemma4> 描述这张图片              # 提问
gemma4> /reset                    # 清空对话 + KV cache
gemma4> /quit                     # 退出
```

## 仓库内容

| 路径 | 用途 |
|------|------|
| `board_runtime/cpp/` | **板端 C++ 推理 runtime** — `gemma4_chat` 主入口，含文本/视觉引擎、KV cache、原生 C++ tokenizer（不依赖 Python）。 |
| `third_party/tokenizers-cpp/` | 自带的 HuggingFace tokenizers C++ 绑定 + sentencepiece。 |
| `leap_llm_gemma4/` | OE-LLM 量化工具链用的 Gemma4 PyTorch 模型定义。 |
| `scripts/` | PC 端脚本：HBM 编译、校准、精度验证。 |
| `docs/QUANTIZATION_TUTORIAL.md` | 完整指南：量化 → 部署 → VLM 推理。 |

## 从源码重新编译模型

如果你想自己重新量化或修改 HBM 模型（需要 128GB 内存的 PC + OE-LLM SDK），请看 [完整量化教程](./docs/QUANTIZATION_TUTORIAL.md)。

## 许可证

MIT
