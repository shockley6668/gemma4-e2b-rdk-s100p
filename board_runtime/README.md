# Board Runtime — S100P 板端 VLM 推理

> **English** | [中文](#中文)

---

## English

C++ inference runtime for Gemma4-E2B VLM on D-Robotics RDK S100P (`march=nash-m`). Loads pre-compiled HBM models and runs real-time Vision-Language inference on the BPU.

### Prerequisites

The S100P board must have the OE-LLM runtime installed:

```bash
# Verify Horizon BPU SDK
ls /usr/hobot/lib/libdnn.so    # BPU inference lib
ls /usr/hobot/lib/libhbucp.so  # Memory management lib
ls /usr/include/hobot/dnn/hb_dnn.h
```

System dependencies (usually pre-installed on OE-LLM images):

```bash
sudo apt install cmake g++ libopencv-dev cargo
# nlohmann-json is provided by the system; if missing: sudo apt install nlohmann-json3-dev
```

> **No Python required.** Tokenization is done in native C++ via
> `tokenizers-cpp` (vendored in `third_party/`), matching the
> OpenExplorer_LLM-s600 reference implementation.

### Directory Layout

```
board_runtime/
└── cpp/                          C++ source code
    ├── CMakeLists.txt            Build entry point (pulls in tokenizers-cpp)
    ├── gemma4_config.h           Model constants (image token IDs, dims, ...)
    ├── gemma4_text_engine.*      Text LLM engine (prefill + decode + KV cache)
    ├── gemma4_vision_engine.*    Vision ViT engine
    ├── gemma4_embeddings.*       Token embedding lookup + vision injection
    ├── gemma4_kv_cache.*         Zero-copy KV cache management
    ├── gemma4_vision_preprocess.* Image resize + patchify
    ├── gemma4_native_tokenizer.* Native C++ tokenizer (from OE-LLM-s600)
    ├── gemma4_tokenizer.*        TokenizerBridge: chat template + image expand
    ├── hb_utils.h                Horizon BPU helpers (tensor, flush, infer)
    ├── gemma4_chat.cpp           ★ Interactive VLM chat (primary entry)
    ├── gemma4_server.cpp         HTTP API server
    ├── gemma4_demo.cpp           Single-shot VLM demo
    ├── gemma4_text_bench.cpp     Text-only benchmark
    └── gemma4_golden_verify.cpp  Golden mask/KV alignment checker

third_party/
└── tokenizers-cpp/               Vendored HF tokenizers C++ binding + sentencepiece
```

### Build

```bash
cd board_runtime/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The first build compiles the vendored `tokenizers-cpp` (HF tokenizers Rust
binding + sentencepiece + abseil), which takes a few minutes. Subsequent
builds are incremental and fast.

This produces 5 executables in `build/`:

| Binary | Description |
|--------|-------------|
| `gemma4_chat` | Interactive VLM chat with streaming output |
| `gemma4_server` | HTTP API server for programmatic access |
| `gemma4_demo` | Single-shot: image + prompt → text |
| `gemma4_text_bench` | Text-only inference benchmark |
| `gemma4_golden_verify` | Verify prefill tensors against golden data |

### Download Pre-compiled Models

```bash
pip install huggingface_hub
hf download ShockleyWong/gemma4-e2b-rdk-s100p --local-dir ~/gemma4_e2b
```

This downloads 3 model files + tokenizer:

```
~/gemma4_e2b/
├── model/
│   ├── gemma4-e2b_vit_ptq.hbm                          # 329 MB  Vision
│   ├── gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm      # 4.5 GB  Text
│   └── tok_embeddings.bin                               # 1.5 GB  Embedding
└── tokenizer/
    ├── tokenizer.json                                   # 32 MB
    ├── tokenizer_config.json
    ├── chat_template.jinja
    └── config.json
```

### Run

Set `GEMMA4_HOME` to point at the model directory, then run:

```bash
export GEMMA4_HOME=~/gemma4_e2b

# Interactive VLM chat
./gemma4_chat

# Inside the chat:
#   /image /path/to/photo.jpg   Load an image
#   What do you see in this image?   Ask a question
#   /reset                       Clear conversation
#   /quit                        Exit
```

Example output:

```
gemma4> /image test.jpg
Processing image: test.jpg...
Image loaded (430080 features).
gemma4> Describe this image
This is a photograph of a Red Panda resting on a wooden structure...
```

### Key Design Decisions

1. **Vision injection is raw** — ViT output `[280, 1536]` is injected directly into `inputs_embeds` at image soft-token positions (token ID 258880). No L2-norm scaling, no √1536 multiplication.

2. **PLE uses pad embedding** — At image positions, the Per-Layer Embedding token-identity path uses `pad_token_id=0` (not 258880), matching HuggingFace's `masked_scatter` behavior.

3. **Chat template** — Prompts are formatted in C++ to the Gemma turn format (`<bos><|turn>user\n...<turn|>\n<|turn>model\n`), matching `chat_template.jinja`. Tokenization uses the native `tokenizers-cpp` (HF tokenizers), not Python.

4. **Zero-copy KV cache** — KV cache memory is allocated once and shared between prefill and decode via pointer assignment, avoiding per-step memcpy.

5. **Chunked prefill** — Prompts longer than `chunk_size=256` tokens are automatically split into multiple prefill chunks.

### Verification

To verify board inference matches the PC golden data:

```bash
# Place golden_mask_kv/ under $GEMMA4_HOME/golden_mask_kv/
./gemma4_golden_verify --prompt prompt_0
# Expected: ALL PASSED (cosine=1.0 for all 5 tensors)
```

---

## 中文

S100P 板端 Gemma4-E2B VLM 推理 C++ runtime，加载预编译 HBM 模型在 BPU 上跑实时视觉语言推理。

### 前置条件

板子需安装 OE-LLM runtime：

```bash
ls /usr/hobot/lib/libdnn.so
ls /usr/hobot/lib/libhbucp.so
```

系统依赖：

```bash
sudo apt install cmake g++ libopencv-dev cargo
```

> **无需 Python**。分词走原生 C++ `tokenizers-cpp`（vendored 在 `third_party/`），与 OpenExplorer_LLM-s600 参考实现一致。

### 编译

```bash
cd board_runtime/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

首次编译会构建 vendored 的 `tokenizers-cpp`（HF tokenizers Rust 绑定 + sentencepiece + abseil），耗时数分钟；之后增量编译很快。

### 下载预编译模型

```bash
pip install huggingface_hub
hf download ShockleyWong/gemma4-e2b-rdk-s100p --local-dir ~/gemma4_e2b
```

### 运行

```bash
export GEMMA4_HOME=~/gemma4_e2b
./gemma4_chat
```

### 核心设计

- **Vision 原样注入**：ViT 输出直接替换 image 位置，不做缩放
- **PLE 用 pad**：image 位置的 per-layer embedding 用 `pad_token_id=0`
- **Chat template**：C++ 内拼成 Gemma turn 格式，原生 `tokenizers-cpp` 分词，不依赖 python
- **零拷贝 KV cache**：prefill/decode 共享内存

### 验证

```bash
./gemma4_golden_verify --prompt prompt_0
# 预期：ALL PASSED
```
