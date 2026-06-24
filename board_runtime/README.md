# Board Runtime / 板端推理代码

> **English** | [中文](#中文)

---

## English

This directory contains the S100P board-side VLM inference runtime code (C++).
Maintained by the board agent.

### Modules to Implement

1. **HBM Loading** — Load Vision HBM and Text HBM (prefill + decode)
2. **Vision Preprocessing** — Image resize 672x960 + [0,1] normalization + patchify
3. **Token Embedding Lookup** — Lookup from `tok_embeddings.bin` + scale by sqrt(1536)
4. **Vision Feature Injection** — `masked_scatter`: replace ViT output `[280,1536]` at `input_ids == 258880` positions
5. **PLE Handling** — Image token positions use pad embedding (token_id=0)
6. **Prompt Construction** — Build `<|image> [soft x280] <image|> text` sequence per chat_template
7. **Mask / KV Cache Management** — Build full_mask / sliding_mask, manage prefill→decode KV cache
8. **Decode Loop** — Sampling, token append, KV cache update

### Key Constraints

- Do NOT apply L2-norm or multiply by sqrt(1536) when injecting vision features
- Mask dtype is float32, forbidden value is -32768
- Text HBM input `_input_0` must contain `inputs_embeds` already scaled by sqrt(1536)

---

## 中文

本目录存放 S100P 板端 VLM 推理的 C++ runtime 代码，由板端 Agent 维护。

### 需要实现的模块

1. **HBM 加载** — 加载 Vision HBM 和 Text HBM（prefill + decode）
2. **Vision 预处理** — 图像 Resize 672×960 + [0,1] 归一化 + patchify
3. **Token embedding 查表** — 从 `tok_embeddings.bin` 查表 + √1536 缩放
4. **Vision 特征注入** — `masked_scatter`：将 ViT 输出 `[280,1536]` 替换到 `input_ids == 258880` 位置
5. **PLE 处理** — image token 位置使用 pad embedding（token_id=0）
6. **Prompt 构建** — 按 chat_template 构建 `<|image> [soft×280] <image|> text` 序列
7. **Mask / KV cache 管理** — 构建 full_mask / sliding_mask，管理 prefill→decode 的 KV cache
8. **Decode 循环** — 采样、追加 token、更新 KV cache

### 关键约束

- Vision 特征注入时**不要**做 L2-norm 或乘 √1536
- Mask dtype 为 float32，禁止值 -32768
- Text HBM 输入 `_input_0` 需要已含 √1536 缩放的 `inputs_embeds`
