# 板端推理代码

> 本目录由板端 Agent 维护，存放 S100P 上 VLM 推理的 C++ runtime 代码。

板端推理需要实现以下模块（参考 `docs/BOARD_VLM_FIX.md` 中的集成要点）：

## 需要实现的功能

1. **HBM 加载** — 加载 Vision HBM 和 Text HBM（prefill + decode）
2. **Vision 预处理** — 图像 Resize 672×960 + [0,1] 归一化 + patchify
3. **Token embedding 查表** — 从 `tok_embeddings.bin` 查表 + √1536 缩放
4. **Vision 特征注入** — `masked_scatter`：将 ViT 输出 `[280,1536]` 替换到 `input_ids == 258880` 位置
5. **PLE 处理** — image token 位置使用 pad embedding（token_id=0）
6. **Prompt 构建** — 按 chat_template 构建 `<|image> [soft×280] <image|> text` 序列
7. **Mask / KV cache 管理** — 构建 full_mask / sliding_mask，管理 prefill→decode 的 KV cache
8. **Decode 循环** — 采样、追加 token、更新 KV cache

## 关键约束

- Vision 特征注入时**不要**做 L2-norm 或乘 √1536
- Mask dtype 为 float32，禁止值 -32768
- Text HBM 输入 `_input_0` 需要已含 √1536 缩放的 `inputs_embeds`

详见 `docs/BOARD_VLM_FIX.md`。
