# Gemma4 E2B 板端 VLM 修复指南（给板端 Agent）

> **编写**：PC 开发机（2026-06-23）  
> **背景**：板端报告 Vision 注入后输出乱码 / 语义错误；PC 已完成端到端 VLM 融合验证。  
> **结论**：量化拆分正确；问题更可能在板端 runtime 集成，而非 Vision HBM 本身。

---

## 1. PC 端刚做了什么（端到端 VLM 验证）

此前 PC 只做了**分段验证**（Vision BC 单独、Text BC 单独），**从未测过 Vision→Text 融合**。本次首次在 PC 跑完整融合链路。

脚本：`scripts/verify/e2e_vlm_verify_fast.py`（如有）  
结果：`output/e2e_vlm_verify_fast.json`

测试图：`calibration_data/images/coco_00_000000000802.jpg`  
Prompt：`What do you see in this image? Describe it briefly.`  
序列格式：`<|image>` + 280×`258880` + `<image|>` + 文本（共 294 tokens）

| 测试 | 内容 | 结果 |
|------|------|------|
| A | 无 Vision 注入（IMAGE placeholder 用 embed_tokens） | baseline |
| B | float32 Vision 输出注入 → PyTorch Text prefill | ground truth |
| C | **BC Vision 输出注入** → PyTorch Text prefill | **部署 ViT 路径** |
| D | BC Vision + Text BC 全链路 | PC 侧脚本 KV cache 维度 bug，未完成 |

### 1.1 关键数字（COCO 校准 Vision HBM）

| 指标 | 值 | 说明 |
|------|-----|------|
| ViT 输出 cosine（f32 vs BC） | **0.9888** | 与此前单独 ViT BC 验证一致 |
| **融合后 logits cosine（f32 ViT 注入 vs BC ViT 注入）** | **0.9998** | PTQ ViT 注入后 Text prefill logits 几乎不变 |
| logits cosine（注入 vs 无注入） | 0.9923 | 注入确实改变了分布（融合通路有效） |

### 1.2 Vision features 幅度（PC 实测，板端应对齐）

| 统计量 | Vision HBM 输出 | Text token embedding（已含 √1536） |
|--------|-----------------|-------------------------------------|
| std | **~0.785** | ~1.19 |
| 每行 L2 norm | **~30.5** | — |
| std 比值 (img/txt) | **~0.66** | — |

**重要**：Vision 输出 std 本来就应该比 text embedding 小，这是正常现象，不是 bug。

### 1.3 板端已观察 vs PC 结论

| 板端现象 | PC 解释 |
|----------|---------|
| 无注入 → "Please provide the image" | 合理：模型知道没有真实图像 |
| 原始注入 std≈0.68 → gibberish | 幅度偏低 + 可能还有方向/格式问题 |
| L2-norm 缩放 std≈1.04 → 连贯但语义错 | **L2 缩放是错误的**；不应强行对齐 text embed 幅度 |
| PC BC 验证「没问题」 | 旧验证**没测融合**，只测了 ViT 和 Text 各自独立 |

---

## 2. 根因判断（PC 侧结论）

1. **Vision / Text HBM 拆分正确**  
   Vision HBM 已含 ViT + projector，输出 `[280, 1536]`；Text HBM 输入 `inputs_embeds [seq, 1536]`，外挂 `tok_embeddings.bin`。

2. **COCO 重校准 Vision HBM 在融合链路上可用**  
   BC ViT 注入后，PyTorch Text 的 prefill logits 与 float32 ViT 注入相差仅 **0.02%**（cosine 0.9998）。

3. **板端问题更可能在 runtime 集成**  
   - Vision 注入方式错误（L2-norm、重复乘 √1536）  
   - PLE（Per-Layer Embeddings）对 image 位置处理不对  
   - Prompt / chat template 格式不对  
   - 仍在使用旧版合成图校准 Vision HBM  

4. **尚未在 PC 验证**  
   Text HBM + BC Vision 注入的全部署路径（section D）；板端应用 golden 对齐 Text prefill 逐步排查。

---

## 3. 板端必须确认的 Vision HBM 版本

**当前权威 SHA256**（COCO 50 图重校准，2026-06-23）：

```
470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1  gemma4-e2b_vit_ptq.hbm
```

旧版（合成图校准，已废弃）请勿再使用。

### 3.1 仅换 Vision（推荐先做）

```bash
ossutil cp oss://gemma-model/gemma4_e2b/gemma4-e2b_vit_ptq.hbm \
  /root/gemma4_e2b/model/gemma4-e2b_vit_ptq.hbm \
  -e oss-cn-beijing.aliyuncs.com

sha256sum /root/gemma4_e2b/model/gemma4-e2b_vit_ptq.hbm
# 必须输出: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
```

### 3.2 全量部署包（可选）

```
oss://gemma-model/gemma4_e2b/gemma4_e2b_deploy.tar.gz
SHA256: b7694c9beebf77b3804688c447c2f6f1c8463673da938a33a74dde1c2d6712af
```

Text HBM / tok_embeddings **未变**：

```
3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f  Text HBM
43552a4faeee1929ad6ab818e52b459c0d70e4b183897e181a2e4709f36523a9  tok_embeddings.bin
```

---

## 4. 板端修复方向（按优先级）

### P0 — Vision 注入（最常见错误点）

**正确做法**（对齐 HuggingFace `masked_scatter`）：

1. 对普通 token：`inputs_embeds[i] = tok_embeddings.bin[token_id]`（bin 内已含 `× √1536`）
2. 对 image soft token（`token_id = 258880`，共 280 个）：  
   `inputs_embeds[i] = vision_hbm_output[row]`，row 从 0 到 279 顺序对应  
3. **直接写入，不做任何后处理**

**禁止**：

- ❌ 对 vision features 做 L2-norm 缩放到 text embedding 的 std  
- ❌ 对 vision features 再乘 `√1536`  
- ❌ 用 IMAGE token 的 embed_tokens 查表值代替 vision 输出  

**自检**（注入后打印统计）：

```
vision_hbm_output: std ≈ 0.78~0.79, L2/row ≈ 30~31
text_embeds:       std ≈ 1.1~1.2
ratio img_std/txt_std ≈ 0.66   ← 正常，不要强行拉到 1.0
```

### P0 — Token 序列格式

特殊 token ID（来自 `config.json`）：

| Token | ID | 含义 |
|-------|-----|------|
| `<|image>` | 255999 | 图像块开始 |
| `<|image|>`（soft placeholder） | **258880** | 每个占 1 个位置，共 **280** 个 |
| `<image|>` | 258882 | 图像块结束 |

单图 prompt 结构：

```
[255999] + [258880]×280 + [258882] + [text tokens...]
```

完整对话应走 **chat template**（`tokenizer/chat_template.jinja`），不是裸文本：

```
<|turn>user
<|image|>
[280 soft tokens — runtime 用 Vision HBM 填充]
<turn|>
<|turn>model
[decode 从这里开始]
```

用户消息里 `type: image` 在 template 中会展开为 `<|image|>`，再由 processor/runtime 插入 280 个 soft token。

### P0 — Vision 预处理（与 HBM 输入必须一致）

| 项 | 值 |
|----|-----|
| Resize | **672 × 960**（H×W） |
| 归一化 | `[0, 1]`（ToTensor，**无** ImageNet mean/std） |
| Patch | 16×16 |
| Patch grid | 42×60 = **2520** |
| HBM 输入 | `[2520, 768]` **f16**（patchify 后） |
| HBM 输出 | `[280, 1536]` **f32** |

### P1 — PLE（Per-Layer Embeddings）

Gemma4 Text 每层有辅助 per-layer embedding，由两部分合并后 `× 1/√2`：

1. **token-identity**：`embed_tokens_per_layer(token_id)`  
   - 对 **image 位置（258880）**：HF 使用 **pad embedding（token_id=0）**，不是 image token 的 per-layer embed  
2. **context-aware**：`RMSNorm(per_layer_model_projection(inputs_embeds))`  
   - 依赖主路径 `inputs_embeds`，因此 **Vision 注入必须正确**

板端 C++ 应用 `embed_tokens_per_layer` 查表时，image 位置请用 **pad（0）** 的 per-layer 向量。

### P1 — Text prefill / decode IO

- `chunk_size=256`, `cache_len=4096`  
- prefill 输入 `_input_0`: `[256, 1536]` f32 — 即 **已注入 Vision 的 inputs_embeds**  
- `_input_1`: `input_ids` [1, 256] i64  
- mask / KV cache dtype 与 golden 一致（mask: f32, 禁止值 -32768）  

详见：`GEMMA4_E2B_PC_DELIVERY_TO_S100.md`、`output/gemma4_e2b_text_io_dump.txt`

### P2 — 用 golden 逐步对齐

OSS：`oss://gemma-model/gemma4_e2b/pc_delivery/golden_mask_kv.tar.gz`  
文档：`GOLDEN_MASK_KV_README.md`

建议顺序：

1. 对齐 `input_ids` / `position_ids` / `inputs_embeds`（含 Vision 注入后）  
2. 对齐 `full_mask` / `sliding_mask`  
3. 对齐 KV cache before/after prefill chunk_0  
4. 再跑 decode step_0  

---

## 5. 板端诊断 checklist

按顺序执行，每步记录日志：

- [ ] Vision HBM SHA256 = `47079184...`  
- [ ] `hrt_model_exec model_info` Vision/Text HBM 均 PASS  
- [ ] 预处理输出 `[2520,768]` f16，数值范围 [0,1] patchify 后  
- [ ] Vision HBM 输出 `[280,1536]` f32，std≈0.78，L2/row≈30.5  
- [ ] 注入后 **未做** L2-norm / √1536  
- [ ] `input_ids` 中 280 个位置均为 `258880`  
- [ ] PLE image 位置用 pad per-layer embed  
- [ ] Prompt 含 `<|turn>user` / `<|turn>model` 结构  
- [ ] Text prefill 用注入后的 `inputs_embeds` 作为 `_input_0`  
- [ ] 与 golden_mask_kv 逐步 cosine 对齐  

---

## 6. 板端应回报的结果

请回报以下信息，便于 PC 侧继续分析：

1. Vision HBM 实际 SHA256  
2. Vision 输出统计：`mean, std, L2/row`（注入前）  
3. 注入后 `inputs_embeds` 在 image 位置的 `std`（应 ≈0.78，不是 1.0）  
4. 是否做了 L2-norm（**应为否**）  
5. 无注入 / 正确注入 各一条生成样例  
6. golden 对齐到哪一步、哪一步开始偏离  
7. `hrt_model_exec` 或 remote verifier 的 cosine（若同网段 PC 可跑）

---

## 7. 相关 OSS / 本地文档

| 资源 | 路径 |
|------|------|
| 部署包 | `oss://gemma-model/gemma4_e2b/gemma4_e2b_deploy.tar.gz` |
| 仅 Vision | `oss://gemma-model/gemma4_e2b/gemma4-e2b_vit_ptq.hbm` |
| PC 交付文档 | `oss://gemma-model/gemma4_e2b/pc_delivery/GEMMA4_E2B_PC_DELIVERY_TO_S100.md` |
| golden mask/KV | `oss://gemma-model/gemma4_e2b/pc_delivery/golden_mask_kv.tar.gz` |
| 板端总交接 | `GEMMA4_E2B_BOARD_HANDOVER.md` |
| PC 全量交接 | `GEMMA4_E2B_CLAUDE_HANDOVER.md` |
| E2E 验证脚本 | `e2e_vlm_verify_fast.py` |
| E2E 验证结果 | `output/e2e_vlm_verify_fast.json` |

---

## 8. 一句话给板端 Agent

**Vision HBM 没问题（融合 logits cosine 0.9998）；请换 COCO 新版 Vision HBM，按 `masked_scatter` 原样注入 `[280,1536]`（禁止 L2-norm），PLE image 位用 pad embed，走 chat template，再用 golden_mask_kv 对齐 Text prefill。**
