# TeachingAssistant 模型训练与部署指引

面向“教学助理”类问答场景，从数据准备、训练、推理到部署（Ollama 与 vLLM）。示例以开源中文/多语 LLM 为主，可按需替换。

## 1. 数据集来源（开源）
- 通用指令/对话：OpenOrca、Alpaca-GPT4、Firefly（中文）、UltraChat、ShareGPT 清洗版。
- 教育/知识问答：CMRC 2018、DRCD、C3（中文阅读理解）、SQuAD、Natural Questions。
- 多轮问答与推理：MathInstruct、GSM8K（数学推导）、BoolQ、CommonsenseQA。
- 安全/守则：Self-Instruct Safety、OpenAI safety policy distill、Llama Guard 数据。
- 可选多模态（如需要视频帧/图片理解）：LLaVA-Instruct、ShareGPT4V（需遵守许可证）。

## 2. 业务数据采集与清洗
- 来源：课程讲义、课堂 QA 记录、教学视频转录（Whisper）、FAQ 文档、教案/作业解析。
- 去除隐私与敏感：脱敏人名/学号/手机号/邮箱；删除内部密级文件；按法规过滤违规内容。
- 结构化预处理：
  - 文本分块：按段落/章节/时间戳切分，控制 300–800 tokens，保留来源与时间戳。
  - 去噪：删除广告、推广、版权声明；标准化符号、全半角、空白。
  - 语义去重：SimHash / MinHash 去重，阈值 0.9 左右。
  - 质量过滤：长度过短（<5 词）、乱码、OCR 噪声过滤；语言检测确保中英文。
- 标注/合成：
  - 指令-输出对：用模板生成 “根据上下文回答”/“提取要点”/“总结”/“生成题目” 等指令。
  - 对抗样本：加入“无关问题”要求回答“不知道”或拒答，训练稳健性。

### 2.1 Tokenizer 处理细节
- **Special Tokens**：
    - **BOS/EOS**：对于 Llama-2/3 等模型，默认不自动添加 `<s>`，需在 Tokenizer 配置中显式设置 `add_bos_token=True`。若数据中已手动包含 `<s>`，则应设为 False 以免双重 BOS。
    - **Pad Token**：Llama 默认无 PAD token，需指定 `tokenizer.pad_token = tokenizer.eos_token` 或新增 `<pad>` 并扩充 embedding 层。
- **Chat Template 陷阱**：
    - 确保 `system` prompt 只有在模型支持时才使用。
    - 警惕 `\n` 处理：某些模板会在 role 后自动加换行，若数据中已有换行可能导致双换行，破坏语义连续性。
- **Padding 策略**：训练时通常使用 `padding_side="right"`（因 Causal LLM 预测下一个 token），但在推理时必须 `padding_side="left"`。
- **最大长度**：超过 `max_seq_len` 的数据需截断。建议保留对话的尾部（最新的 context）而非头部。
- **Masking**：在计算 Loss 时，**只计算 Output 部分的 Loss**，将 Input（指令/问题）部分的 Label 设为 -100（PyTorch CrossEntropyLoss 的 `ignore_index`），避免模型学习“如何提问”。

## 3. 训练方法细节
- 目标：SFT（监督微调）+ 可选 DPO/ORPO 进行偏好对齐。
- 模型选择：7B/8B（成本友好）或 13B（更强）。中文可选 Qwen/InternLM/Baichuan 开源权重，多语可选 LLaMA 系列。
- 数据格式：统一为 Alpaca/ShareGPT 风格 JSON，字段示例：
  ```json
  {"instruction": "根据给定讲义回答问题", "input": "讲义内容…", "output": "答案…", "meta": {"source": "course_A", "timestamp": "00:12:30"}}
  ```
- 训练超参（参考 7B）：
  - 设备：A100 80G x1 或 A6000 48G x2（使用 LoRA 可在 24G 卡上跑）。
  - 批次：global batch 128–256（梯度累积实现）；seq_len 2048；epochs 2–3；lr 1e-5～2e-5；cosine decay；warmup 3%。
  - LoRA：r=64, alpha=128, dropout=0.05，目标模块 q_proj/k_proj/v_proj/o_proj（或 mlp/down_proj），dtype bfloat16/fp16。
  - 混合精度：bf16 (BFloat16) 优先（动态范围大，训练更稳）；若显卡不支持（如 T4/V100）则用 fp16 + DeepSpeed/Scaler 避免溢出。
  - **加速与显存优化**：
    - **FlashAttention-2**：大幅降低显存占用并提升速度（需 Ampere 架构以上）。
    - **Gradient Checkpointing**：以“时间换空间”，显存占用可减半（约由 O(N) 降为 O(sqrt(N))），但训练速度慢 20-30%。
    - **DeepSpeed ZeRO**：
      - Stage 1: 切分 Optimizer States（显存节省 ~4x）。
      - Stage 2: 切分 Gradients（显存节省 ~8x）。
      - Stage 3: 切分 Parameters（多卡必备）。
      - Offload: 将状态卸载到 CPU 内存（速度慢，救急用）。
  - **损失函数**：标准 Causal LM Loss (Cross Entropy)。确保 Pad token 被 mask 掉。
  - 正则/稳定性：label smoothing 0.05, gradient clipping 1.0 (防止梯度爆炸).
- 偏好对齐（可选 DPO/ORPO）：
  - 收集正/负回答对（可用 GPT-4/4o 打分或规则生成）。
  - lr 5e-6～1e-5，batch 64，epochs 1–2；prompt 长度保持一致。
- 评测：构建业务评测集（事实问答、拒答、格式要求、时间戳引用），用 rouge / bleu / exact match + 人审。
### 3.1 常见问题与排查 (Troubleshooting)
- **Loss 不下降 / 震荡**：
  - 检查 LR 是否过大（建议 7B 模型 LoRA 起步 2e-4, 全量 1e-5）。
  - 检查数据质量：是否有大量重复、空值或错误标注。
  - 检查 Warmup：是否设置了足够的 warmup steps (如 total steps 的 3-5%)。
- **Loss 突刺 (Spike)**：
  - 数据中可能存在脏数据（极长文本、特殊符号）。
  - 尝试重启训练（resume from checkpoint）通常能跳过该不稳定点。
- **OOM (显存不足)**：
  - 显存占用公式估算：`Model Weights + Optimizer States + Gradients + Activations + Temp Buffer`。
  - 解决路径：减小 `per_device_train_batch_size` -> 开启 Component Gradient Checkpointing -> 开启 ZeRO Stage 2 -> 降低 LoRA Rank -> 使用 4-bit/8-bit 训练 (QLoRA)。
- **灾难性遗忘 (Catastrophic Forgetting)**：
  - 表现：微调后模型不会说“人话”或失去了原有通用能力。
  - 解决：降低 LR；混入通用数据（Replay Buffer）如 Alpaca/ShareGPT 数据一起训练；减少 Epochs。
- **过拟合 (Overfitting)**：
  - 表现：训练 Loss 很低但验证集 Loss 升高，复读机现象。
  - 解决：增加 Weight Decay (0.1)，增大 Dropout (0.05-0.1)，通过 Early Stopping 停止。

- **Loss 异常**：
  - **Loss = 0**：Label全是 -100（数据处理错误）或 Learning Rate 极大导致溢出。
  - **Loss = NaN**：梯度爆炸（检查 Gradient Clipping）、混合精度溢出（尝试 bf16）、或数据中有非法字符（如 NaN 浮点数）。
  - **Loss 不下降**：LR 过小或过大；数据 shuffle 不够导致 batch 相关性过高。
- **硬件/环境**：
  - **CUDA OOM**：显存碎片化严重，尝试 `torch.cuda.empty_cache()` 或设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`。
  - **NCCL Timeout**：多机多卡互联卡死，通常是防火墙或 DNS 问题，检查 `NCCL_IB_DISABLE=1` 或 `NCCL_P2P_DISABLE=1` 调试。
- **推理质量**：
  - **复读机/重复循环**：推理参数 `repetition_penalty` 设置过低（<1.0）或过高（>1.2 导致也不敢说话）；建议 1.05-1.1。
  - **输出为空/乱码**：EOS token ID 映射错误，模型预测了错误的停止符。

### 3.2 深度监控指标 (Advanced Metrics)
- **Grad Norm (梯度范数)**：
  - 正常应平稳波动。若突增（Spike），通常是遇到“脏数据”或 LR 调度刚进入未稳态。若长期极大，说明正则化不足或 LR 过高。
- **Token Accuracy**：
  - Next-token prediction 的准确率。训练初期应迅速从 ~0% 升至 40-60%（取决于词表大小和数据难度）。若长期停留在较低值，模型可能未收敛。
- **MFU (Model Flops Utilization)**：
  - 衡量计算效率。A100 上优化良好的训练应达到 50% 以上。低 MFU 意味着 IO 瓶颈或算子未优化。
## 4. 推理与量化
- 量化：训练完成后可导出 int4/int8（GPTQ/ AWQ）或 GGUF（用于 Ollama）；保持评测集对比量化前后质量。
- 序列长度：若需长文本（讲义/长视频），考虑 LongLoRA/Position Interpolation/NTK-aware scaling；或用 RAG 解决长上下文。
- vLLM 推理：
  - 安装：`pip install vllm==0.5.0`（按 CUDA 版本选择），启动示例：
    ```bash
    python -m vllm.entrypoints.openai.api_server \
      --model /path/to/your/model \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --max-model-len 4096 \
      --dtype auto \
      --gpu-memory-utilization 0.9
    ```
  - 调优：打开 PagedAttention；批量推理时增大 `--max-num-seqs`; 合理设置 `--swap-space` 避免 OOM。
  - 客户端调用（OpenAI 兼容）：
    ```python
    import openai
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    resp = client.chat.completions.create(
        model="teachingassistant",
        messages=[{"role": "user", "content": "解释牛顿第二定律"}],
        temperature=0.3,
    )
    print(resp.choices[0].message.content)
    ```

## 5. 与 Ollama 集成
- 准备 GGUF 权重：用 `llama.cpp`/`llamafactory` 导出 GGUF；将模型放入 `~/.ollama/models`。
- 创建 `Modelfile`（示例）：
  ```
  FROM your-teachingassistant.gguf
  PARAMETER temperature 0.3
  PARAMETER stop "Human:"
  PARAMETER stop "Assistant:"
  TEMPLATE """
  {{- if .System }}<s>[INST] <<SYS>>
  {{ .System }}
  <</SYS>>{{- end }}
  {{- range .Messages }}
  {{- if eq .Role "user" }}{{ .Content }}[/INST]{{ end }}
  {{- if eq .Role "assistant" }}{{ .Content }}</s>{{ end }}
  {{- end }}
  """
  ```
- 加载模型：`ollama create teachingassistant -f Modelfile`
- 调用：
  ```bash
  ollama run teachingassistant "解释能量守恒定律"
  ```
  或 Python：
  ```python
  import ollama
  resp = ollama.chat(
      model="teachingassistant",
      messages=[{"role": "user", "content": "讲一下勾股定理的应用"}],
      options={"temperature": 0.3}
  )
  print(resp["message"]["content"])
  ```

## 6. RAG 与业务化建议
- 数据索引：用教案/视频转录构建向量库（FAISS/Chroma），存储时间戳与章节。切片 300–500 字，重叠 50–80。
- 提示模板：明确“只基于给定上下文回答；无相关信息时说不知道；返回时间戳引用”。
- 过滤与审核：上线前加入敏感词/越权问题过滤；日志抽样人工复核。

## 7. GPU 与成本提示
- 训练建议：LoRA on 7B 可用 24G 显存；全参 7B 需约 48G~80G（取决于 Context Len 和 ZeRO 配置）；13B 全参推荐 80G A100/H800。
  - **显存估算经验值** (LoRA, seq_len=2048, batch=1):
    - 7B (bf16): ~16GB (无 Checkpointing) / ~10GB (有 Checkpointing)。
    - QLoRA (4-bit): ~6-8GB。
- 推理：vLLM + bf16 在 7B 上单卡 24G 可跑；多并发可用张量并行或多实例。Ollama+GGUF 支持 CPU/GPU 量化低成本推理。
- 监控：记录 GPU 利用率、吞吐、显存峰值；对比量化前后指标，保留回滚版本。

## 8. 最小落地路线（建议）
1) 选 7B 基座 + LoRA；用开源指令数据 + 清洗后的业务 QA 生成 SFT 数据。  
2) 训练 2–3 epoch，评测业务集；量化成 GGUF。  
3) 部署两条链路：vLLM（高吞吐在线）+ Ollama（本地/桌面）。  
4) 接入 RAG，覆盖长文本/视频；上线前增加拒答与安全评测。  
5) 持续数据闭环：收集用户提问/错误答案，定期再训练或增量 LoRA。 
