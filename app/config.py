import os

VIDEO_PATH = "runtime/uploaded_video.mp4"
FRAME_DIR = "runtime/frames/"
MODEL = "qwen3:8b"  # Ollama 模型名称
# MODEL = "gemma3:4b"  # 备选模型
os.makedirs(FRAME_DIR, exist_ok=True)

# ============================================================================
# vLLM 配置（已注释，改用 Ollama）
# ============================================================================

# vLLM 模型路径（支持 HuggingFace 模型 ID 或本地路径）
#
# 关于使用 ollama 已下载的 qwen3:8b：
# - ollama 的 qwen3:8b 与 HuggingFace 的 Qwen/Qwen2.5-8B-Instruct 对应
# - vLLM 需要 HuggingFace 格式，不能直接复用 ollama 本地格式
# - 解决方案：使用 HuggingFace 模型 ID（首次会自动下载）
#   或者手动下载 HuggingFace 格式到本地并指定本地路径
#
# 选项 1：使用 HuggingFace 模型 ID（推荐）
# 当前建议：Qwen2.5-VL-3B-Instruct（视觉语言模型，支持图像输入）
# 原 7B 模型已注释，改用更小的 3B 模型以降低显存占用
# VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
#
# 选项 2：使用本地 HuggingFace 格式模型（已手动下载）
# VLLM_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
# 原 7B 模型路径（已注释）：
# VLLM_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct"
# VLLM_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
# 或者
# VLLM_MODEL_PATH = "/path/to/your/local/qwen/model"
#
# 下载配置（可选）：
# - 设置 HuggingFace 镜像：export HF_ENDPOINT="https://hf-mirror.com"
# - 设置 HuggingFace Token：export HF_TOKEN="your_token"
#
# 其他可选模型（已注释）：
# - "Qwen/Qwen2.5-8B-Instruct" (纯文本模型，不支持图像)
# - "Qwen/Qwen2-VL-7B-Instruct" (旧版视觉语言模型)

# vLLM 配置参数
# 针对 RTX 4090D (24GB) 优化，适配 Qwen2.5-VL-3B-Instruct（视觉语言模型）
# 注意：trust_remote_code 在 llm_utils.py 中显式传递，不在此配置中
# 显存优化：VL 模型在 profile_run 阶段需要额外显存处理图像/视频输入
# VLLM_CONFIG = {
#     "max_model_len": 4096,  # 最大序列长度（降低以减少显存占用）
#     "gpu_memory_utilization": 0.50,  # GPU 显存利用率（如仍 OOM，可降到 0.30-0.40）
#     "tensor_parallel_size": 1,  # 张量并行大小（单 GPU 为 1）
#     "dtype": "half",  # 数据类型：half (FP16), float16, bfloat16, int8, int4
#     "max_num_seqs": 32,  # 最大并发序列数（降低以减少显存压力，特别是VL模型）
#     # "trust_remote_code": True,  # 已在 llm_utils.py 中显式传递，避免重复
#     # Flash Attention 相关（vLLM 默认启用，无需额外配置）
#     # "enable_prefix_caching": True,  # 启用前缀缓存（可选）
#     # "quantization": None,  # 量化方式：None, "awq", "gptq", "squeezellm"
#     # "enforce_eager": True,  # 某些环境下可提高稳定性，但会降低吞吐
#     # 注意：Qwen2.5-VL 是多模态模型，支持图像和视频输入
#     # 如果仍然遇到显存问题，可以进一步降低 max_model_len 到 2048 或 gpu_memory_utilization 到 0.30
# }
