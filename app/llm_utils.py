
import os
from app.config import FRAME_DIR, MODEL  # 启用 vLLM 时请同时导入 VLLM_MODEL_PATH, VLLM_CONFIG
from app.prompts import prompt_dict
import base64

# ============================================================================
# Ollama 导入和初始化
# ============================================================================
ollama = None
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception as ollama_import_error:
    OLLAMA_AVAILABLE = False
    ollama_error = str(ollama_import_error)
    ollama = None  # 确保 ollama 变量存在，即使导入失败

# ============================================================================
# vLLM 相关代码已注释，改用 Ollama
# ============================================================================
# try:
#     from vllm import LLM, SamplingParams
#     VLLM_AVAILABLE = True
# except Exception as vllm_import_error:
#     VLLM_AVAILABLE = False
#     vllm_error = str(vllm_import_error)

import streamlit as st
from typing import List, Dict
from app.rag_utils import VideoRAGSystem


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def filter_promotional_content(text: str) -> str:
    """
    过滤掉推广内容、订阅链接、newsletter等信息
    精确匹配特定的推广内容，避免误删有用信息
    """
    import re
    
    if not text:
        return text
    
    # 定义需要过滤的精确模式（更具体的推广内容）
    promotional_patterns = [
        # 特定的网站和域名
        r'blog\.bybigo\.com',
        r'bybigo\.com',
        # newsletter相关
        r'subscribe.*newsletter',
        r'newsletter.*subscribe',
        r'system design newsletter',
        # 特定的推广文本
        r'If you like our videos.*we might like.*newsletter',
        r'trusted by.*\d+.*readers',
        r'subscribe to blog\.',
        r'subscribe to.*blog',
        # URL模式（但只过滤明显的推广链接）
        r'http[s]?://[^\s]*blog[^\s]*',
        r'http[s]?://[^\s]*newsletter[^\s]*',
        r'http[s]?://[^\s]*subscribe[^\s]*',
    ]
    
    # 按行分割文本
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            filtered_lines.append(line)  # 保留空行以维持格式
            continue
        
        # 检查是否包含推广内容
        is_promotional = False
        matched_pattern = None
        
        for pattern in promotional_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_promotional = True
                matched_pattern = pattern
                break
        
        if not is_promotional:
            # 不包含推广内容，保留该行
            filtered_lines.append(line)
        else:
            # 包含推广内容，尝试提取有用部分
            if matched_pattern:
                match = re.search(matched_pattern, line, re.IGNORECASE)
                if match:
                    # 保留推广内容之前的部分
                    useful_part = line[:match.start()].strip()
                    # 如果前面有有用内容（长度合理），保留它
                    if useful_part and len(useful_part) > 10:
                        filtered_lines.append(useful_part)
                    # 否则完全跳过这一行
    
    # 重新组合文本
    filtered_text = '\n'.join(filtered_lines)
    
    # 清理多余的空白行（保留单个空行，删除连续多个空行）
    filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
    
    # 移除文本末尾的推广内容
    # 检查最后几行是否包含推广内容
    final_lines = filtered_text.split('\n')
    while final_lines:
        last_line = final_lines[-1].strip().lower()
        # 如果最后一行包含推广关键词，移除它
        if any(re.search(pattern, last_line, re.IGNORECASE) for pattern in promotional_patterns):
            final_lines.pop()
        else:
            break
    
    filtered_text = '\n'.join(final_lines)
    
    return filtered_text.strip()


from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================================
# Ollama 模型初始化
# ============================================================================

def check_ollama_connection():
    """检查Ollama连接"""
    if not OLLAMA_AVAILABLE or ollama is None:
        error_detail = ollama_error if 'ollama_error' in locals() or 'ollama_error' in globals() else '未知错误'
        raise RuntimeError(
            f"❌ Ollama 未安装或导入失败: {error_detail}\n\n"
            "请安装 Ollama: pip install ollama\n"
            "并确保 Ollama 服务正在运行: ollama serve"
        )
    
    try:
        # 测试连接
        ollama.list()
        return True
    except Exception as e:
        raise RuntimeError(
            f"❌ 无法连接到 Ollama 服务: {e}\n\n"
            "请确保：\n"
            f"1. Ollama 服务正在运行: ollama serve\n"
            f"2. 模型已下载: ollama pull {MODEL}\n"
            "3. 检查网络连接"
        )

# ============================================================================
# vLLM 模型初始化（已注释，改用 Ollama）
# ============================================================================
# @st.cache_resource
# def get_vllm_model():
#     """
#     获取并缓存 vLLM 模型实例
#     支持 Flash Attention 加速
#
#     多卡场景提示：
#     - tensor_parallel_size 建议设为 GPU 数量（例如 torch.cuda.device_count()）
#     - 单卡时保持 1，多卡时按需调整 max_model_len / max_num_seqs
#     - 多卡下每卡的 gpu_memory_utilization 需更保守，避免 profile_run OOM
#     - 多模态模型需要更多显存处理图像输入，建议先用小分辨率图像验证
#     """
#     if not VLLM_AVAILABLE:
#         raise RuntimeError(
#             f"❌ vLLM 未安装或导入失败: {vllm_error}\n\n"
#             "请安装 vLLM: pip install vllm\n"
#             "注意：vLLM 需要 CUDA 和 GPU 支持"
#         )
#     
#     try:
#         # 检查 CUDA 是否可用
#         import torch
#         if not torch.cuda.is_available():
#             raise RuntimeError(
#                 "❌ CUDA 不可用。vLLM 需要 NVIDIA GPU 和 CUDA 支持。\n"
#                 "请确保：\n"
#                 "1. 已安装 NVIDIA GPU 驱动\n"
#                 "2. 已安装 CUDA toolkit\n"
#                 "3. PyTorch 支持 CUDA"
#             )
#         
#         # 设置 PyTorch CUDA 内存分配器配置（避免内存碎片）
#         # 这有助于减少 OOM 错误，特别是在处理多模态输入时
#         if not os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
#             os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#         
#         # 设置 HuggingFace 镜像源（解决网络访问问题）
#         # 如果环境变量未设置，使用国内镜像源
#         if not os.getenv("HF_ENDPOINT"):
#             os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#         
#         # 如果设置了 HF_TOKEN，使用它
#         hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         if hf_token:
#             os.environ["HF_TOKEN"] = hf_token
#             os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
#         
#         # 从配置获取模型路径和参数
#         model_path = VLLM_MODEL_PATH
#         if not model_path:
#             raise RuntimeError("未设置 VLLM_MODEL_PATH，请在 app/config.py 中配置")
#         if os.path.sep in model_path and not os.path.exists(model_path):
#             raise RuntimeError(f"本地模型路径不存在: {model_path}")
#
#         config = VLLM_CONFIG.copy()
#         
#         # 多卡场景建议（可选）
#         # 如果未在 VLLM_CONFIG 中显式设置，可按 GPU 数量覆盖
#         # gpu_count = torch.cuda.device_count()
#         # if gpu_count > 1 and not config.get("tensor_parallel_size"):
#         #     config["tensor_parallel_size"] = gpu_count
#         
#         st.info(f"🚀 正在加载 vLLM 模型: {model_path}\n"
#                 f"配置: GPU利用率={config.get('gpu_memory_utilization', 0.85)}, "
#                 f"最大长度={config.get('max_model_len', 8192)}\n"
#                 f"HuggingFace 镜像: {os.getenv('HF_ENDPOINT', '默认')}")
#         
#         # 创建 vLLM 实例
#         llm = LLM(
#             model=model_path,
#             trust_remote_code=True,
#             **config
#         )
#         
#         st.success("✅ vLLM 模型加载成功！Flash Attention 已启用")
#         return llm
#         
#     except Exception as e:
#         error_msg = str(e)
#         
#         # 提供更详细的错误信息和解决方案
#         error_lower = error_msg.lower()
#         solutions = []
#         
#         if "not a local folder" in error_lower or "valid model identifier" in error_lower:
#             solutions.append("1. **模型路径问题**：")
#             solutions.append("   - 如果使用 HuggingFace 模型 ID，确保网络可以访问 huggingface.co")
#             solutions.append("   - 或者先手动下载模型到本地，然后使用本地路径")
#         
#         if "cuda" in error_lower or "gpu" in error_lower:
#             solutions.append("2. **GPU/CUDA 问题**：")
#             solutions.append("   - 运行 `nvidia-smi` 检查 GPU 是否可用")
#             solutions.append("   - 检查 CUDA 版本：`nvcc --version`")
#         
#         if "memory" in error_lower or "out of memory" in error_lower:
#             solutions.append("3. **显存不足**：")
#             solutions.append("   - 降低 max_model_len（如 2048 或 4096）")
#             solutions.append("   - 降低 gpu_memory_utilization（如 0.40 或 0.50）")
#             solutions.append("   - 检查是否有其他进程占用 GPU 显存：`nvidia-smi`")
#             solutions.append("   - 清理 GPU 显存：`kill -9 <占用显存的进程PID>`")
#             solutions.append("   - 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 来减少内存碎片")
#         
#         if not solutions:
#             solutions.append("1. 检查模型路径是否正确")
#             solutions.append("2. 检查网络连接（如果使用 HuggingFace 模型）")
#         
#         st.error(
#             f"❌ vLLM 模型加载失败: {error_msg}\n\n"
#             "**解决方案：**\n" + "\n".join(solutions)
#         )
#         raise


@st.cache_resource
def get_embedding_model():
    """获取并缓存Embedding模型"""
    import os
    
    # 设置 HuggingFace 镜像源（解决401错误）
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return embedding_model
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "Unauthorized" in error_str:
            st.warning("⚠️ 模型下载遇到认证问题，尝试其他方法...")
            try:
                os.environ["HF_ENDPOINT"] = "https://huggingface.co"
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                return embedding_model
            except Exception as e2:
                st.error(
                    f"❌ 无法下载模型: {e2}\n\n"
                    "解决方案：\n"
                    "1. 设置 HuggingFace token: export HF_TOKEN=your_token\n"
                    "2. 或者使用本地已下载的模型\n"
                )
                raise
        else:
            raise


def get_rag_system():
    """获取RAG系统实例"""
    embedding_model = get_embedding_model()
    return VideoRAGSystem(embedding_model)


def contextualize_query(query: str, history: List[Dict]) -> str:
    """
    根据对话历史重写查询，使其独立化
    使用 Ollama 进行查询重写
    """
    if not history:
        return query
        
    if not OLLAMA_AVAILABLE:
        return query

    # 构建重写提示词
    conversation_str = ""
    for msg in history[-4:]:  # 只看最近几轮
        role = msg.get("role")
        content = msg.get("content")
        conversation_str += f"{role}: {content}\n"
    
    prompt = f"""Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{conversation_str}

User Question: {query}

Standalone Question:"""

    try:
        # 使用 Ollama 进行查询重写
        if ollama is None:
            return query
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        rewritten_query = response["message"]["content"].strip()
        
        # 简单的验证：如果返回太长或者是废话，回退到原问题
        if len(rewritten_query) > len(query) * 2 and len(query) > 10:
             return query
        return rewritten_query
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return query


def get_response(text, question, full_transcript, prompt_key, summarized_transcript, 
                 segments=None, use_rag=True, frame_paths=None):
    """
    获取LLM响应，支持RAG增强
    使用 Ollama 进行回答
    """
    # 检查 Ollama 是否可用
    if not OLLAMA_AVAILABLE or ollama is None:
        error_detail = ollama_error if 'ollama_error' in locals() or 'ollama_error' in globals() else '未知错误'
        raise RuntimeError(
            f"❌ Ollama 未安装或导入失败: {error_detail}\n\n"
            "请安装 Ollama: pip install ollama\n"
            "并确保 Ollama 服务正在运行: ollama serve"
        )
    
    # 检查 Ollama 连接
    try:
        check_ollama_connection()
    except Exception as e:
        raise RuntimeError(str(e))

    prompt = prompt_dict[prompt_key]

    # 准备对话历史用于查询重写
    history_messages = []
    if "qa_conversation_history" in st.session_state and st.session_state.qa_conversation_history:
        for item in st.session_state.qa_conversation_history[-6:]:
            if item.get("question"):
                history_messages.append({"role": "user", "content": item.get("question")})
            if item.get("answer"):
                history_messages.append({"role": "assistant", "content": item.get("answer")})

    # RAG检索相关上下文
    retrieved_contexts = []
    if use_rag and question and segments:
        try:
            if "rag_system" not in st.session_state:
                st.session_state.rag_system = get_rag_system()
            rag_system = st.session_state.rag_system
            
            if rag_system.vectorstore is None:
                with st.spinner("🔍 构建向量索引..."):
                    rag_system.build_vector_store(segments)
            
            search_query = question
            if history_messages:
                search_query = contextualize_query(question, history_messages)
            
            retrieved_contexts = rag_system.retrieve_relevant_context(
                search_query, 
                top_k=3
            )
        except Exception as e:
            st.warning(f"⚠️ RAG检索失败，使用默认上下文: {e}")
            retrieved_contexts = []

    prompt_inputs = []
    image_sources = frame_paths if frame_paths is not None else [
        os.path.join(FRAME_DIR, path)
        for path in os.listdir(FRAME_DIR)
        if path.endswith(".jpg")
    ]
    for image_path in image_sources:
        if os.path.exists(image_path):
            prompt_inputs.append(image_to_base64(image_path))

    # 构建增强的上下文（如果使用RAG）
    if prompt_key == "video_qa":
        if retrieved_contexts and use_rag:
            retrieved_info = "\n\n".join([
                f"[时间戳: {ctx['timestamp']}] {ctx['text']}"
                for ctx in retrieved_contexts
            ])
            enhanced_context = f"{summarized_transcript}\n\n=== 相关片段（基于语义检索） ===\n{retrieved_info}"
            
            if "video_qa_rag" in prompt_dict:
                prompt = prompt_dict["video_qa_rag"].format(
                    text=text, 
                    question=question, 
                    global_context=summarized_transcript,
                    retrieved_contexts=retrieved_info
                )
            else:
                prompt = prompt.format(
                    text=text, 
                    question=question, 
                    global_context=enhanced_context
                )
        else:
            prompt = prompt.format(text=text, question=question, global_context=summarized_transcript)
    elif prompt_key == "bullet_points":
        prompt = prompt.format(text=full_transcript)
    elif prompt_key == "qa_style":
        prompt = prompt.format(text=full_transcript)
    elif prompt_key == "video_summary":
        prompt = prompt.format(text=full_transcript)

    # 构建多轮对话消息
    messages = []
    
    conversation_pairs = []
    try:
        if "qa_conversation_history" in st.session_state and st.session_state.qa_conversation_history:
            for item in st.session_state.qa_conversation_history[-6:]:
                conversation_pairs.append((item.get("question", ""), item.get("answer", "")))
    except Exception:
        conversation_pairs = []

    # 添加对话历史
    if conversation_pairs:
        for (prev_q, prev_a) in conversation_pairs:
            if prev_q:
                messages.append({"role": "user", "content": prev_q})
            if prev_a:
                messages.append({"role": "assistant", "content": prev_a})
    
    # 构建当前消息
    # 注意：Ollama 支持图像输入，但需要模型支持多模态
    # 如果 prompt_inputs 存在，可以添加到消息中
    current_prompt = prompt
    if prompt_inputs:
        # 如果有图像，尝试使用多模态（如果模型支持）
        # 否则在提示词中说明有图像信息
        current_prompt += f"\n\n[注意：本次查询包含 {len(prompt_inputs)} 个视频帧图像。请基于视频帧的文本描述和转录内容进行综合分析。]"
    
    messages.append({
        "role": "user",
        "content": current_prompt
    })

    full_answer = ""
    placeholder = None
    
    try:
        placeholder = st.empty()
    except Exception:
        pass

    def safe_update_placeholder(content, show_cursor=False):
        """安全地更新占位符内容"""
        if placeholder is not None:
            try:
                if show_cursor:
                    placeholder.markdown(content + "▌")
                else:
                    placeholder.markdown(content)
            except Exception:
                st.markdown(content)

    try:
        # 使用 Ollama 进行推理
        if ollama is None:
            raise RuntimeError("Ollama 模块未正确导入")
        
        # 尝试流式输出
        try:
            stream = ollama.chat(
                model=MODEL,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                try:
                    if chunk.get("message") and chunk["message"].get("content"):
                        new_text = chunk["message"]["content"]
                        full_answer += new_text
                        safe_update_placeholder(full_answer, show_cursor=True)
                except Exception:
                    continue

            full_answer = filter_promotional_content(full_answer)
            safe_update_placeholder(full_answer, show_cursor=False)
        except Exception as stream_error:
            # 如果流式输出失败，使用非流式模式
            st.warning(f"⚠️ 流式输出失败，使用非流式模式: {stream_error}")
            response = ollama.chat(
                model=MODEL,
                messages=messages
            )
            full_answer = response["message"]["content"]
            full_answer = filter_promotional_content(full_answer)
            safe_update_placeholder(full_answer, show_cursor=False)
        
    except Exception as e:
        error_msg = str(e)
        st.error(
            f"❌ Ollama 推理失败: {error_msg}\n\n"
            "请检查：\n"
            f"1. Ollama 服务是否正在运行: ollama serve\n"
            f"2. 模型是否已下载: ollama pull {MODEL}\n"
            "3. 网络连接是否正常\n"
        )
        
        try:
            if placeholder is not None:
                placeholder.error(f"❌ 生成失败: {error_msg}")
            else:
                st.error(f"❌ 生成失败: {error_msg}")
        except Exception:
            st.error(f"❌ 生成失败: {error_msg}")
    
    return full_answer
