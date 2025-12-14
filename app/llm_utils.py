import os
from config import FRAME_DIR, prompt_dict, MODEL
import base64
import time
try:
    import ollama
except Exception as ollama_import_error:
    # 这里不直接抛异常，延迟到调用处给出更友好的提示
    ollama = None
import streamlit as st
from typing import List, Dict, Optional
from rag_utils import VideoRAGSystem


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


@st.cache_resource
def get_embedding_model():
    """获取并缓存Embedding模型"""
    import os
    
    # 设置 HuggingFace 镜像源（解决401错误）
    # 如果环境变量未设置，使用国内镜像源
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 如果设置了 HF_TOKEN，使用它
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    try:
        # 尝试加载模型
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return embedding_model
    except Exception as e:
        error_str = str(e)
        # 如果是401错误，尝试其他方法
        if "401" in error_str or "Unauthorized" in error_str:
            st.warning("⚠️ 模型下载遇到认证问题，尝试其他方法...")
            # 方法1: 尝试使用不同的镜像源
            try:
                os.environ["HF_ENDPOINT"] = "https://huggingface.co"
                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                return embedding_model
            except Exception as e2:
                # 方法2: 如果仍然失败，提示用户设置token或使用本地模型
                st.error(
                    f"❌ 无法下载模型: {e2}\n\n"
                    "解决方案：\n"
                    "1. 设置 HuggingFace token: export HF_TOKEN=your_token\n"
                    "2. 或者使用本地已下载的模型\n"
                    "3. 检查网络连接"
                )
                raise
        else:
            # 其他错误直接抛出
            raise


def get_rag_system():
    """获取RAG系统实例"""
    # 不再缓存整个RAG系统，因为vectorstore是状态相关的
    embedding_model = get_embedding_model()
    return VideoRAGSystem(embedding_model)


def contextualize_query(query: str, history: List[Dict]) -> str:
    """
    根据对话历史重写查询，使其独立化
    :param query: 当前用户的查询
    :param history: 对话历史 [{"role": "user", "content": "..."}, ...]
    :return: 重写后的独立查询
    """
    if not history:
        return query
        
    if ollama is None:
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
        response = call_ollama_with_retry(
            ollama.generate,
            model=MODEL,
            prompt=prompt,
            options={"num_predict": 100}
        )
        rewritten_query = response["response"].strip()
        # 简单的验证：如果返回太长或者是废话，回退到原问题
        if len(rewritten_query) > len(query) * 2 and len(query) > 10:
             return query
        st.write(f"DEBUG: Rewritten query: {rewritten_query}") 
        return rewritten_query
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return query


def check_ollama_connection(max_retries=2, retry_delay=1):
    """
    检查 Ollama 服务是否可用
    通过尝试一个简单的生成请求来测试连接
    :param max_retries: 最大重试次数
    :param retry_delay: 重试延迟（秒）
    :return: True if connection is OK, False otherwise
    """
    if ollama is None:
        return False
    
    for attempt in range(max_retries):
        try:
            # 尝试一个简单的生成请求来检查连接
            # 使用一个非常短的 prompt 来快速测试
            result = ollama.generate(
                model=MODEL,
                prompt="test",
                options={"num_predict": 1}  # 只生成1个token，快速测试
            )
            return True
        except Exception as e:
            error_str = str(e).lower()
            # 如果是模型不存在错误，说明连接是正常的，只是模型问题
            if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
                # 连接正常，但模型不存在
                return True  # 返回 True，让后续的错误处理更清晰
            # 如果是连接错误，重试
            if any(keyword in error_str for keyword in ['eof', 'connection', 'timeout', 'refused', 'status code: -1']):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return False
            else:
                # 其他错误，可能是模型问题，但连接是正常的
                return True
    return False


def call_ollama_with_retry(func, *args, max_retries=3, retry_delay=2, **kwargs):
    """
    带重试机制的 Ollama API 调用
    :param func: 要调用的函数（ollama.chat 或 ollama.generate）
    :param max_retries: 最大重试次数
    :param retry_delay: 重试延迟（秒）
    :return: 函数返回值
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # 检查是否是连接错误
            if any(keyword in error_str for keyword in ['eof', 'connection', 'timeout', 'refused', 'status code: -1']):
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Ollama 连接失败 (尝试 {attempt + 1}/{max_retries})，{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    # 清除连接检查缓存，强制重新检查
                    if "ollama_connection_checked" in st.session_state:
                        del st.session_state["ollama_connection_checked"]
                else:
                    # 清除连接检查缓存
                    if "ollama_connection_checked" in st.session_state:
                        del st.session_state["ollama_connection_checked"]
                    raise RuntimeError(f"Ollama 连接失败，已重试 {max_retries} 次: {e}")
            else:
                # 非连接错误，直接抛出
                raise
    
    if last_error:
        raise last_error


def get_response(text, question, full_transcript, prompt_key, summarized_transcript, 
                 segments=None, use_rag=True, frame_paths=None):
    """
    获取LLM响应，支持RAG增强
    :param text: 局部转录文本
    :param question: 问题
    :param full_transcript: 完整转录
    :param prompt_key: 提示词键
    :param summarized_transcript: 摘要转录
    :param segments: Whisper片段（用于RAG）
    :param use_rag: 是否使用RAG
    :param frame_paths: 本次调用需要附带的帧路径（None时回退到读取目录下的所有jpg）
    """
    # 检查 ollama 是否可用
    if ollama is None:
        raise RuntimeError(
            "未找到 Python 包 'ollama' 或导入失败：请先执行 `pip install ollama`，"
            "并确保本机已安装并启动 Ollama 服务（macOS 安装 App 或命令行安装），"
            "且已拉取模型 gemma3:4b（命令：`ollama pull gemma3:4b`）。"
        )
    
    # 检查 Ollama 服务连接（使用缓存，避免每次都检查）
    connection_check_key = "ollama_connection_checked"
    if connection_check_key not in st.session_state:
        # 只在第一次调用时检查连接
        if not check_ollama_connection():
            st.session_state[connection_check_key] = False
            raise RuntimeError(
                "无法连接到 Ollama 服务。请确保：\n"
                "1. Ollama 服务正在运行（检查：`ollama list`）\n"
                "2. 模型已下载（检查：`ollama list`，确保看到 gemma3:4b）\n"
                "3. 如果使用自定义端口，请检查环境变量 OLLAMA_HOST"
            )
        else:
            st.session_state[connection_check_key] = True

    prompt = prompt_dict[prompt_key]

    # 准备对话历史用于查询重写
    history_messages = []
    if "qa_conversation_history" in st.session_state and st.session_state.qa_conversation_history:
        for item in st.session_state.qa_conversation_history[-6:]:
            if item.get("question"):
                history_messages.append({"role": "user", "content": item.get("question")})
            if item.get("answer"):
                history_messages.append({"role": "assistant", "content": item.get("answer")})
    elif "conversation" in st.session_state and st.session_state.conversation:
        # 兼容旧格式
        for q, a in st.session_state.conversation[-6:]:
            if q: history_messages.append({"role": "user", "content": q})
            if a: history_messages.append({"role": "assistant", "content": a})

    # RAG检索相关上下文
    retrieved_contexts = []
    if use_rag and question and segments:
        try:
            # 尝试从session_state获取rag_system，避免每次重新创建
            if "rag_system" not in st.session_state:
                st.session_state.rag_system = get_rag_system()
            rag_system = st.session_state.rag_system
            
            # 如果向量存储不存在，构建它
            if rag_system.vectorstore is None:
                with st.spinner("🔍 构建向量索引..."):
                    rag_system.build_vector_store(segments)
            
            # 多轮对话查询重写
            search_query = question
            if history_messages:
                search_query = contextualize_query(question, history_messages)
            
            # 检索相关上下文
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
            # 合并检索到的上下文
            retrieved_info = "\n\n".join([
                f"[时间戳: {ctx['timestamp']}] {ctx['text']}"
                for ctx in retrieved_contexts
            ])
            enhanced_context = f"{summarized_transcript}\n\n=== 相关片段（基于语义检索） ===\n{retrieved_info}"
            
            # 使用RAG增强的提示词
            if "video_qa_rag" in prompt_dict:
                prompt = prompt_dict["video_qa_rag"].format(
                    text=text, 
                    question=question, 
                    global_context=summarized_transcript,
                    retrieved_contexts=retrieved_info
                )
            else:
                # 回退到原始提示词，但使用增强的上下文
                prompt = prompt.format(
                    text=text, 
                    question=question, 
                    global_context=enhanced_context
                )
        else:
            prompt = prompt.format(text=text, question=question, global_context=summarized_transcript)
    elif prompt_key == "bullet_points":
        # Use the full transcript for bullet points
        prompt = prompt.format(text=full_transcript)
    elif prompt_key == "qa_style":
        # Use the full transcript for question-answer pairs
        prompt = prompt.format(text=full_transcript)
    elif prompt_key == "video_summary":
        # Use the full transcript for comprehensive video analysis
        prompt = prompt.format(text=full_transcript)

    # Build multi-turn messages from session conversation when available
    messages = []
    try:
        conversation_pairs = []
        if "qa_conversation_history" in st.session_state and st.session_state.qa_conversation_history:
            for item in st.session_state.qa_conversation_history[-6:]:
                conversation_pairs.append((item.get("question", ""), item.get("answer", "")))
        elif "conversation" in st.session_state and st.session_state.conversation:
            conversation_pairs = st.session_state.conversation[-6:]

        if conversation_pairs:
            # Provide system guidance for consistent behavior
            messages.append({
                "role": "system",
                "content": "You are a helpful teaching assistant. Answer concisely and factually using provided video context. Do NOT include any promotional content, subscription links, newsletter mentions, website URLs (such as blog.bybigo.com), or advertising in your responses. Only provide answers based on the video content."
            })
            for (prev_q, prev_a) in conversation_pairs:
                if prev_q:
                    messages.append({"role": "user", "content": prev_q})
                if prev_a:
                    messages.append({"role": "assistant", "content": prev_a})
        # Current turn prompt with images
        messages.append({"role": "user", "content": prompt, 'images': prompt_inputs})
    except Exception:
        messages = [{"role": "user", "content": prompt, 'images': prompt_inputs}]

    full_answer = ""
    placeholder = None
    
    # 安全地创建占位符
    try:
        placeholder = st.empty()
    except Exception:
        pass  # 如果创建占位符失败，后续直接输出

    def safe_update_placeholder(content, show_cursor=False):
        """安全地更新占位符内容"""
        if placeholder is not None:
            try:
                if show_cursor:
                    placeholder.markdown(content + "▌")
                else:
                    placeholder.markdown(content)
            except Exception:
                # 如果占位符操作失败，直接输出到页面
                st.markdown(content)

    try:
        # 使用重试机制调用 ollama.chat
        response = call_ollama_with_retry(
            ollama.chat,
            model=MODEL,
            messages=messages,
            stream=True,
            max_retries=3,
            retry_delay=2
        )

        # 流式输出
        for chunk in response:
            try:
                new_text = chunk["message"]["content"]
                full_answer += new_text
                safe_update_placeholder(full_answer, show_cursor=True)
            except Exception as chunk_error:
                # 如果某个chunk处理失败，继续处理下一个
                continue

        # 最终输出（不显示光标）
        # 过滤推广内容
        full_answer = filter_promotional_content(full_answer)
        safe_update_placeholder(full_answer, show_cursor=False)
        
    except Exception as e:
        error_msg = str(e)
        # 检查是否是连接错误
        if any(keyword in error_msg.lower() for keyword in ['eof', 'connection', 'timeout', 'refused', 'status code: -1']):
            st.error(f"❌ Ollama 连接错误: {error_msg}\n\n"
                    f"请检查：\n"
                    f"1. Ollama 服务是否正在运行（运行 `ollama list` 测试）\n"
                    f"2. 模型 {MODEL} 是否已下载（运行 `ollama pull {MODEL}`）\n"
                    f"3. 网络连接是否正常")
        else:
            st.warning(f"⚠️ Streaming failed. Falling back to generate(). Reason: {error_msg}")
        
        # 尝试使用 generate() 作为降级方案
        try:
            result = call_ollama_with_retry(
                ollama.generate,
                model=MODEL,
                prompt=prompt,
                images=prompt_inputs or [],
                max_retries=2,
                retry_delay=2
            )
            full_answer = result["response"]
            # 过滤推广内容
            full_answer = filter_promotional_content(full_answer)
            safe_update_placeholder(full_answer, show_cursor=False)
        except Exception as gen_error:
            full_answer = f"❌ 生成失败: {gen_error}\n\n"
            full_answer += "请尝试：\n"
            full_answer += f"1. 检查 Ollama 服务：运行 `ollama list`\n"
            full_answer += f"2. 重启 Ollama 服务\n"
            full_answer += f"3. 检查模型是否存在：运行 `ollama pull {MODEL}`\n"
            full_answer += f"4. 查看 Ollama 日志以获取更多信息"
            
            # 如果占位符失败，直接使用 st.error
            try:
                if placeholder is not None:
                    placeholder.error(full_answer)
                else:
                    st.error(full_answer)
            except Exception:
                st.error(full_answer)
    
    return full_answer
