#!/bin/bash
# 启动脚本：设置 HuggingFace 镜像源并运行 Streamlit

# 激活虚拟环境
source venv/bin/activate

# 设置 HuggingFace 镜像源（解决401错误）
export HF_ENDPOINT=https://hf-mirror.com

# 如果需要使用 token，取消下面的注释并设置你的 token
# export HF_TOKEN=your_huggingface_token_here

# 运行 Streamlit
streamlit run app/main.py

