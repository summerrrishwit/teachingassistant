#!/bin/bash
# 启动脚本：设置 HuggingFace 镜像源并运行 Streamlit

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d "venv/Scripts" ]; then
    source venv/Scripts/activate
fi

# 设置 HuggingFace 镜像源（解决401错误）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 PYTHONPATH，确保可以导入 app 模块
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 如果需要使用 token，取消下面的注释并设置你的 token
# export HF_TOKEN=your_huggingface_token_here

# 运行 Streamlit（从项目根目录运行）
streamlit run app/main.py




