"""
CSS样式定义 - 优化版本
"""

MAIN_CSS = """
<style>
    /* ========== CSS变量定义 ========== */
    :root {
        --primary-color: #667eea;
        --primary-dark: #5568d3;
        --secondary-color: #764ba2;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #f44336;
        --text-primary: #333333;
        --text-secondary: #666666;
        --bg-light: #f8f9fa;
        --bg-white: #ffffff;
        --border-color: #e0e0e0;
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.15);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --spacing-xs: 0.5rem;
        --spacing-sm: 1rem;
        --spacing-md: 1.5rem;
        --spacing-lg: 2rem;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ========== 主标题 ========== */
    .main-header {
        text-align: center;
        padding: var(--spacing-lg) 0;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border-radius: var(--radius-lg);
        margin-bottom: var(--spacing-lg);
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-header h1 {
        font-size: clamp(2rem, 5vw, 3rem);
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: clamp(1rem, 2vw, 1.2rem);
        margin: var(--spacing-xs) 0 0 0;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }

    /* ========== 功能卡片 ========== */
    .feature-card {
        background: var(--bg-white);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-color);
    }

    /* ========== 分析模式卡片 ========== */
    .analysis-mode-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        text-align: left;
        margin: var(--spacing-sm) 0;
        border: 2px solid transparent;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        height: 280px;  /* 固定高度，确保所有卡片大小一致 */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .analysis-mode-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .analysis-mode-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-4px) scale(1.01);
        box-shadow: var(--shadow-lg);
    }
    
    .analysis-mode-card:hover::after {
        transform: scaleX(1);
    }
    
    .analysis-mode-card h3 {
        color: var(--primary-color);
        margin-bottom: var(--spacing-sm);
        font-size: 1.3rem;
        font-weight: 600;
        flex-shrink: 0;  /* 标题不收缩，保持固定大小 */
        line-height: 1.4;
    }
    
    .analysis-mode-card p {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.7;
        flex-grow: 1;  /* 内容区域自动填充剩余空间 */
        overflow: hidden;  /* 防止内容溢出 */
        margin: 0;  /* 移除默认margin，确保布局一致 */
    }
    
    /* 确保Streamlit列容器高度一致 */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="column"] > div {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    /* ========== 进度容器 ========== */
    .progress-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        border-left: 4px solid var(--primary-color);
        box-shadow: var(--shadow-sm);
        margin: var(--spacing-sm) 0;
    }
    
    .progress-container h3 {
        color: var(--primary-color);
        margin-bottom: var(--spacing-sm);
        font-weight: 600;
    }

    /* ========== 结果容器 ========== */
    .result-container {
        background: var(--bg-white);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        margin: var(--spacing-sm) 0;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    
    .result-container:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }
    
    .result-container h1, 
    .result-container h2, 
    .result-container h3, 
    .result-container h4, 
    .result-container h5, 
    .result-container h6 {
        color: var(--text-primary);
        margin-top: var(--spacing-sm);
        margin-bottom: var(--spacing-xs);
    }
    
    .result-container p, 
    .result-container li, 
    .result-container div {
        color: var(--text-primary);
        line-height: 1.7;
    }
    
    .result-container code {
        background: var(--bg-light);
        padding: 0.2rem 0.4rem;
        border-radius: var(--radius-sm);
        font-size: 0.9em;
    }
    
    .result-container pre {
        background: var(--bg-light);
        padding: var(--spacing-sm);
        border-radius: var(--radius-sm);
        overflow-x: auto;
        border-left: 3px solid var(--primary-color);
    }

    /* ========== 按钮样式 ========== */
    .stButton > button {
        border-radius: 25px;
        font-weight: 600;
        transition: var(--transition);
        padding: 0.5rem 1.5rem;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* 主要按钮 */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
    }
    
    /* 次要按钮 */
    .stButton > button[kind="secondary"] {
        background: var(--bg-white);
        color: var(--primary-color);
        border: 2px solid var(--primary-color);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--primary-color);
        color: white;
    }

    /* ========== 成功提示 ========== */
    .upload-success {
        background: linear-gradient(135deg, var(--success-color), #45a049);
        color: white;
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        text-align: center;
        margin: var(--spacing-sm) 0;
        box-shadow: var(--shadow-md);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .upload-success h4 {
        margin: 0;
        font-size: 1.2rem;
    }
    
    .upload-success p {
        margin: var(--spacing-xs) 0 0 0;
        opacity: 0.95;
    }

    /* ========== 信息提示框 ========== */
    .info-box {
        background: #e3f2fd;
        padding: var(--spacing-sm);
        border-radius: var(--radius-sm);
        margin-bottom: var(--spacing-sm);
        border-left: 4px solid #1976d2;
        box-shadow: var(--shadow-sm);
    }
    
    .info-box p {
        margin: 0;
        color: #1976d2;
        line-height: 1.6;
    }

    /* ========== 响应式设计 ========== */
    /* 中等屏幕：2列布局 */
    @media (max-width: 1200px) and (min-width: 769px) {
        .analysis-mode-card {
            height: 300px;  /* 稍微增加高度以适应较小屏幕 */
            padding: var(--spacing-sm);
        }
        
        .analysis-mode-card h3 {
            font-size: 1.15rem;
        }
        
        .analysis-mode-card p {
            font-size: 0.9rem;
        }
    }
    
    /* 移动端：1列布局 */
    @media (max-width: 768px) {
        .main-header {
            padding: var(--spacing-md) 0;
            margin-bottom: var(--spacing-md);
        }
        
        .analysis-mode-card {
            height: auto;  /* 移动端允许自适应高度 */
            min-height: 250px;  /* 但保持最小高度一致 */
            padding: var(--spacing-sm);
        }
        
        .analysis-mode-card h3 {
            font-size: 1.1rem;
        }
        
        .analysis-mode-card p {
            font-size: 0.9rem;
        }
        
        .result-container {
            padding: var(--spacing-sm);
        }
        
        .stButton > button {
            width: 100%;
            margin-bottom: var(--spacing-xs);
        }
    }

    /* ========== 骨架屏加载动画 ========== */
    @keyframes skeleton-loading {
        0% {
            background-position: -200px 0;
        }
        100% {
            background-position: calc(200px + 100%) 0;
        }
    }
    
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200px 100%;
        animation: skeleton-loading 1.5s ease-in-out infinite;
        border-radius: var(--radius-sm);
    }

    /* ========== 滚动条样式 ========== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }

    /* ========== 输入框样式 ========== */
    .stTextInput > div > div > input {
        border-radius: var(--radius-sm);
        border: 2px solid var(--border-color);
        transition: var(--transition);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* ========== 展开器样式 ========== */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .streamlit-expanderContent {
        padding-top: var(--spacing-sm);
    }

    /* ========== 视频播放器容器 ========== */
    .video-container {
        background: #000;
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        margin: var(--spacing-sm) 0;
    }

    /* ========== 关键帧网格 ========== */
    .frame-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--spacing-sm);
        margin: var(--spacing-sm) 0;
    }
    
    .frame-item {
        border-radius: var(--radius-sm);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    
    .frame-item:hover {
        transform: scale(1.05);
        box-shadow: var(--shadow-md);
    }

    /* ========== 对话历史样式 ========== */
    .conversation-item {
        background: var(--bg-white);
        padding: var(--spacing-sm);
        border-radius: var(--radius-sm);
        margin-bottom: var(--spacing-sm);
        border-left: 4px solid var(--primary-color);
        box-shadow: var(--shadow-sm);
    }
    
    .conversation-item:hover {
        box-shadow: var(--shadow-md);
    }

    /* ========== GPT风格对话气泡样式 ========== */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-md);
        padding: var(--spacing-md);
        max-height: 600px;
        overflow-y: auto;
        background: var(--bg-light);
        border-radius: var(--radius-md);
        margin: var(--spacing-md) 0;
    }

    .message-wrapper {
        display: flex;
        flex-direction: column;
        margin-bottom: var(--spacing-md);
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .message-user {
        align-self: flex-end;
        max-width: 75%;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }

    .message-assistant {
        align-self: flex-start;
        max-width: 75%;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }

    .message-bubble {
        padding: var(--spacing-sm) var(--spacing-md);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        word-wrap: break-word;
        line-height: 1.6;
    }

    .message-bubble-user {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border-bottom-right-radius: 4px;
    }

    .message-bubble-assistant {
        background: var(--bg-white);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 4px;
    }

    .message-bubble-assistant > * {
        margin: 0;
    }

    .message-bubble-assistant .streamlit-expander {
        margin-top: 0.5rem;
    }

    .message-meta {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: var(--spacing-xs);
        padding: 0 var(--spacing-xs);
    }

    .message-timestamp {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: rgba(0, 0, 0, 0.05);
        padding: 0.2rem 0.5rem;
        border-radius: var(--radius-sm);
        font-weight: 500;
    }

    .message-timestamp-user {
        color: rgba(255, 255, 255, 0.9);
        background: rgba(255, 255, 255, 0.2);
    }

    .message-content {
        margin-top: var(--spacing-xs);
    }

    .message-content p {
        margin: 0.5rem 0;
    }

    .message-content p:first-child {
        margin-top: 0;
    }

    .message-content p:last-child {
        margin-bottom: 0;
    }

    .transcript-snippet {
        background: var(--bg-light);
        padding: var(--spacing-xs) var(--spacing-sm);
        border-radius: var(--radius-sm);
        margin-top: var(--spacing-xs);
        font-size: 0.85rem;
        color: var(--text-secondary);
        border-left: 3px solid var(--primary-color);
        max-height: 100px;
        overflow-y: auto;
    }

    .chat-input-container {
        background: var(--bg-white);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        position: sticky;
        bottom: 0;
        z-index: 10;
    }

    .empty-chat {
        text-align: center;
        padding: var(--spacing-lg);
        color: var(--text-secondary);
    }

    .empty-chat-icon {
        font-size: 3rem;
        margin-bottom: var(--spacing-sm);
        opacity: 0.5;
    }

    /* ========== 工具提示 ========== */
    [data-testid="stTooltip"] {
        font-size: 0.9rem;
    }

    /* ========== 可访问性改进 ========== */
    *:focus-visible {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* ========== 打印样式 ========== */
    @media print {
        .main-header {
            background: white;
            color: black;
        }
        
        .stButton {
            display: none;
        }
    }
</style>
"""
