# 👨‍💻 开发指南

本文档包含开发相关的所有信息，包括代码优化、重构、Bug修复等。

## 目录

1. [代码重构](#代码重构)
2. [代码优化建议](#代码优化建议)
3. [Bug修复记录](#bug修复记录)
4. [快速开始优化](#快速开始优化)

---

## 代码重构

### 重构概述

项目进行了代码结构优化，整合了多个小文件，提高了代码的可维护性和可读性。

### 重构内容

#### 1. 创建 `core.py` - 整合基础工具模块

**整合的文件**：
- ✅ `exceptions.py` → `core.py` (异常类)
- ✅ `constants.py` → `core.py` (常量定义)
- ✅ `decorators.py` → `core.py` (装饰器)
- ✅ `logger.py` → `core.py` (日志系统)
- ✅ `validators.py` → `core.py` (验证器)
- ✅ `singleton_class.py` → `core.py` (单例模式)

**优势**：
- 所有基础工具集中在一个文件，便于管理
- 减少文件数量，降低维护成本
- 统一的导入路径：`from core import ...`

#### 2. 文件重命名

- ✅ `ui_sections.py` → `ui.py` (更简洁的名称)

#### 3. 创建 `__init__.py`

新增 `__init__.py` 文件，统一导出常用内容：
```python
from app import VideoConstants, validate_video_file, setup_logger
```

### 文件结构对比

**重构前** (15个文件)
```
app/
├── __init__.py (空)
├── config.py
├── constants.py
├── decorators.py
├── exceptions.py
├── llm_utils.py
├── logger.py
├── main.py
├── rag_utils.py
├── singleton_class.py
├── transcript_utils.py
├── ui_sections.py
├── validators.py
├── video_utils.py
└── workflows.py
```

**重构后** (10个文件)
```
app/
├── __init__.py (导出常用内容)
├── config.py
├── core.py (所有基础工具)
├── llm_utils.py
├── main.py
├── rag_utils.py
├── transcript_utils.py
├── ui.py
├── video_utils.py
└── workflows.py
```

### 迁移指南

**旧代码**：
```python
from singleton_class import Singleton
from exceptions import VideoProcessingError
from constants import VideoConstants
from decorators import streamlit_error_handler
from logger import setup_logger
from validators import validate_video_file
from ui_sections import render_sidebar
```

**新代码**：
```python
from core import (
    Singleton,
    VideoProcessingError,
    VideoConstants,
    streamlit_error_handler,
    setup_logger,
    validate_video_file
)
from ui import render_sidebar
```

或者使用包导入：
```python
from app import (
    Singleton,
    VideoProcessingError,
    VideoConstants,
    streamlit_error_handler,
    setup_logger,
    validate_video_file
)
from app.ui import render_sidebar
```

---

## 代码优化建议

### 优先级分类

#### 🔴 高优先级（立即实施）

1. **统一异常处理机制**
   - 使用 `core.py` 中的异常类和装饰器
   - 统一的错误处理策略

2. **配置管理优化**
   - 使用环境变量管理配置
   - 分离开发/生产环境配置

3. **日志系统**
   - 使用 `core.py` 中的日志系统
   - 配置不同级别的日志

4. **安全性**
   - 文件上传验证（已在 `core.py` 中实现）
   - 路径遍历防护

#### 🟡 中优先级（近期实施）

1. **性能优化**
   - 视频处理异步化
   - 缓存策略优化
   - 向量索引优化

2. **代码质量**
   - 完善类型提示
   - 消除代码重复
   - 提取常量

#### 🟢 低优先级（长期规划）

1. **测试和文档**
   - 单元测试
   - API文档完善
   - 性能监控

### 详细优化建议

详见 [OPTIMIZATION_SUGGESTIONS.md](OPTIMIZATION_SUGGESTIONS.md)（已整合到本文档）

---

## Bug修复记录

### FAISS 向量索引加载问题

#### 问题描述

错误信息：
```
无法加载向量索引: Error in faiss::FileIOReader::FileIOReader(const char*) 
at /project/third-party/faiss/faiss/impl/io.cpp:69: 
Error: 'f' failed: could not open runtime/faiss_index_60a4607f30aa/index.faiss 
for reading: No such file or directory
```

#### 问题原因

1. 索引文件不存在：索引目录或文件被删除、从未创建，或路径不匹配
2. 视频签名变化：视频文件被修改后，签名变化导致无法找到对应的索引
3. 索引构建失败：索引构建过程中出错，但错误被忽略
4. 路径问题：索引路径计算错误

#### 解决方案

**1. 改进的索引加载逻辑**

- ✅ 加载前检查文件是否存在
- ✅ 区分文件不存在和其他错误
- ✅ 文件不存在时不显示警告（正常情况）
- ✅ 自动触发索引重建

**2. 改进的索引构建逻辑**

- ✅ 验证输入数据（segments不为空）
- ✅ 确保目录存在
- ✅ 验证保存是否成功
- ✅ 更好的错误提示

**3. 索引清理功能**

新增 `cleanup_invalid_indices()` 方法：
```python
from app.rag_utils import get_rag_system

rag_system = get_rag_system()
# 清理所有无效索引
rag_system.cleanup_invalid_indices()

# 只保留指定的视频签名
rag_system.cleanup_invalid_indices(keep_signatures=["60a4607f30aa"])
```

#### 代码变更

**`app/rag_utils.py`**:

1. **`load_vector_store()` 方法**:
   - 添加文件存在性检查
   - 改进错误处理
   - 区分文件不存在和其他错误

2. **`build_vector_store()` 方法**:
   - 添加输入验证
   - 改进路径处理
   - 添加保存验证

3. **新增 `cleanup_invalid_indices()` 方法**:
   - 清理无效索引
   - 支持保留指定签名

**`app/workflows.py`**:

1. **`ensure_vector_index()` 方法**:
   - 添加加载提示
   - 改进错误处理

### 其他已知问题

- HuggingFace 模型下载 401 错误 → 已通过设置镜像源解决
- FFmpeg 未安装 → 已添加安装说明
- LangChain 弃用警告 → 已更新到 `langchain-huggingface`

---

## 快速开始优化

### 已创建的优化文件

以下文件已经创建，可以直接使用：

1. **`app/core.py`** - 整合所有基础工具
2. **`app/__init__.py`** - 统一导出

### 如何使用

#### 1. 在现有代码中集成异常处理

**修改前**:
```python
def save_uploaded_video(uploaded_file, save_path: Path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
```

**修改后**:
```python
from core import streamlit_error_handler, validate_video_file

@streamlit_error_handler
def save_uploaded_video(uploaded_file, save_path: Path):
    validate_video_file(uploaded_file)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
```

#### 2. 添加日志记录

**修改前**:
```python
def handle_summary_mode(video_path: Path, frame_dir: Path):
    # ...
```

**修改后**:
```python
from core import setup_logger, log_execution_time

logger = setup_logger(__name__)

@log_execution_time
def handle_summary_mode(video_path: Path, frame_dir: Path):
    logger.info(f"开始处理视频总结模式: {video_path}")
    # ...
```

#### 3. 使用常量替代魔法数字

**修改前**:
```python
def extract_frames_around(video_path: Path, timestamp: float, frame_dir: Path, window: int = 2, fps: int = 1):
    # ...
```

**修改后**:
```python
from core import VideoConstants

def extract_frames_around(
    video_path: Path, 
    timestamp: float, 
    frame_dir: Path, 
    window: int = VideoConstants.DEFAULT_WINDOW_SECONDS, 
    fps: int = VideoConstants.DEFAULT_FPS
):
    # ...
```

### 优化检查清单

- [ ] 所有函数都有类型提示
- [ ] 所有异常都被正确捕获和处理
- [ ] 所有魔法数字都提取为常量
- [ ] 关键操作都有日志记录
- [ ] 文件上传都有验证
- [ ] 代码没有重复逻辑
- [ ] 配置可以通过环境变量设置

---

## 开发环境设置

### 推荐工具

- **IDE**: VS Code / PyCharm
- **Python版本**: 3.12+
- **包管理**: pip / poetry

### 开发工作流

1. **创建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **开发并测试**
   ```bash
   source venv/bin/activate
   streamlit run app/main.py
   ```

3. **代码检查**
   ```bash
   # 类型检查（如果使用mypy）
   mypy app/
   
   # 代码格式化
   black app/
   
   # 代码检查
   flake8 app/
   ```

4. **提交代码**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

### 测试建议

- 单元测试：使用 pytest
- 集成测试：测试完整工作流
- 性能测试：测量关键操作耗时

---

## 贡献指南

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型提示
- 添加文档字符串
- 保持函数简洁（单一职责）

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

类型：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

---

## 常见问题

### Q: 如何清理Python缓存？

A: 
```bash
find app -type d -name __pycache__ -exec rm -r {} +
find app -type f -name "*.pyc" -delete
```

### Q: 如何验证重构后的代码？

A: 
1. 清理缓存
2. 激活虚拟环境
3. 运行应用测试所有功能
4. 检查日志文件

### Q: 如何添加新的优化？

A: 
1. 在 `core.py` 中添加基础工具
2. 更新 `__init__.py` 导出
3. 更新相关文档
4. 测试验证

---

*最后更新: 2025-12-14*

