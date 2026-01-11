#!/usr/bin/env python3
"""
下载 Hugging Face 模型到本地目录，支持断点续传
使用 huggingface_hub 直接下载到目标目录
"""
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# ============================================================================
# 原来的 Qwen2.5-VL-7B-Instruct 模型下载函数（已注释）
# ============================================================================
# def download_model_7b(resume=True, force_download=False):
#     """
#     下载 Qwen2.5-VL-7B-Instruct 模型到本地目录，支持断点续传
#     
#     Args:
#         resume: 是否支持断点续传（默认 True）
#         force_download: 是否强制重新下载，即使文件已存在（默认 False）
#     """
#     model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
#     base_dir = Path(__file__).parent
#     local_dir = base_dir / "model" / "Qwen2.5-VL-7B-Instruct"
#     
#     # 创建父目录
#     local_dir.parent.mkdir(parents=True, exist_ok=True)
#     
#     print(f"开始下载模型 {model_id}...")
#     print(f"目标目录: {local_dir.absolute()}")
#     
#     # 检查是否已有部分文件
#     if resume and local_dir.exists() and not force_download:
#         existing_files = list(local_dir.glob("*.safetensors"))
#         incomplete_files = list(local_dir.rglob("*.incomplete"))
#         
#         if existing_files or incomplete_files:
#             print(f"⚠️  检测到已存在的文件:")
#             if existing_files:
#                 print(f"   - 已完成的模型文件: {len(existing_files)} 个")
#             if incomplete_files:
#                 print(f"   - 未完成的下载文件: {len(incomplete_files)} 个")
#             print("   📥 将从断点处继续下载...")
#     elif force_download:
#         # 如果强制重新下载，删除目录
#         if local_dir.exists():
#             print(f"🔄 强制重新下载，正在删除现有目录: {local_dir}")
#             shutil.rmtree(local_dir)
#     
#     try:
#         print("\n正在下载模型文件...")
#         # 使用 snapshot_download，它会自动处理断点续传
#         # resume_download=True 是默认行为，无需显式指定
#         snapshot_download(
#             repo_id=model_id,
#             local_dir=str(local_dir),
#             local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
#             resume_download=resume,  # 启用断点续传
#         )
#         
#         print(f"\n✅ 模型下载完成！")
#         print(f"模型位置: {local_dir.absolute()}")
#         print(f"目录大小: {get_dir_size(local_dir) / (1024**3):.2f} GB")
#         
#     except KeyboardInterrupt:
#         print(f"\n⚠️  下载被用户中断")
#         print(f"💡 提示: 可以重新运行脚本，将从断点处继续下载")
#         raise
#     except Exception as e:
#         print(f"\n❌ 下载失败: {e}")
#         if resume:
#             print(f"💡 提示: 可以重新运行脚本，将从断点处继续下载")
#         raise

# ============================================================================
# 新的 Qwen2.5-VL-3B-Instruct 模型下载函数
# ============================================================================
def download_model(resume=True, force_download=False):
    """
    下载 Qwen2.5-VL-3B-Instruct 模型到本地目录，支持断点续传
    
    Args:
        resume: 是否支持断点续传（默认 True）
        force_download: 是否强制重新下载，即使文件已存在（默认 False）
    """
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    base_dir = Path(__file__).parent
    local_dir = base_dir / "model" / "Qwen2.5-VL-3B-Instruct"
    
    # 创建父目录
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载模型 {model_id}...")
    print(f"目标目录: {local_dir.absolute()}")
    
    # 检查是否已有部分文件
    if resume and local_dir.exists() and not force_download:
        existing_files = list(local_dir.glob("*.safetensors"))
        incomplete_files = list(local_dir.rglob("*.incomplete"))
        
        if existing_files or incomplete_files:
            print(f"⚠️  检测到已存在的文件:")
            if existing_files:
                print(f"   - 已完成的模型文件: {len(existing_files)} 个")
            if incomplete_files:
                print(f"   - 未完成的下载文件: {len(incomplete_files)} 个")
            print("   📥 将从断点处继续下载...")
    elif force_download:
        # 如果强制重新下载，删除目录
        if local_dir.exists():
            print(f"🔄 强制重新下载，正在删除现有目录: {local_dir}")
            shutil.rmtree(local_dir)
    
    try:
        print("\n正在下载模型文件...")
        # 使用 snapshot_download，它会自动处理断点续传
        # resume_download=True 是默认行为，无需显式指定
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=resume,  # 启用断点续传
        )
        
        print(f"\n✅ 模型下载完成！")
        print(f"模型位置: {local_dir.absolute()}")
        print(f"目录大小: {get_dir_size(local_dir) / (1024**3):.2f} GB")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  下载被用户中断")
        print(f"💡 提示: 可以重新运行脚本，将从断点处继续下载")
        raise
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if resume:
            print(f"💡 提示: 可以重新运行脚本，将从断点处继续下载")
        raise

def get_dir_size(path):
    """计算目录大小（字节）"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    force_download = "--force" in sys.argv or "-f" in sys.argv
    
    download_model(resume=True, force_download=force_download)

