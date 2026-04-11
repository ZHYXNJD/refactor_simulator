"""
路径配置模块

用于统一项目的导入路径，确保从任何目录运行都能正确找到模块。
"""
import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 定义子目录
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DYNAMIC_MATCHING_DIR = os.path.join(PROJECT_ROOT, 'dynamic_matching')
DYNAMIC_REPO_DIR = os.path.join(PROJECT_ROOT, 'dynamic_repo')
MY_DATA_DIR = os.path.join(PROJECT_ROOT, 'my_data')

# 添加到 sys.path
for directory in [SRC_DIR, DYNAMIC_MATCHING_DIR, DYNAMIC_REPO_DIR, MY_DATA_DIR]:
    if os.path.isdir(directory) and directory not in sys.path:
        sys.path.insert(0, directory)