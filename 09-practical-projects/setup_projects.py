"""
批量为所有项目创建基础文件结构

使用方法:
    cd 实战项目
    python setup_projects.py
"""

import os
from pathlib import Path

# 项目列表
PROJECTS = [
    # LSTM 时间序列项目
    "04_时间序列项目/01_温度预测_LSTM中级",
    "04_时间序列项目/02_股票价格预测_LSTM高级",

    # XGBoost 项目（机器学习基础）
    "01_机器学习基础项目/01_Titanic生存预测_XGBoost入门",
    "01_机器学习基础项目/02_Otto分类挑战_XGBoost中级",
    "01_机器学习基础项目/04_XGBoost高级技巧_高级",

    # Transformer NLP 项目
    "03_自然语言处理项目/02_Transformer文本分类_入门",
    "03_自然语言处理项目/03_Transformer命名实体识别_中级",
    "03_自然语言处理项目/04_Transformer机器翻译_高级",

    # SVM 文本分类项目
    "01_机器学习基础项目/03_SVM文本分类_中级",
]

# 通用requirements.txt内容
REQUIREMENTS_COMMON = """# 深度学习框架
tensorflow>=2.13.0
keras>=2.13.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 机器学习
scikit-learn>=1.3.0

# 进度条
tqdm>=4.65.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0

# 工具
requests>=2.31.0
"""

REQUIREMENTS_XGBOOST = """# XGBoost相关
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# 超参数调优
optuna>=3.3.0

# 模型解释
shap>=0.42.0

# 不平衡数据处理
imbalanced-learn>=0.11.0
"""

REQUIREMENTS_TRANSFORMER = """# Transformers
transformers>=4.30.0
tokenizers>=0.13.0

# 深度学习
torch>=2.0.0

# NLP工具
nltk>=3.8.0
spacy>=3.6.0
"""

# __init__.py内容
INIT_PY = '''"""
项目模块
"""

__version__ = '1.0.0'
'''

# .gitkeep内容
GITKEEP = "# 此文件用于保持目录在git中被追踪\n"

# data/README.md模板
DATA_README = """# 数据说明

## 数据集

### 数据来源

[待补充]

### 数据集描述

[待补充]

### 数据下载

#### 方法1: 使用下载脚本（推荐）

```bash
cd data
python download_data.py
```

#### 方法2: 手动下载

[待补充]

### 数据预处理

[待补充]

---

**更新日期**: 2025-11-29
"""

# download_data.py模板
DOWNLOAD_DATA = '''"""
数据集下载脚本

使用方法:
    cd data
    python download_data.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def download_data():
    """
    下载数据集
    """
    print("="*60)
    print("下载数据集")
    print("="*60)

    print("\\n请根据项目需求实现数据下载逻辑")
    print("\\n提示:")
    print("  1. 使用Kaggle API下载")
    print("  2. 使用requests下载")
    print("  3. 使用tensorflow/keras内置数据集")

    # TODO: 实现数据下载逻辑


if __name__ == '__main__':
    download_data()
'''


def create_project_structure(project_path):
    """为单个项目创建基础文件结构"""

    project_dir = Path(project_path)
    project_name = project_dir.name

    print(f"\\n{'='*60}")
    print(f"设置项目: {project_name}")
    print(f"{'='*60}")

    # 确保目录存在
    for subdir in ['src', 'data', 'models', 'results', 'notebooks']:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)

    # 创建requirements.txt
    req_path = project_dir / 'requirements.txt'
    if not req_path.exists():
        content = REQUIREMENTS_COMMON

        # 根据项目类型添加特定依赖
        if 'XGBoost' in project_name or 'Otto' in project_name or 'Titanic' in project_name:
            content += REQUIREMENTS_XGBOOST
        elif 'Transformer' in project_name:
            content += REQUIREMENTS_TRANSFORMER

        req_path.write_text(content)
        print(f"✓ 创建: requirements.txt")

    # 创建src/__init__.py
    init_path = project_dir / 'src' / '__init__.py'
    if not init_path.exists():
        init_path.write_text(INIT_PY)
        print(f"✓ 创建: src/__init__.py")

    # 创建.gitkeep文件
    for subdir in ['models', 'results', 'data']:
        gitkeep_path = project_dir / subdir / '.gitkeep'
        if not gitkeep_path.exists():
            gitkeep_path.write_text(GITKEEP)
            print(f"✓ 创建: {subdir}/.gitkeep")

    # 创建data/README.md
    data_readme_path = project_dir / 'data' / 'README.md'
    if not data_readme_path.exists():
        data_readme_path.write_text(DATA_README)
        print(f"✓ 创建: data/README.md")

    # 创建data/download_data.py
    download_path = project_dir / 'data' / 'download_data.py'
    if not download_path.exists():
        download_path.write_text(DOWNLOAD_DATA)
        print(f"✓ 创建: data/download_data.py")

    print(f"✓ 项目 {project_name} 设置完成")


def main():
    """主函数"""
    print("="*60)
    print("批量设置项目基础文件结构")
    print("="*60)

    base_dir = Path(__file__).parent

    created_count = 0
    skipped_count = 0

    for project in PROJECTS:
        project_path = base_dir / project

        if project_path.exists():
            create_project_structure(project_path)
            created_count += 1
        else:
            print(f"\\n⚠ 跳过: {project} (目录不存在)")
            skipped_count += 1

    print(f"\\n{'='*60}")
    print("设置完成")
    print(f"{'='*60}")
    print(f"\\n成功设置: {created_count} 个项目")
    print(f"跳过: {skipped_count} 个项目")

    print(f"\\n下一步:")
    print(f"  1. 查看各项目的README.md了解项目详情")
    print(f"  2. 根据需要实现src/目录下的代码文件")
    print(f"  3. 实现data/download_data.py下载数据")
    print(f"  4. 创建notebooks/目录下的教学notebook")


if __name__ == '__main__':
    main()
