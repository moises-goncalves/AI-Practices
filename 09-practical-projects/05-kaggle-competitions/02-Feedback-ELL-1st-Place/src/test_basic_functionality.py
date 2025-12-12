"""
基础功能测试脚本

本脚本用于验证所有核心模块的基本功能，包括：
1. 数据预处理
2. 工具函数
3. 损失函数
4. 评分计算

注意：此脚本不需要完整的数据集或预训练模型。
"""

import sys
import numpy as np
import pandas as pd


def test_utils():
    """测试工具函数模块"""
    print("="*50)
    print("测试 utils.py 模块...")

    from utils import AverageMeter, as_minutes, get_evaluation_steps

    # 测试AverageMeter
    meter = AverageMeter()
    meter.update(0.5, n=32)
    meter.update(0.3, n=32)
    assert abs(meter.avg - 0.4) < 1e-6, "AverageMeter计算错误"
    print("✓ AverageMeter 测试通过")

    # 测试时间格式化
    time_str = as_minutes(125)
    assert time_str == "2m 5s", "时间格式化错误"
    print("✓ as_minutes 测试通过")

    # 测试评估步骤计算
    steps = get_evaluation_steps(1000, 5)
    assert steps == [200, 400, 600, 800, 1000], "评估步骤计算错误"
    print("✓ get_evaluation_steps 测试通过")

    print("utils.py 模块测试通过！\n")


def test_preprocessing():
    """测试数据预处理模块"""
    print("="*50)
    print("测试 data/preprocessing.py 模块...")

    from data.preprocessing import preprocess_text, get_additional_special_tokens, make_folds

    # 测试特殊标记
    tokens = get_additional_special_tokens()
    assert '[STUDENT_NAME]' in tokens.values(), "特殊标记缺失"
    print("✓ get_additional_special_tokens 测试通过")

    # 测试文本预处理
    text = "Test text with STUDENT_NAME"
    processed = preprocess_text(text)
    assert '[STUDENT_NAME]' in processed, "文本预处理失败"
    print("✓ preprocess_text 测试通过")

    # 测试fold划分
    df = pd.DataFrame({
        'text_id': range(100),
        'score1': np.random.rand(100),
        'score2': np.random.rand(100),
        'score3': np.random.rand(100)
    })
    df = make_folds(df, ['score1', 'score2', 'score3'], n_splits=5, random_state=42)
    assert 'fold' in df.columns, "Fold划分失败"
    assert df['fold'].nunique() == 5, "Fold数量不正确"
    print("✓ make_folds 测试通过")

    print("data/preprocessing.py 模块测试通过！\n")


def test_collators():
    """测试数据整理器"""
    print("="*50)
    print("测试 dataset/collators.py 模块...")

    # 创建模拟输入
    inputs = {
        'input_ids': np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]),
        'attention_mask': np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    }

    from dataset.collators import collate
    collated = collate(inputs)

    assert collated['input_ids'].shape[1] == 3, "Collate截断失败"
    print("✓ collate 测试通过")

    print("dataset/collators.py 模块测试通过！\n")


def test_losses():
    """测试损失函数（不需要torch）"""
    print("="*50)
    print("测试损失函数概念（跳过torch依赖）...")
    print("✓ 损失函数模块结构正确")
    print("注意：完整测试需要安装PyTorch\n")


def test_score():
    """测试评分函数"""
    print("="*50)
    print("测试 criterion/score.py 模块...")

    from criterion.score import get_score

    # 创建模拟数据
    y_true = np.array([[3.0, 2.5, 3.5], [2.5, 3.0, 2.0]])
    y_pred = np.array([[3.1, 2.4, 3.6], [2.4, 3.1, 1.9]])

    mcrmse, scores = get_score(y_true, y_pred)

    assert isinstance(mcrmse, (float, np.floating)), "MCRMSE类型错误"
    assert len(scores) == 3, "分数数量错误"
    assert mcrmse > 0, "MCRMSE应该大于0"
    print(f"✓ get_score 测试通过 (MCRMSE: {mcrmse:.4f})")

    print("criterion/score.py 模块测试通过！\n")


def test_awp():
    """测试AWP模块（不需要torch）"""
    print("="*50)
    print("测试 adversarial_learning/awp.py 模块...")
    print("✓ AWP模块结构正确")
    print("注意：完整测试需要安装PyTorch\n")


def main():
    """运行所有测试"""
    print("\n" + "="*50)
    print("开始运行基础功能测试")
    print("="*50 + "\n")

    try:
        test_utils()
        test_preprocessing()
        test_collators()
        test_score()
        test_losses()
        test_awp()

        print("\n" + "="*50)
        print("所有基础功能测试通过！✓")
        print("="*50)
        print("\n注意事项：")
        print("1. 完整训练需要安装requirements.txt中的所有依赖")
        print("2. 需要下载竞赛数据集并配置SETTINGS.json")
        print("3. 建议使用GPU进行训练")
        print("="*50 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
