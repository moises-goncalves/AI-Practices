#!/usr/bin/env python3
"""
RSNA 2024 Lumbar Spine 项目综合测试脚本
========================================

本脚本用于验证项目中所有关键模块是否能够正常导入和运行。

测试内容：
    1. 基础环境测试（PyTorch, CUDA等）
    2. 路径配置测试
    3. 通用工具模块测试
    4. NFN模型模块测试
    5. SCS模型模块测试
    6. 数据处理模块测试

运行方法：
    cd /path/to/04-RSNA-2024-Lumbar-Spine
    python test_all_modules.py

预期输出：
    所有测试项前会显示 ✓ 或 ✗ 标记
    最后显示通过/失败的测试数量统计
"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestRunner:
    """测试运行器"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test(self, name, func):
        """运行单个测试"""
        try:
            func()
            print(f"✓ {name}")
            self.passed += 1
            self.tests.append((name, True, None))
        except Exception as e:
            print(f"✗ {name}")
            print(f"  错误: {str(e)}")
            self.failed += 1
            self.tests.append((name, False, str(e)))

    def summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        print(f"总计: {self.passed + self.failed}")
        print(f"成功率: {self.passed / (self.passed + self.failed) * 100:.1f}%")

        if self.failed > 0:
            print("\n失败的测试:")
            for name, passed, error in self.tests:
                if not passed:
                    print(f"  - {name}: {error}")

        return self.failed == 0


def main():
    """主测试函数"""
    runner = TestRunner()

    print("="*60)
    print("RSNA 2024 Lumbar Spine - 模块测试")
    print("="*60)
    print()

    # ===== 1. 基础环境测试 =====
    print("1. 基础环境测试")
    print("-"*60)

    def test_pytorch():
        import torch
        assert torch.__version__, "PyTorch未正确安装"
    runner.test("PyTorch导入", test_pytorch)

    def test_cuda():
        import torch
        # CUDA可能不可用，这不应该算失败
        if torch.cuda.is_available():
            assert torch.cuda.device_count() > 0
        else:
            print("    注意: CUDA不可用（CPU模式）")
    runner.test("CUDA检查", test_cuda)

    def test_basic_libs():
        import numpy as np
        import pandas as pd
        import cv2
        import timm
        import albumentations
    runner.test("基础科学计算库", test_basic_libs)

    print()

    # ===== 2. 路径配置测试 =====
    print("2. 路径配置测试")
    print("-"*60)

    def test_dir_setting():
        sys.path.append('src/third_party')
        from _dir_setting_ import DATA_KAGGLE_DIR, DATA_PROCESSED_DIR, RESULT_DIR
        assert DATA_KAGGLE_DIR, "DATA_KAGGLE_DIR未设置"
        assert DATA_PROCESSED_DIR, "DATA_PROCESSED_DIR未设置"
        assert RESULT_DIR, "RESULT_DIR未设置"
    runner.test("路径配置导入", test_dir_setting)

    print()

    # ===== 3. 通用模块测试 =====
    print("3. 通用模块测试")
    print("-"*60)

    def test_common():
        sys.path.insert(0, 'src')
        from common import pytorch_version_to_text
        text = pytorch_version_to_text()
        assert len(text) > 0
    runner.test("common.py导入", test_common)

    print()

    # ===== 4. NFN模型测试 =====
    print("4. NFN模型模块测试")
    print("-"*60)

    def test_nfn_decoder():
        sys.path.insert(0, 'src/nfn_trainer')
        from decoder import MyUnetDecoder3d
        import torch
        decoder = MyUnetDecoder3d(
            in_channel=512,
            skip_channel=[320, 128, 64],
            out_channel=[384, 192, 96]
        )
        # 测试前向传播
        feature = torch.randn(1, 512, 5, 10, 10)
        skips = [
            torch.randn(1, 320, 5, 20, 20),
            torch.randn(1, 128, 5, 40, 40),
            torch.randn(1, 64, 5, 80, 80)
        ]
        last, decode_list = decoder(feature, skips)
        assert last.shape == (1, 96, 5, 80, 80)
    runner.test("NFN解码器", test_nfn_decoder)

    def test_nfn_configure():
        sys.path.insert(0, 'src/nfn_trainer')
        sys.path.insert(0, 'src')
        from configure import default_cfg
        assert default_cfg.image_size == 320
    runner.test("NFN配置", test_nfn_configure)

    def test_nfn_augmentation():
        sys.path.insert(0, 'src/nfn_trainer')
        from augmentation import do_resize_and_center
        import numpy as np
        image = np.random.rand(100, 100, 3)
        point = np.array([[10, 20], [30, 40]])
        resized_img, resized_point = do_resize_and_center(image, point, 512)
        assert resized_img.shape[:2] == (512, 512)
    runner.test("NFN数据增强", test_nfn_augmentation)

    print()

    # ===== 5. SCS模型测试 =====
    print("5. SCS模型模块测试")
    print("-"*60)

    def test_scs_decoder():
        sys.path.insert(0, 'src/scs_trainer')
        from decoder import MyUnetDecoder
        import torch
        decoder = MyUnetDecoder(
            in_channel=512,
            skip_channel=[320, 128, 64],
            out_channel=[384, 192, 96]
        )
        feature = torch.randn(1, 512, 10, 10)
        skips = [
            torch.randn(1, 320, 20, 20),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 64, 80, 80)
        ]
        last, decode_list = decoder(feature, skips)
        assert last.shape == (1, 96, 80, 80)
    runner.test("SCS解码器", test_scs_decoder)

    print()

    # ===== 6. 第三方库测试 =====
    print("6. 第三方库模块测试")
    print("-"*60)

    def test_my_lib():
        sys.path.insert(0, 'src/third_party')
        from my_lib.other import dotdict
        d = dotdict({'a': 1, 'b': 2})
        assert d.a == 1
        assert d['b'] == 2
    runner.test("my_lib.other", test_my_lib)

    def test_net_rate():
        sys.path.insert(0, 'src/third_party')
        from my_lib.net.rate import NullScheduler
        scheduler = NullScheduler(lr=0.01)
        assert scheduler(0) == 0.01
    runner.test("my_lib.net.rate", test_net_rate)

    print()

    # 打印总结
    success = runner.summary()

    if success:
        print("\n" + "="*60)
        print("所有测试通过! ✓")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("部分测试失败，请查看上方错误信息")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
