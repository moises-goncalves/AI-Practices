#!/usr/bin/env python3
"""
Python语法检查脚本
==================

检查项目中所有Python文件的语法是否正确。
不需要导入依赖库，只检查语法。

运行方法：
    python check_syntax.py
"""

import os
import py_compile
import sys
from pathlib import Path


def check_syntax(file_path):
    """
    检查单个Python文件的语法

    参数：
        file_path: Python文件路径

    返回：
        (bool, str): (是否通过, 错误信息)
    """
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    """主函数"""
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'

    # 查找所有Python文件
    py_files = list(src_dir.rglob('*.py'))

    print("="*60)
    print("Python语法检查")
    print("="*60)
    print(f"检查目录: {src_dir}")
    print(f"找到 {len(py_files)} 个Python文件")
    print()

    passed = 0
    failed = 0
    errors = []

    for py_file in sorted(py_files):
        relative_path = py_file.relative_to(project_root)
        success, error = check_syntax(py_file)

        if success:
            print(f"✓ {relative_path}")
            passed += 1
        else:
            print(f"✗ {relative_path}")
            print(f"  {error}")
            failed += 1
            errors.append((relative_path, error))

    # 打印总结
    print()
    print("="*60)
    print("检查总结")
    print("="*60)
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")

    if failed > 0:
        print("\n语法错误详情:")
        for path, error in errors:
            print(f"\n{path}:")
            print(f"  {error}")
        return 1
    else:
        print("\n所有文件语法检查通过! ✓")
        return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
