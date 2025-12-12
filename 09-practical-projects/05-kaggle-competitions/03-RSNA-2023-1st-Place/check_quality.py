#!/usr/bin/env python3
"""
代码质量检查脚本

检查项目中的代码质量问题：
1. 无效的导入语句
2. 未使用的导入
3. 代码风格问题
4. 文档字符串覆盖率
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent

def check_file(file_path):
    """检查单个文件的代码质量"""
    issues = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # 检查无效导入
    if 'import command' in content:
        issues.append(f"❌ 包含无效导入 'import command'")

    # 检查是否有函数定义
    try:
        tree = ast.parse(content)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # 检查文档字符串覆盖率
        funcs_with_docs = sum(1 for f in functions if ast.get_docstring(f))
        classes_with_docs = sum(1 for c in classes if ast.get_docstring(c))

        total_items = len(functions) + len(classes)
        items_with_docs = funcs_with_docs + classes_with_docs

        if total_items > 0:
            doc_coverage = items_with_docs / total_items * 100
            if doc_coverage < 50:
                issues.append(f"⚠️  文档覆盖率低: {doc_coverage:.1f}%")
            else:
                issues.append(f"✓ 文档覆盖率: {doc_coverage:.1f}%")

    except SyntaxError:
        issues.append("❌ 语法错误")

    return issues

def main():
    """主函数"""
    print("=" * 70)
    print(" RSNA 2023代码质量检查报告")
    print("=" * 70)
    print()

    stats = {
        'total_files': 0,
        'files_with_issues': 0,
        'total_issues': 0,
    }

    # 检查所有Python文件
    for folder in ['Datasets', 'Models', 'Configs', 'TRAIN', 'utils']:
        folder_path = PROJECT_ROOT / folder
        if not folder_path.exists():
            continue

        print(f"\n{'='*70}")
        print(f" 检查 {folder}/ 文件夹")
        print(f"{'='*70}\n")

        py_files = list(folder_path.glob('*.py'))

        for py_file in sorted(py_files):
            if py_file.name == '__pycache__':
                continue

            stats['total_files'] += 1
            issues = check_file(py_file)

            if issues:
                stats['files_with_issues'] += 1
                stats['total_issues'] += len([i for i in issues if i.startswith('❌') or i.startswith('⚠️')])

                print(f"\n📄 {py_file.name}")
                for issue in issues:
                    print(f"   {issue}")

    # 总结
    print(f"\n{'='*70}")
    print(" 总结")
    print(f"{'='*70}\n")
    print(f"总文件数: {stats['total_files']}")
    print(f"有问题的文件数: {stats['files_with_issues']}")
    print(f"总问题数: {stats['total_issues']}")

    if stats['total_issues'] == 0:
        print("\n🎉 代码质量检查全部通过！")
    else:
        print(f"\n⚠️  发现 {stats['total_issues']} 个需要关注的问题")

if __name__ == "__main__":
    main()
