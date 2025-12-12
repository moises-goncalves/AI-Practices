"""
快速修复所有Config文件的torch导入问题

这个脚本会批量修复所有配置文件中的torch.device导入问题
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CONFIGS_DIR = PROJECT_ROOT / "Configs"

def fix_torch_import(file_path):
    """修复单个配置文件的torch导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否包含 device = torch.device('cuda')
    if "device = torch.device('cuda')" in content and "try:" not in content[:500]:
        # 在import torch后添加try-except
        fixed_content = re.sub(
            r'(import torch\n)',
            r'\1\ntry:\n    device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')\nexcept:\n    device = \'cuda\'\n\n',
            content
        )

        # 移除类内部的 device = torch.device('cuda')
        fixed_content = re.sub(
            r'    device = torch\.device\(\'cuda\'\)',
            '    device = device',
            fixed_content
        )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        return True
    return False

def main():
    """修复所有配置文件"""
    print("开始修复Config文件...")
    fixed_count = 0

    for cfg_file in CONFIGS_DIR.glob("*_cfg.py"):
        if fix_torch_import(cfg_file):
            print(f"✓ 已修复: {cfg_file.name}")
            fixed_count += 1
        else:
            print(f"- 跳过: {cfg_file.name} (无需修复或已修复)")

    print(f"\n总计修复 {fixed_count} 个文件")

if __name__ == "__main__":
    main()
