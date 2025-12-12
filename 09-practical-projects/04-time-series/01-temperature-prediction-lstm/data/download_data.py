"""
数据集下载脚本

Jena气候数据集下载
数据集信息：
    - 时间跨度：2009-2016年
    - 采样频率：每10分钟
    - 特征数：14个气象指标
    - 样本数：420,551条记录

使用方法:
    cd data
    python download_data.py
"""

import sys
import os
import requests
import zipfile
from pathlib import Path


def download_jena_climate_data(save_dir='.'):
    """
    下载Jena气候数据集

    数据来源：Max Planck Institute for Biogeochemistry
    """
    print("="*60)
    print("下载Jena气候数据集")
    print("="*60)

    # 数据集URL
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    zip_path = save_dir / "jena_climate_2009_2016.csv.zip"
    csv_path = save_dir / "jena_climate_2009_2016.csv"

    # 检查是否已存在
    if csv_path.exists():
        print(f"\n数据集已存在: {csv_path}")
        print(f"文件大小: {csv_path.stat().st_size / (1024*1024):.2f} MB")
        return str(csv_path)

    try:
        print(f"\n正在下载...")
        print(f"URL: {url}")

        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  进度: {progress:.1f}%", end='', flush=True)

        print(f"\n下载完成: {zip_path}")

        # 解压文件
        print("\n正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"解压完成: {csv_path}")

        # 删除zip文件
        zip_path.unlink()
        print("已删除临时文件")

        # 显示数据集信息
        print("\n" + "="*60)
        print("数据集信息")
        print("="*60)
        print(f"文件路径: {csv_path}")
        print(f"文件大小: {csv_path.stat().st_size / (1024*1024):.2f} MB")

        # 读取并显示前几行
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=5)
            print(f"\n数据预览:")
            print(df.head())
            print(f"\n列名:")
            for col in df.columns:
                print(f"  - {col}")
        except ImportError:
            print("\n提示: 安装pandas后可以预览数据")

        print("\n" + "="*60)
        print("下载成功！")
        print("="*60)

        return str(csv_path)

    except requests.exceptions.RequestException as e:
        print(f"\n下载失败: {e}")
        print("\n备选方案:")
        print("  1. 手动下载: https://www.kaggle.com/datasets/mnassrib/jena-climate")
        print("  2. 或使用: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip")
        print(f"  3. 将文件保存到: {save_dir}")
        return None
    except Exception as e:
        print(f"\n错误: {e}")
        return None


def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent

    # 下载数据
    csv_path = download_jena_climate_data(script_dir)

    if csv_path:
        print(f"\n下一步:")
        print(f"  1. 查看数据: head {csv_path}")
        print(f"  2. 训练模型: cd .. && python src/train.py")
    else:
        print("\n请手动下载数据集")
        sys.exit(1)


if __name__ == '__main__':
    main()
