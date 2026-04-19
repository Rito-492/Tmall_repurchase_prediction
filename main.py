#!/usr/bin/env python3
"""
重复购买预测 - 主入口脚本
从项目根目录运行
"""
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering import FeatureEngineering
from train_model import RepeatBuyerModel
from test_predict import predict_test_set, analyze_predictions


def main():
    print("="*70)
    print("重复购买预测 - 主入口")
    print("="*70)
    print("\n使用方法:")
    print("  python main.py train    - 训练完整模型")
    print("  python main.py predict  - 使用已有模型预测")
    print("  python main.py full     - 完整流程 (训练 + 预测)")
    print("="*70)


if __name__ == "__main__":
    main()
