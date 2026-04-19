"""
快速开始脚本 - 一键完成特征工程、训练和预测
"""
import os
import sys
import time


def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)


def run_step(step_name, command, check_output=None):
    """运行单个步骤"""
    print("\n" + "="*70)
    print(f"执行步骤: {step_name}")
    print("="*70)
    
    start_time = time.time()
    
    # 执行命令
    ret = os.system(command)
    
    elapsed_time = time.time() - start_time
    
    if ret != 0:
        print(f"\n❌ {step_name} 执行失败！")
        return False
    
    # 检查输出文件
    if check_output:
        if not check_file_exists(check_output):
            print(f"\n❌ 输出文件不存在: {check_output}")
            return False
        print(f"\n✓ 输出文件生成成功: {check_output}")
    
    print(f"\n✓ {step_name} 完成！耗时: {elapsed_time:.2f} 秒")
    return True


def quick_start(skip_features=False):
    """快速开始完整流程"""
    
    print("="*70)
    print("重复购买预测 - 快速开始")
    print("="*70)
    
    total_start = time.time()
    
    # 检查数据文件
    print("\n检查数据文件...")
    required_files = [
        './data_format1/user_log_format1.csv',
        './data_format1/user_info_format1.csv',
        './data_format1/train_format1.csv',
        './data_format1/test_format1.csv'
    ]
    
    missing_files = [f for f in required_files if not check_file_exists(f)]
    if missing_files:
        print("\n❌ 缺少以下数据文件:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n请将数据文件放在 ./data_format1/ 目录下")
        return False
    print("✓ 所有数据文件已就绪")
    
    # 步骤1: 特征工程（可选跳过）
    if not skip_features:
        if not run_step(
            "步骤1: 特征工程",
            f"{sys.executable} feature_engineering.py",
            check_output='./features/train_features.csv'
        ):
            return False
    else:
        print("\n跳过特征工程步骤（使用已有特征）")
        # 检查特征文件是否存在
        if not check_file_exists('./features/train_features.csv'):
            print("❌ 特征文件不存在，无法跳过特征工程！")
            return False
        print("✓ 特征文件已存在")
    
    # 步骤2: 模型训练
    if not run_step(
        "步骤2: 模型训练",
        f"{sys.executable} train_model.py",
        check_output='model_cv.pkl'
    ):
        return False
    
    # 步骤3: 测试集预测
    if not run_step(
        "步骤3: 测试集预测",
        f"{sys.executable} test_predict.py",
        check_output='prediction.csv'
    ):
        return False
    
    # 完成
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("✓ 所有步骤完成！")
    print("="*70)
    print(f"\n总耗时: {total_time/60:.2f} 分钟")
    print("\n生成的文件:")
    print("  特征文件:")
    print("    - features/train_features.csv")
    print("    - features/test_features.csv")
    print("  模型文件:")
    print("    - model_cv.pkl")
    print("    - model_single.pkl")
    print("  结果文件:")
    print("    - prediction.csv (提交文件)")
    print("    - detailed_prediction.csv (详细结果)")
    print("    - oof_predictions.csv (OOF预测)")
    print("    - feature_importance.csv (特征重要性)")
    
    return True


def show_usage():
    """显示使用说明"""
    print("""
使用方法:
    
1. 完整运行（包含特征工程）:
   python quick_start.py
   
2. 跳过特征工程（使用已有特征，适合调参）:
   python quick_start.py --skip-features
   
3. 只运行特定步骤:
   python feature_engineering.py  # 只构建特征
   python train_model.py          # 只训练模型
   python test_predict.py         # 只预测测试集

注意:
- 首次运行必须构建特征（步骤1）
- 特征构建完成后，可以使用 --skip-features 跳过特征工程，直接训练和预测
- 这样可以快速进行模型调参和实验
""")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='重复购买预测 - 快速开始')
    parser.add_argument('--skip-features', action='store_true',
                       help='跳过特征工程步骤（使用已有特征）')
    parser.add_argument('--help-usage', action='store_true',
                       help='显示详细使用说明')
    
    args = parser.parse_args()
    
    if args.help_usage:
        show_usage()
        return
    
    # 运行快速开始流程
    success = quick_start(skip_features=args.skip_features)
    
    if success:
        print("\n🎉 成功！现在可以提交 prediction.csv 文件了！")
    else:
        print("\n❌ 执行失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
