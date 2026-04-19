"""
重复购买预测 - 完整流程执行脚本
一键运行特征工程、模型训练和预测的完整pipeline
"""
import os
import sys
import time

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngineering
from train_model import RepeatBuyerModel
from test_predict import predict_test_set, analyze_predictions
import warnings
warnings.filterwarnings('ignore')


def check_data_files(data_path='./data_format1/'):
    """检查数据文件是否存在"""
    required_files = [
        'user_log_format1.csv',
        'user_info_format1.csv',
        'train_format1.csv',
        'test_format1.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ 缺少以下数据文件:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\n请将数据文件放置在 {data_path} 目录下")
        return False
    
    print("✓ 所有必需的数据文件都已就绪")
    return True


def run_full_pipeline(data_path='./data_format1/', use_cv=True, n_splits=5):
    """
    运行完整的预测流程
    
    Args:
        data_path: 数据文件夹路径
        use_cv: 是否使用交叉验证（推荐True）
        n_splits: 交叉验证折数
    """
    
    print("="*70)
    print("重复购买预测 - 完整Pipeline")
    print("="*70)
    
    start_time = time.time()
    
    # ============== 步骤1: 检查数据文件 ==============
    print("\n【步骤 1/4】检查数据文件")
    print("-" * 70)
    if not check_data_files(data_path):
        return
    
    # ============== 步骤2: 特征工程 ==============
    print("\n【步骤 2/4】特征工程")
    print("-" * 70)
    
    fe = FeatureEngineering()
    fe.load_data(data_path)
    
    # 构建训练集特征
    print("\n构建训练集特征...")
    train_features = fe.build_features(fe.train_data, is_train=True)
    train_features.to_csv('outputs/features/train_features.csv', index=False)
    print("✓ 训练特征已保存到 outputs/train_features.csv")
    
    # ============== 步骤3: 模型训练 ==============
    print("\n【步骤 3/4】模型训练")
    print("-" * 70)
    
    model = RepeatBuyerModel()
    
    if use_cv:
        print(f"\n使用 {n_splits} 折交叉验证训练...")
        oof_preds, oof_auc = model.train_with_cv(train_features, n_splits=n_splits)
        
        # 保存模型和OOF预测
        model.save_model('checkpoints/model_cv.pkl')
        
        oof_results = train_features[['user_id', 'merchant_id', 'label']].copy()
        oof_results['oof_pred'] = oof_preds
        oof_results.to_csv('results/oof_predictions.csv', index=False)
        print("✓ OOF预测结果已保存到 oof_predictions.csv")
        
        model_path = 'checkpoints/model_cv.pkl'
    else:
        print("\n使用简单训练验证划分...")
        X_train, X_val, y_train, y_val, _ = model.prepare_data(train_features)
        lgb_model = model.train_single_model(X_train, y_train, X_val, y_val)
        model.models = [lgb_model]
        
        # 评估并保存
        model.evaluate(X_val, y_val)
        model.save_model('checkpoints/model_single.pkl')
        
        model_path = 'checkpoints/model_single.pkl'
    
    # ============== 步骤4: 测试集预测 ==============
    print("\n【步骤 4/4】测试集预测")
    print("-" * 70)
    
    # 重新加载特征工程器以构建测试集特征
    fe_test = FeatureEngineering()
    fe_test.load_data(data_path)
    
    # 必须先构建训练集以获得全局统计特征
    print("\n重新构建训练集特征（用于测试集特征工程）...")
    _ = fe_test.build_features(fe_test.train_data, is_train=True)
    
    # 构建测试集特征
    print("\n构建测试集特征...")
    test_features = fe_test.build_features(fe_test.test_data, is_train=False)
    
    # 加载模型进行预测
    print("\n加载模型进行预测...")
    model_pred = RepeatBuyerModel()
    model_pred.load_model(model_path)
    
    # 预测
    feature_cols = [col for col in test_features.columns 
                   if col not in ['user_id', 'merchant_id', 'label']]
    X_test = test_features[feature_cols]
    predictions = model_pred.predict(X_test)
    
    # 生成提交文件
    submission = test_features[['user_id', 'merchant_id']].copy()
    submission['prob'] = predictions
    submission.to_csv('results/prediction.csv', index=False)
    print("✓ 提交文件已保存到 prediction.csv")
    
    # 保存详细结果
    detailed_submission = test_features[['user_id', 'merchant_id']].copy()
    detailed_submission['prob'] = predictions
    key_features = ['um_total_actions', 'um_action_2_cnt', 'user_buy_cnt']
    for feat in key_features:
        if feat in test_features.columns:
            detailed_submission[feat] = test_features[feat]
    detailed_submission.to_csv('results/detailed_prediction.csv', index=False)
    print("✓ 详细结果已保存到 detailed_prediction.csv")
    
    # 分析预测结果
    print("\n预测结果分析:")
    analyze_predictions('results/prediction.csv')
    
    # ============== 完成 ==============
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("✓ 完整Pipeline执行完成！")
    print("="*70)
    print(f"\n总耗时: {total_time/60:.2f} 分钟")
    print("\n生成的文件:")
    print("  1. train_features.csv - 训练集特征")
    if use_cv:
        print("  2. model_cv.pkl - 交叉验证模型")
        print("  3. oof_predictions.csv - OOF预测结果")
    else:
        print("  2. model_single.pkl - 单次训练模型")
    print("  4. prediction.csv - 测试集预测结果（提交文件）")
    print("  5. detailed_prediction.csv - 详细预测结果")
    
    print("\n下一步:")
    print("  - 提交 prediction.csv 到比赛平台")
    print("  - 查看 oof_predictions.csv 分析模型在训练集上的表现")
    print("  - 查看 detailed_prediction.csv 进行深入分析")


def quick_predict(model_path='checkpoints/model_cv.pkl', data_path='./data_format1/'):
    """
    快速预测（假设已经训练好模型）
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件夹路径
    """
    print("="*70)
    print("快速预测模式（使用已训练的模型）")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件 {model_path} 不存在")
        print("请先运行完整pipeline训练模型，或指定正确的模型路径")
        return
    
    try:
        submission = predict_test_set(model_path=model_path, data_path=data_path)
        analyze_predictions('results/prediction.csv')
        
        print("\n✓ 快速预测完成！")
        print("✓ 提交文件: prediction.csv")
        
    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='重复购买预测完整Pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'quick'],
                       help='运行模式: full=完整流程, quick=仅预测')
    parser.add_argument('--data_path', type=str, default='./data_format1/',
                       help='数据文件夹路径')
    parser.add_argument('--use_cv', type=bool, default=True,
                       help='是否使用交叉验证')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_cv.pkl',
                       help='模型文件路径（quick模式使用）')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        # 运行完整流程
        run_full_pipeline(
            data_path=args.data_path,
            use_cv=args.use_cv,
            n_splits=args.n_splits
        )
    elif args.mode == 'quick':
        # 快速预测
        quick_predict(
            model_path=args.model_path,
            data_path=args.data_path
        )
