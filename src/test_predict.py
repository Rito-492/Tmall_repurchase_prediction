
import pandas as pd
import numpy as np
from train_model import RepeatBuyerModel
from feature_engineering import FeatureEngineering
import warnings
warnings.filterwarnings('ignore')


def predict_test_set(model_path='checkpoints/model_cv.pkl', feature_dir='./outputs/features/'):
    """
    对测试集进行预测并生成提交文件
    
    Args:
        model_path: 模型文件路径
        feature_dir: 特征文件夹路径
    """
    
    print("="*70)
    print("开始测试集预测")
    print("="*70)
    
    # 1. 加载训练好的模型
    print("\n步骤1: 加载训练好的模型...")
    model = RepeatBuyerModel()
    model.load_model(model_path)
    print(f"成功加载 {len(model.models)} 个模型")
    
    # 2. 加载测试集特征
    print("\n步骤2: 加载测试集特征...")
    try:
        _, test_features, feature_info = FeatureEngineering.load_features(feature_dir)
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请先运行特征工程脚本生成特征:")
        print("  python feature_engineering.py")
        return None
    
    # 3. 准备预测数据
    print("\n步骤3: 准备预测数据...")
    feature_cols = [col for col in test_features.columns 
                   if col not in ['user_id', 'merchant_id', 'label']]
    
    X_test = test_features[feature_cols]
    
    print(f"测试集特征形状: {X_test.shape}")
    print(f"特征数量: {len(feature_cols)}")
    
    # 4. 进行预测
    print("\n步骤4: 进行预测...")
    predictions = model.predict(X_test)
    
    print(f"预测完成，生成 {len(predictions)} 个预测值")
    print(f"预测值统计:")
    print(f"  最小值: {predictions.min():.6f}")
    print(f"  最大值: {predictions.max():.6f}")
    print(f"  平均值: {predictions.mean():.6f}")
    print(f"  中位数: {np.median(predictions):.6f}")
    
    # 5. 生成提交文件
    print("\n步骤5: 生成提交文件...")
    submission = pd.DataFrame({
        'user_id': test_features['user_id'],
        'merchant_id': test_features['merchant_id'],
        'prob': predictions
    })
    
    # 保存提交文件
    submission.to_csv('results/prediction.csv', index=False)
    print("\n✓ 提交文件已保存到 prediction.csv")
    print(f"  提交文件形状: {submission.shape}")
    print("\n提交文件前10行预览:")
    print(submission.head(10))
    
    # 6. 预测值分布分析
    print("\n步骤6: 预测值分布分析:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(predictions, bins=bins)
    
    print("\n预测概率分布:")
    for i in range(len(bins)-1):
        print(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}): {hist[i]:6d} ({hist[i]/len(predictions)*100:5.2f}%)")
    
    # 7. 保存详细结果（包含更多信息用于分析）
    detailed_submission = test_features[['user_id', 'merchant_id']].copy()
    detailed_submission['prob'] = predictions
    
    # 添加一些关键特征用于后续分析
    key_features = ['um_total_actions', 'um_action_2_cnt', 'user_buy_cnt', 
                   'merchant_buy_ratio', 'um_buy_ratio']
    for feat in key_features:
        if feat in test_features.columns:
            detailed_submission[feat] = test_features[feat]
    
    detailed_submission.to_csv('results/detailed_prediction.csv', index=False)
    print("\n✓ 详细预测结果已保存到 detailed_prediction.csv")
    
    print("\n" + "="*70)
    print("预测完成！")
    print("="*70)
    
    return submission


def analyze_predictions(prediction_file='results/prediction.csv'):
    """
    分析预测结果
    
    Args:
        prediction_file: 预测文件路径
    """
    print("\n" + "="*70)
    print("预测结果分析")
    print("="*70)
    
    predictions = pd.read_csv(prediction_file)
    
    print(f"\n总预测数: {len(predictions)}")
    print(f"\n概率统计:")
    print(predictions['prob'].describe())
    
    # 不同阈值下的预测分析
    print("\n不同阈值下的正样本数量:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        positive_count = (predictions['prob'] >= threshold).sum()
        positive_ratio = positive_count / len(predictions)
        print(f"  阈值 {threshold:.1f}: {positive_count:6d} ({positive_ratio*100:5.2f}%)")
    
    # 高置信度预测
    high_conf_positive = predictions[predictions['prob'] >= 0.7]
    high_conf_negative = predictions[predictions['prob'] <= 0.3]
    
    print(f"\n高置信度预测:")
    print(f"  高概率重复购买 (prob >= 0.7): {len(high_conf_positive)} ({len(high_conf_positive)/len(predictions)*100:.2f}%)")
    print(f"  低概率重复购买 (prob <= 0.3): {len(high_conf_negative)} ({len(high_conf_negative)/len(predictions)*100:.2f}%)")


def main():
    """主预测流程"""
    
    print("="*70)
    print("测试集预测 - 从保存的特征和模型加载")
    print("="*70)
    
    # 选择使用哪个模型进行预测
    # model_cv.pkl: K折交叉验证训练的模型（推荐）
    # model_single.pkl: 单次训练的模型
    
    model_path = 'model_cv.pkl'  # 使用交叉验证模型
    feature_dir = './outputs/features/'  # 特征文件目录
    
    try:
        # 进行预测
        submission = predict_test_set(model_path=model_path, feature_dir=feature_dir)
        
        if submission is not None:
            # 分析预测结果
            analyze_predictions('results/prediction.csv')
            
            print("\n" + "="*70)
            print("✓ 预测流程完成！")
            print("="*70)
            print(f"\n生成的文件:")
            print(f"  1. prediction.csv - 提交文件")
            print(f"  2. detailed_prediction.csv - 详细结果")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("  1. 已运行 feature_engineering.py 生成特征")
        print("  2. 已运行 train_model.py 训练模型")
    except Exception as e:
        print(f"\n预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()