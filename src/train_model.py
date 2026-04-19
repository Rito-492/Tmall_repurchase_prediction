import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineering


class RepeatBuyerModel:
    
    def __init__(self, params=None):
        """初始化模型参数"""
        if params is None:
            self.params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 40,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 8,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        else:
            self.params = params
        
        self.models = []
        self.feature_importance = None
        
    def prepare_data(self, features, test_size=0.2, random_state=42):
        """准备训练数据"""
        print("正在准备训练数据...")
        
        # 分离特征和标签
        feature_cols = [col for col in features.columns 
                       if col not in ['user_id', 'merchant_id', 'label']]
        
        X = features[feature_cols]
        y = features['label']
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"正样本比例 - 训练集: {y_train.mean():.4f}, 验证集: {y_val.mean():.4f}")
        
        return X_train, X_val, y_train, y_val, feature_cols
    
    def train_single_model(self, X_train, y_train, X_val, y_val, 
                          num_boost_round=1000, early_stopping_rounds=50):
        """训练单个模型"""
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练模型
        print("\n开始训练模型...")
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(50)
            ]
        )
        
        return model
    
    def train_with_cv(self, features, n_splits=7, random_state=42):
        """使用K折交叉验证训练模型"""
        print(f"\n{'='*50}")
        print(f"开始{n_splits}折交叉验证训练")
        print(f"{'='*50}")
        
        # 准备数据
        feature_cols = [col for col in features.columns 
                       if col not in ['user_id', 'merchant_id', 'label']]
        
        X = features[feature_cols].values
        y = features['label'].values
        
        # K折交叉验证
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        oof_preds = np.zeros(len(features))
        feature_importance_list = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*50}")
            print(f"训练第 {fold + 1}/{n_splits} 折")
            print(f"{'='*50}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 训练模型
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=1500,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(50)
                ]
            )
            
            # 预测验证集
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds
            
            # 计算验证集AUC
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"\n第 {fold + 1} 折验证集 AUC: {val_auc:.6f}")
            
            # 保存模型
            self.models.append(model)
            
            # 保存特征重要性
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance(),
                'fold': fold + 1
            })
            feature_importance_list.append(importance)
        
        # 计算整体OOF AUC
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"\n{'='*50}")
        print(f"整体 OOF AUC: {oof_auc:.6f}")
        print(f"{'='*50}")
        
        # 汇总特征重要性
        self.feature_importance = pd.concat(feature_importance_list)
        avg_importance = self.feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        print("\n前20个重要特征:")
        print(avg_importance.head(20))
        
        return oof_preds, oof_auc
    
    def evaluate(self, X, y, threshold=0.5):
        """评估模型性能"""
        print("\n评估模型性能...")
        
        # 预测
        if len(self.models) > 1:
            # 多个模型的平均预测
            preds = np.zeros(len(X))
            for model in self.models:
                preds += model.predict(X) / len(self.models)
        else:
            preds = self.models[0].predict(X)
        
        # 计算各项指标
        auc = roc_auc_score(y, preds)
        
        preds_binary = (preds >= threshold).astype(int)
        acc = accuracy_score(y, preds_binary)
        precision = precision_score(y, preds_binary)
        recall = recall_score(y, preds_binary)
        f1 = f1_score(y, preds_binary)
        
        print(f"\nAUC Score: {auc:.6f}")
        print(f"Accuracy: {acc:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"F1 Score: {f1:.6f}")
        
        return {
            'auc': auc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': preds
        }
    
    def predict(self, X):
        """预测新数据"""
        if len(self.models) == 0:
            raise ValueError("模型尚未训练！")
        
        # 多个模型的平均预测
        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict(X) / len(self.models)
        
        return preds
    
    def save_model(self, filepath='models/repeat_buyer_model.pkl'):
        """保存模型"""
        print(f"\n保存模型到 {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'params': self.params,
                'feature_importance': self.feature_importance
            }, f)
        print("模型保存成功！")
    
    def load_model(self, filepath='models/repeat_buyer_model.pkl'):
        """加载模型"""
        print(f"\n从 {filepath} 加载模型")
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.models = saved_data['models']
            self.params = saved_data['params']
            self.feature_importance = saved_data['feature_importance']
        print("模型加载成功！")
        return self


def main():
    
    print("="*70)
    print("模型训练 - 从保存的特征文件加载")
    print("="*70)
    
    # 1. 加载已保存的特征
    print("\n步骤1: 加载特征...")
    try:
        train_features, test_features, feature_info = FeatureEngineering.load_features('./outputs/features/')
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请先运行特征工程脚本生成特征:")
        print("  python feature_engineering.py")
        return
    
    print(f"\n训练集形状: {train_features.shape}")
    print(f"测试集形状: {test_features.shape}")
    print(f"正样本比例: {train_features['label'].mean():.4f}")
    
    # 2. 初始化模型
    print("\n步骤2: 初始化模型...")
    model = RepeatBuyerModel()
    
    # 3. 方法1: 简单训练验证划分
    print("\n" + "="*70)
    print("方法1: 简单训练验证划分")
    print("="*70)
    
    X_train, X_val, y_train, y_val, feature_cols = model.prepare_data(train_features)
    
    # 训练模型
    lgb_model = model.train_single_model(X_train, y_train, X_val, y_val)
    model.models = [lgb_model]
    
    # 评估模型
    results = model.evaluate(X_val, y_val)
    
    # 保存模型
    model.save_model('checkpoints/model_single.pkl')
    
    # 4. 方法2: K折交叉验证（推荐）
    print("\n" + "="*70)
    print("方法2: K折交叉验证（更稳定，推荐使用）")
    print("="*70)
    
    model_cv = RepeatBuyerModel()
    oof_preds, oof_auc = model_cv.train_with_cv(train_features, n_splits=7)
    
    # 保存交叉验证模型
    model_cv.save_model('checkpoints/model_cv.pkl')
    
    # 保存OOF预测结果
    oof_results = train_features[['user_id', 'merchant_id', 'label']].copy()
    oof_results['oof_pred'] = oof_preds
    oof_results.to_csv('results/oof_predictions.csv', index=False)
    print("\nOOF预测结果已保存到 oof_predictions.csv")
    
    # 保存特征重要性
    if model_cv.feature_importance is not None:
        importance_path = 'results/feature_importance.csv'
        model_cv.feature_importance.to_csv(importance_path, index=False)
        print(f"特征重要性已保存到 {importance_path}")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  1. model_single.pkl - 单次训练模型")
    print("  2. model_cv.pkl - 交叉验证模型（推荐用于预测）")
    print("  3. oof_predictions.csv - OOF预测结果")
    print("  4. feature_importance.csv - 特征重要性")


if __name__ == "__main__":
    main()