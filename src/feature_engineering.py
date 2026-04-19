"""
重复购买预测 - 特征工程模块（特征保存版本）
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self, feature_dir='./outputs/features/'):
        self.user_features = None
        self.merchant_features = None
        self.feature_dir = feature_dir
        
        # 创建特征保存目录
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
            print(f"创建特征保存目录: {feature_dir}")
        
    def load_data(self, data_path='./data/'):
        """加载所有数据文件"""
        print("正在加载数据...")
        
        # 加载用户行为日志
        self.user_log = pd.read_csv(f'{data_path}user_log_format1.csv')
        
        # 加载用户画像
        self.user_info = pd.read_csv(f'{data_path}user_info_format1.csv')
        
        # 加载训练数据
        self.train_data = pd.read_csv(f'{data_path}train_format1.csv')
        
        # 加载测试数据
        self.test_data = pd.read_csv(f'{data_path}test_format1.csv')
        
        print(f"用户行为日志: {self.user_log.shape}")
        print(f"用户画像: {self.user_info.shape}")
        print(f"训练数据: {self.train_data.shape}")
        print(f"测试数据: {self.test_data.shape}")
        
        return self
    
    def create_user_features(self):
        """创建用户特征"""
        print("\n正在构建用户特征...")
        
        user_log = self.user_log.copy()
        
        # 按用户聚合特征
        user_features = []
        
        # 1. 基础行为统计
        user_basic = user_log.groupby('user_id').agg({
            'item_id': 'count',  # 总行为次数
            'merchant_id': 'nunique',  # 访问商家数
            'brand_id': 'nunique',  # 访问品牌数
            'cat_id': 'nunique',  # 访问品类数
        }).reset_index()
        
        user_basic.columns = ['user_id', 'user_total_actions', 'user_merchant_cnt', 
                              'user_brand_cnt', 'user_cat_cnt']
        user_features.append(user_basic)
        
        # 2. 各类行为统计
        for action in [0, 1, 2, 3]:
            action_df = user_log[user_log['action_type'] == action].groupby('user_id').size().reset_index()
            action_df.columns = ['user_id', f'user_action_{action}_cnt']
            user_features.append(action_df)
        
        # 3. 购买转化率
        user_buy = user_log[user_log['action_type'] == 2].groupby('user_id').size().reset_index()
        user_buy.columns = ['user_id', 'user_buy_cnt']
        user_features.append(user_buy)
        
        # 4. 时间特征
        user_log['time_stamp'] = user_log['time_stamp'].astype(str).str.zfill(4)
        user_log['month'] = user_log['time_stamp'].str[:2].astype(int)
        user_log['day'] = user_log['time_stamp'].str[2:].astype(int)
        
        # 用户活跃天数
        user_active_days = user_log.groupby('user_id')['time_stamp'].nunique().reset_index()
        user_active_days.columns = ['user_id', 'user_active_days']
        user_features.append(user_active_days)
        
        # 用户首次和最后一次行为时间
        user_time = user_log.groupby('user_id')['time_stamp'].agg(['min', 'max']).reset_index()
        user_time.columns = ['user_id', 'user_first_time', 'user_last_time']
        user_features.append(user_time)
        
        # 合并所有用户特征
        user_feat = user_features[0]
        for feat in user_features[1:]:
            user_feat = user_feat.merge(feat, on='user_id', how='left')
        
        # 填充缺失值
        user_feat = user_feat.fillna(0)
        
        # 计算衍生特征
        user_feat['user_buy_ratio'] = user_feat['user_buy_cnt'] / (user_feat['user_total_actions'] + 1)
        user_feat['user_avg_actions_per_day'] = user_feat['user_total_actions'] / (user_feat['user_active_days'] + 1)
        
        # 合并用户画像
        user_feat = user_feat.merge(self.user_info, on='user_id', how='left')
        
        self.user_features = user_feat
        print(f"用户特征维度: {user_feat.shape}")
        
        return user_feat
    
    def create_merchant_features(self):
        """创建商家特征"""
        print("\n正在构建商家特征...")
        
        user_log = self.user_log.copy()
        
        merchant_features = []
        
        # 1. 商家基础统计
        merchant_basic = user_log.groupby('merchant_id').agg({
            'user_id': 'nunique',  # 访问用户数
            'item_id': ['count', 'nunique'],  # 总行为数和商品数
            'brand_id': 'nunique',  # 品牌数
            'cat_id': 'nunique',  # 品类数
        }).reset_index()
        
        merchant_basic.columns = ['merchant_id', 'merchant_user_cnt', 'merchant_total_actions',
                                  'merchant_item_cnt', 'merchant_brand_cnt', 'merchant_cat_cnt']
        merchant_features.append(merchant_basic)
        
        # 2. 商家各类行为统计
        for action in [0, 1, 2, 3]:
            action_df = user_log[user_log['action_type'] == action].groupby('merchant_id').size().reset_index()
            action_df.columns = ['merchant_id', f'merchant_action_{action}_cnt']
            merchant_features.append(action_df)
        
        # 3. 商家购买转化率
        merchant_buy = user_log[user_log['action_type'] == 2].groupby('merchant_id').agg({
            'user_id': 'nunique'
        }).reset_index()
        merchant_buy.columns = ['merchant_id', 'merchant_buy_user_cnt']
        merchant_features.append(merchant_buy)
        
        # 合并所有商家特征
        merchant_feat = merchant_features[0]
        for feat in merchant_features[1:]:
            merchant_feat = merchant_feat.merge(feat, on='merchant_id', how='left')
        
        merchant_feat = merchant_feat.fillna(0)
        
        # 计算衍生特征
        merchant_feat['merchant_buy_ratio'] = merchant_feat['merchant_buy_user_cnt'] / (merchant_feat['merchant_user_cnt'] + 1)
        merchant_feat['merchant_avg_actions_per_user'] = merchant_feat['merchant_total_actions'] / (merchant_feat['merchant_user_cnt'] + 1)
        
        self.merchant_features = merchant_feat
        print(f"商家特征维度: {merchant_feat.shape}")
        
        return merchant_feat
    
    def create_user_merchant_features_fast(self, data):
        print("\n正在高效构建用户-商家交互特征...")

        user_log = self.user_log.copy()
        # 一次性 groupby 聚合
        um_agg = user_log.groupby(['user_id', 'merchant_id']).agg(
            um_total_actions=('item_id', 'count'),
            um_action_0_cnt=('action_type', lambda x: (x==0).sum()),
            um_action_1_cnt=('action_type', lambda x: (x==1).sum()),
            um_action_2_cnt=('action_type', lambda x: (x==2).sum()),
            um_action_3_cnt=('action_type', lambda x: (x==3).sum()),
            um_item_cnt=('item_id', 'nunique'),
            um_cat_cnt=('cat_id', 'nunique'),
            um_brand_cnt=('brand_id', 'nunique'),
            um_active_days=('time_stamp', 'nunique')
        ).reset_index()

        # 衍生特征
        um_agg['um_buy_ratio'] = um_agg['um_action_2_cnt'] / (um_agg['um_total_actions'] + 1)
        um_agg['um_cart_ratio'] = um_agg['um_action_1_cnt'] / (um_agg['um_total_actions'] + 1)
        um_agg['um_fav_ratio'] = um_agg['um_action_3_cnt'] / (um_agg['um_total_actions'] + 1)

        # 合并
        result = data.merge(um_agg, on=['user_id', 'merchant_id'], how='left').fillna(0)
        print(f"✓ 用户-商家交互特征维度: {result.shape}")
        return result


    def create_user_merchant_features(self, data):
        """创建用户-商家交互特征"""
        print("\n正在构建用户-商家交互特征...")
        
        user_log = self.user_log.copy()
        
        # 为每个user_id-merchant_id对创建特征
        um_features = []
        
        for idx, row in data.iterrows():
            user_id = row['user_id']
            merchant_id = row['merchant_id']
            
            # 筛选该用户和该商家的交互记录
            um_log = user_log[(user_log['user_id'] == user_id) & 
                              (user_log['merchant_id'] == merchant_id)]
            
            if len(um_log) == 0:
                # 没有交互记录
                um_feat = {
                    'user_id': user_id,
                    'merchant_id': merchant_id,
                    'um_total_actions': 0,
                    'um_action_0_cnt': 0,
                    'um_action_1_cnt': 0,
                    'um_action_2_cnt': 0,
                    'um_action_3_cnt': 0,
                    'um_item_cnt': 0,
                    'um_cat_cnt': 0,
                    'um_brand_cnt': 0,
                    'um_active_days': 0,
                }
            else:
                # 有交互记录
                um_feat = {
                    'user_id': user_id,
                    'merchant_id': merchant_id,
                    'um_total_actions': len(um_log),
                    'um_action_0_cnt': len(um_log[um_log['action_type'] == 0]),
                    'um_action_1_cnt': len(um_log[um_log['action_type'] == 1]),
                    'um_action_2_cnt': len(um_log[um_log['action_type'] == 2]),
                    'um_action_3_cnt': len(um_log[um_log['action_type'] == 3]),
                    'um_item_cnt': um_log['item_id'].nunique(),
                    'um_cat_cnt': um_log['cat_id'].nunique(),
                    'um_brand_cnt': um_log['brand_id'].nunique(),
                    'um_active_days': um_log['time_stamp'].nunique(),
                }
            
            um_features.append(um_feat)
            
            if (idx + 1) % 1000 == 0:
                print(f"已处理 {idx + 1}/{len(data)} 条记录")
        
        um_feat_df = pd.DataFrame(um_features)
        
        # 计算衍生特征
        um_feat_df['um_buy_ratio'] = um_feat_df['um_action_2_cnt'] / (um_feat_df['um_total_actions'] + 1)
        um_feat_df['um_cart_ratio'] = um_feat_df['um_action_1_cnt'] / (um_feat_df['um_total_actions'] + 1)
        um_feat_df['um_fav_ratio'] = um_feat_df['um_action_3_cnt'] / (um_feat_df['um_total_actions'] + 1)
        
        print(f"用户-商家交互特征维度: {um_feat_df.shape}")
        
        return um_feat_df
    
    def build_and_save_features(self):
        """
        构建所有特征并保存到文件
        这个函数只需要运行一次，特征会保存到 feature_dir 目录
        """
        print(f"\n{'='*70}")
        print("开始构建并保存所有特征")
        print(f"{'='*70}")
        
        # 1. 创建用户特征
        user_features = self.create_user_features()
        user_feat_path = os.path.join(self.feature_dir, 'outputs/features/user_features.csv')
        user_features.to_csv(user_feat_path, index=False)
        print(f"✓ 用户特征已保存: {user_feat_path}")
        
        # 2. 创建商家特征
        merchant_features = self.create_merchant_features()
        merchant_feat_path = os.path.join(self.feature_dir, 'outputs/features/merchant_features.csv')
        merchant_features.to_csv(merchant_feat_path, index=False)
        print(f"✓ 商家特征已保存: {merchant_feat_path}")
        
        # 3. 创建训练集的用户-商家交互特征
        print("\n" + "="*70)
        print("构建训练集特征")
        print("="*70)
        train_um_features = self.create_user_merchant_features_fast(self.train_data)
        
        # 合并训练集的所有特征
        train_features = self.train_data[['user_id', 'merchant_id', 'label']].copy()
        train_features = train_features.merge(user_features, on='user_id', how='left')
        train_features = train_features.merge(merchant_features, on='merchant_id', how='left')
        train_features = train_features.merge(train_um_features, on=['user_id', 'merchant_id'], how='left')
        train_features = train_features.fillna(0)
        
        train_feat_path = os.path.join(self.feature_dir, 'outputs/features/train_features.csv')
        train_features.to_csv(train_feat_path, index=False)
        print(f"\n✓ 训练集特征已保存: {train_feat_path}")
        print(f"  特征维度: {train_features.shape}")
        
        # 4. 创建测试集的用户-商家交互特征
        print("\n" + "="*70)
        print("构建测试集特征")
        print("="*70)
        test_um_features = self.create_user_merchant_features_fast(self.test_data)
        
        # 合并测试集的所有特征
        test_features = self.test_data[['user_id', 'merchant_id']].copy()
        test_features = test_features.merge(user_features, on='user_id', how='left')
        test_features = test_features.merge(merchant_features, on='merchant_id', how='left')
        test_features = test_features.merge(test_um_features, on=['user_id', 'merchant_id'], how='left')
        test_features = test_features.fillna(0)
        
        test_feat_path = os.path.join(self.feature_dir, 'outputs/features/test_features.csv')
        test_features.to_csv(test_feat_path, index=False)
        print(f"\n✓ 测试集特征已保存: {test_feat_path}")
        print(f"  特征维度: {test_features.shape}")
        
        # 5. 保存特征列名（用于后续训练）
        feature_cols = [col for col in train_features.columns 
                       if col not in ['user_id', 'merchant_id', 'label']]
        feature_info = {
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'train_shape': train_features.shape,
            'test_shape': test_features.shape
        }
        
        import json
        info_path = os.path.join(self.feature_dir, 'outputs/features/feature_info.json')
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"\n✓ 特征信息已保存: {info_path}")
        
        print(f"\n{'='*70}")
        print("特征构建完成！")
        print(f"{'='*70}")
        print(f"\n所有特征已保存到目录: {self.feature_dir}")
        print("生成的文件:")
        print(f"  1. user_features.csv - 用户特征")
        print(f"  2. merchant_features.csv - 商家特征")
        print(f"  3. train_features.csv - 训练集完整特征")
        print(f"  4. test_features.csv - 测试集完整特征")
        print(f"  5. feature_info.json - 特征信息")
        
        return train_features, test_features
    
    @staticmethod
    def load_features(feature_dir='./outputs/features/'):
        """
        从保存的文件中加载特征
        这个函数用于训练和预测时快速加载特征
        
        Returns:
            train_features: 训练集特征 DataFrame
            test_features: 测试集特征 DataFrame
            feature_info: 特征信息字典
        """
        print(f"\n从 {feature_dir} 加载已保存的特征...")
        
        # 加载训练集特征
        train_feat_path = os.path.join(feature_dir, 'outputs/features/train_features.csv')
        if not os.path.exists(train_feat_path):
            raise FileNotFoundError(f"训练集特征文件不存在: {train_feat_path}")
        train_features = pd.read_csv(train_feat_path)
        print(f"✓ 训练集特征加载完成: {train_features.shape}")
        
        # 加载测试集特征
        test_feat_path = os.path.join(feature_dir, 'outputs/features/test_features.csv')
        if not os.path.exists(test_feat_path):
            raise FileNotFoundError(f"测试集特征文件不存在: {test_feat_path}")
        test_features = pd.read_csv(test_feat_path)
        print(f"✓ 测试集特征加载完成: {test_features.shape}")
        
        # 加载特征信息
        import json
        info_path = os.path.join(feature_dir, 'outputs/features/feature_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                feature_info = json.load(f)
            print(f"✓ 特征信息加载完成: {feature_info['n_features']} 个特征")
        else:
            feature_info = None
            print("⚠ 特征信息文件不存在")
        
        return train_features, test_features, feature_info


def main():
    """构建并保存特征"""
    print("="*70)
    print("特征工程 - 构建并保存特征")
    print("="*70)
    
    # 创建特征工程实例
    fe = FeatureEngineering(feature_dir='./outputs/features/')
    
    # 加载原始数据
    fe.load_data('./data/')
    
    # 构建并保存所有特征
    train_features, test_features = fe.build_and_save_features()
    
    print("\n" + "="*70)
    print("特征工程完成！后续可以直接使用 FeatureEngineering.load_features() 加载特征")
    print("="*70)


if __name__ == "__main__":
    main()