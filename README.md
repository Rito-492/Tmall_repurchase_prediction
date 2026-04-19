# 重复购买预测系统

> 比赛链接：[天猫复购预测-挑战Baseline](https://tianchi.aliyun.com/competition/entrance/231576)
> 实验所需数据集请在比赛官网下载，并放入 `data/` 目录下。

预测"双十一"期间新客户在未来 6 个月内成为忠实客户（重复购买）的概率。

**特点**：特征工程与模型训练分离，特征只需计算一次，后续训练/预测直接加载。

## 项目结构

```
├── main.py              # 主入口
├── scripts/             # 运行脚本
├── src/                 # 源代码
├── data/                # 原始数据
├── outputs/features/    # 特征文件
├── checkpoints/         # 模型检查点
└── results/             # 预测结果
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 完整流程：特征工程 + 模型训练 + 预测
python main.py full

# 仅训练模型
python main.py train

# 仅预测
python main.py predict
```

或使用脚本：

```bash
python scripts/run_pipeline.py --mode full
```

## 输出文件

| 目录 | 文件 | 说明 |
|------|------|------|
| `outputs/features/` | `train_features.csv` | 训练集特征 |
| `checkpoints/` | `model_cv.pkl` | 交叉验证模型 |
| `results/` | `prediction.csv` | 提交文件 |

## 提示

- 特征工程只需运行一次
- 推荐使用交叉验证模型（`model_cv.pkl`）
- 查看 `results/feature_importance.csv` 了解重要特征
