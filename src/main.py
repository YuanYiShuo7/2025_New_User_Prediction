import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from utils import process_data
import uuid

# 配置路径
DATA_DIR = "../data"
SUBMISSION_DIR = "../submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 生成随机代码用于提交文件名
submission_code = str(uuid.uuid4())[:8]  # 取UUID前8位作为唯一标识

def load_data():
    """加载并预处理数据"""
    # 训练数据
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    X_train = process_data(train.drop(columns=["is_new_did"]))
    y_train = train["is_new_did"]
    
    # 测试数据
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    X_test = process_data(test)
    
    return X_train, y_train, X_test

def train_model(X, y):
    """训练LightGBM模型"""
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 参数配置
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": 42,
    }
    
    # 训练
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    
    return model

def make_submission(model, X_test, code):
    """生成预测结果并保存"""
    # 预测概率
    preds = model.predict(X_test)
    
    # 创建提交DataFrame
    submission = pd.DataFrame({
        "is_new_did": preds  # 概率预测
        # 如果比赛要求二值结果: "is_new_did": (preds > 0.5).astype(int)
    })
    
    # 保存文件
    submission_path = f"{SUBMISSION_DIR}/submit_{code}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    
    return submission_path

if __name__ == "__main__":
    # 数据加载与处理
    X_train, y_train, X_test = load_data()
    
    # 模型训练
    model = train_model(X_train, y_train)
    
    # 生成提交
    saved_path = make_submission(model, X_test, submission_code)
    
    # 打印特征重要性
    lgb.plot_importance(model, max_num_features=20, figsize=(10, 6))