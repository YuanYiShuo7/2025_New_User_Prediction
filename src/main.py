import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from utils import process_data
import uuid
import matplotlib.pyplot as plt

# 配置路径
DATA_DIR = "./data"
SUBMISSION_DIR = "./submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
print("当前工作目录:", os.getcwd())
print("DATA_DIR 绝对路径:", os.path.abspath(f"{DATA_DIR}/train.csv"))

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
    
    return train, test, X_train, y_train, X_test

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
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbosity": 1,
        "seed": 42,
    }
    
    # 训练
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[
        lgb.log_evaluation(50),  # 每50轮打印一次验证指标
        lgb.early_stopping(50),  # 可选：早停机制
    ],
    )
    
    return model

def make_submission(model, X_test, train, test, code, threshold=0.5, cover = False):
    """生成预测结果并保存，正确处理train和test中的重复did"""
    # 预测概率
    preds = model.predict(X_test)
    
    # 初始化预测结果
    predictions = (preds > threshold).astype(int) if threshold is not None else preds

    if cover:
        """生成预测结果并保存，要求did和common_ts同时匹配才覆盖"""
        # 预测概率
        preds = model.predict(X_test)
        
        # 初始化预测结果
        predictions = (preds > threshold).astype(int) if threshold is not None else preds
        
        # 创建提交DataFrame，保留did和common_ts列用于匹配
        submission = pd.DataFrame({
            "did": test["did"],
            "common_ts": test["common_ts"],  # 新增common_ts列
            "is_new_did": predictions
        })
        
        # 创建train的复合键映射
        train_composite_keys = train.set_index(["did", "common_ts"])["is_new_did"].to_dict()
        
        # 创建test的复合键用于匹配
        test_keys = list(zip(submission["did"], submission["common_ts"]))
        
        # 找出需要覆盖的记录
        mask = [key in train_composite_keys for key in test_keys]
        
        # 覆盖匹配记录的预测值
        if any(mask):
            submission.loc[mask, "is_new_did"] = \
                [train_composite_keys[key] for key in test_keys if key in train_composite_keys]
            
            # 统计信息
            num_overridden = sum(mask)
            num_unique_overridden_pairs = len({key for key in test_keys if key in train_composite_keys})
            print(f"覆盖了 {num_overridden} 条记录（涉及 {num_unique_overridden_pairs} 个唯一did+ts组合）")
        
        # 保存文件（不包含did和common_ts列）
        submission_path = f"{SUBMISSION_DIR}/submit_{code}.csv"
        submission[["is_new_did"]].to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")
    
    else:
        submission = pd.DataFrame({"is_new_did": predictions})
        # 保存文件
        submission_path = f"{SUBMISSION_DIR}/submit_{code}.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")

if __name__ == "__main__":
    # 数据加载与处理
    train, test, X_train, y_train, X_test = load_data()
    
    # 模型训练
    model = train_model(X_train, y_train)
    
    # 生成提交
    saved_path = make_submission(model, X_test, train, test, submission_code, threshold= 0.4, cover= True)
    
    # 打印特征重要性
    lgb.plot_importance(model, max_num_features=20, figsize=(10, 6))
    plt.show()