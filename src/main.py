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
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "max_depth": 18,
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "verbosity": 1,
        "seed": 114514,
    }
    
    # 训练
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1500,
        callbacks=[
        lgb.log_evaluation(50),  # 每50轮打印一次验证指标
        lgb.early_stopping(50),  # 可选：早停机制
    ],
    )
    
    return model

def make_submission(model, X_test, train, test, code, threshold=0.5, cover=False):
    """生成预测结果并保存，正确处理train和test中的重复did"""
    # 预测概率
    preds = model.predict(X_test)
    
    # 初始化预测结果
    predictions = (preds > threshold).astype(int) if threshold is not None else preds

    if cover:
        # 创建提交DataFrame，保留did和common_ts列用于匹配
        submission = pd.DataFrame({
            "did": test["did"],
            "common_ts": test["common_ts"],
            "is_new_did": predictions
        })
        
        # 按did分组train数据，便于后续查找
        train_groups = train.groupby("did")
        
        # 记录覆盖的行数
        overridden_rows = 0
        
        for idx, row in submission.iterrows():
            did = row["did"]
            ts = row["common_ts"]
            original_pred = row["is_new_did"]
            
            if did in train_groups.groups:
                # 获取该did在train中的所有记录
                train_records = train_groups.get_group(did)
                
                # 检查是否存在时间戳<=test且is_new_did=0的记录
                condition1 = (train_records["common_ts"] <= ts) & (train_records["is_new_did"] == 0)
                # 检查是否存在时间戳>=test且is_new_did=1的记录
                condition2 = (train_records["common_ts"] >= ts) & (train_records["is_new_did"] == 1)
                
                if condition1.any():
                    # 满足条件1，设置为0
                    submission.at[idx, "is_new_did"] = 0
                    if original_pred != 0:
                        overridden_rows += 1
                elif condition2.any():
                    # 满足条件2，设置为1
                    submission.at[idx, "is_new_did"] = 1
                    if original_pred != 1:
                        overridden_rows += 1
        
        print(f"覆盖了 {overridden_rows} 条记录")
        
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
    saved_path = make_submission(model, X_test, train, test, submission_code, threshold = 0.45, cover= True)
    
    # 打印特征重要性
    lgb.plot_importance(model, max_num_features=20, figsize=(10, 6))
    plt.show()