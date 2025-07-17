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
        # 创建提交DataFrame
        submission = pd.DataFrame({
            "did": test["did"],
            "common_ts": test["common_ts"],
            "is_new_did": predictions
        })
        
        # 预处理：为train添加标记列
        train = train.copy()
        train['is_zero_condition'] = (train['is_new_did'] == 0).astype(int)
        train['is_one_condition'] = (train['is_new_did'] == 1).astype(int)
        
        # 对每个did，找出最小的时间戳满足is_new_did=0（用于条件1）
        condition1 = train[train['is_new_did'] == 0].groupby('did')['common_ts'].max().reset_index()
        condition1.columns = ['did', 'max_ts_for_zero']
        
        # 对每个did，找出最大的时间戳满足is_new_did=1（用于条件2）
        condition2 = train[train['is_new_did'] == 1].groupby('did')['common_ts'].min().reset_index()
        condition2.columns = ['did', 'min_ts_for_one']
        
        # 合并条件到submission
        submission = submission.merge(condition1, on='did', how='left')
        submission = submission.merge(condition2, on='did', how='left')
        
        # 应用条件1：存在时间戳<=test且is_new_did=0
        mask_condition1 = (submission['common_ts'] >= submission['max_ts_for_zero']) & ~submission['max_ts_for_zero'].isna()
        
        # 应用条件2：存在时间戳>=test且is_new_did=1 (且不满足条件1)
        mask_condition2 = (submission['common_ts'] <= submission['min_ts_for_one']) & ~submission['min_ts_for_one'].isna() & ~mask_condition1
        
        # 记录原始预测用于统计
        original_preds = submission['is_new_did'].copy()
        
        # 应用覆盖规则
        submission.loc[mask_condition1, 'is_new_did'] = 0
        submission.loc[mask_condition2, 'is_new_did'] = 1
        
        # 计算覆盖行数
        overridden_rows = ((mask_condition1 | mask_condition2) & 
                          (submission['is_new_did'] != original_preds)).sum()
        
        print(f"覆盖了 {overridden_rows} 条记录")
        
        # 保存文件
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