import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def process_data(raw_df):
    """
    数据预处理全流程函数
    输入: 原始DataFrame (包含所有字段)
    输出: 处理后的DataFrame (仅含特征列)
    """
    df = raw_df.copy()
    
    # 1. 移除用户实际ID (did列)
    df = df.drop(columns=['did'])
    
    # 2. 时间戳特征工程
    df = _process_timestamp(df)
    
    # 3. 解析udmap中的JSON字段
    df = _process_udmap(df)
    
    # 4. 类别特征编码
    df = _encode_categorical(df)
    
    # 5. 数值特征归一化
    df = _normalize_features(df)
    
    return df

def _process_timestamp(df):
    """时间戳特征提取"""
    # 转换为datetime对象
    df['datetime'] = pd.to_datetime(df['common_ts'], unit='ms')
    
    # 提取时间特征
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # 计算时间相关衍生特征
    df['days_since_first'] = (df['datetime'] - df.groupby('mid')['datetime'].transform('min')).dt.days
    df['days_since_last'] = (df['datetime'] - df.groupby('mid')['datetime'].transform('max')).dt.days
    
    # 移除原始时间列
    df = df.drop(columns=['common_ts', 'datetime'])
    
    return df

def _process_udmap(df):
    """解析udmap字段"""
    # 1. 处理空值并统一为字符串
    df['udmap'] = df['udmap'].fillna('{}').astype(str)
    
    # 2. 标准化JSON格式（单引号转双引号，移除首尾空格）
    df['udmap'] = df['udmap'].str.strip().apply(
        lambda x: x.replace("'", '"') if x.startswith('{') and x.endswith('}') else x
    )
    
    # 3. 直接解析JSON
    udmap_data = df['udmap'].apply(json.loads)
    
    # 4. 提取字段
    df['botId'] = udmap_data.apply(lambda x: x.get('botId', 'missing'))
    df['pluginId'] = udmap_data.apply(lambda x: x.get('pluginId', 'missing'))
    
    return df.drop(columns=['udmap'])

def _encode_categorical(df):
    """类别特征编码"""
    cat_cols = ['botId', 'pluginId']
    
    # 确保列存在
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # # 频率编码 (对高基数类别更有效)
    # for col in cat_cols:
    #     freq = df[col].value_counts(normalize=True)
    #     df[col+'_freq'] = df[col].map(freq)
        
    # 标签编码 (保留原始列)
    le = LabelEncoder()
    for col in cat_cols:
        df[col+'_code'] = le.fit_transform(df[col].astype(str))
        df.drop(col, axis=1, inplace=True)
    return df

def _normalize_features(df):
    """数值特征归一化"""
    num_cols = ['botId_code', 'pluginId_code', ]
    num_cols = [col for col in num_cols if col in df.columns]
    
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df