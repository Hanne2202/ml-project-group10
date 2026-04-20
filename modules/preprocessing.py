import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Xóa cột không cần thiết."""
    df = df.drop(columns=columns)
    print(f"Dropped columns: {columns}")
    return df

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa hàng có giá trị thiếu (NaN hoặc '?')."""
    initial_shape = df.shape
    df = df.replace('?', np.nan).dropna()
    print(f"Dropped rows with missing values: {initial_shape[0] - df.shape[0]}")
    return df

def encode_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Mã hóa cột phân loại bằng one-hot encoding."""
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"Encoded categorical columns: {cat_cols}")
    return df

def map_target_variable(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Mã hóa biến mục tiêu thành 0 và 1."""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return df
    df[target_col] = df[target_col].map({'<=50K': 0, '>50K': 1})
    print(f"Mapped target variable '{target_col}' to binary.")
    return df



def apply_ohe(df: pd.DataFrame, cat_cols: list, encoder: OneHotEncoder = None):
    """Chuẩn hoá các cột categories bằng OneHotEncoder"""
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[cat_cols])
    else:
        encoded_data = encoder.transform(df[cat_cols])
        
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    
    # Xóa các cột gốc và nối các cột đã OHE vào
    df_out = df.drop(columns=cat_cols)
    df_out = pd.concat([df_out, encoded_df], axis=1)
    
    print(f"Applied OHE on {len(cat_cols)} columns, resulted in {len(encoded_cols)} new columns.")
    return df_out, encoder


def scale_numeric(df: pd.DataFrame, num_cols: list, scaler: StandardScaler = None):
    """Chuẩn hoá các cột số bằng StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    print(f"Scaled {len(num_cols)} numeric columns: {num_cols}")
    return df, scaler


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """Chia dữ liệu thành tập Train / Val / Test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Tách test trước
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Tách val từ phần còn lại
    # val_size tương đối so với X_temp để đảm bảo đúng tỷ lệ tuyệt đối
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )

    print(f"Split data → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Class dist (train) >50K: {y_train.mean():.1%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str | float = 'auto',
    k_neighbors: int = 5,
    random_state: int = 42
):
    """
    Áp dụng SMOTE để cân bằng lớp trên tập train.

    Chỉ áp dụng trên tập train để tránh data leakage sang val/test.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix của tập train (đã được encode & scale).
    y_train : pd.Series
        Nhãn của tập train.
    sampling_strategy : str or float, default='auto'
        - 'auto'   : cân bằng lớp thiểu số lên bằng lớp đa số.
        - float    : tỷ lệ minority/majority mong muốn (vd: 0.5 → 1:2).
    k_neighbors : int, default=5
        Số láng giềng gần nhất dùng khi sinh mẫu tổng hợp.
    random_state : int, default=42
        Seed để tái lập kết quả.

    Returns
    -------
    X_res : pd.DataFrame
        Feature matrix sau SMOTE.
    y_res : pd.Series
        Nhãn sau SMOTE.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Giữ nguyên tên cột và kiểu dữ liệu
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    before = y_train.value_counts().to_dict()
    after  = y_res.value_counts().to_dict()
    print(f"SMOTE applied (strategy='{sampling_strategy}', k={k_neighbors})")
    print(f"  Before → {before}  |  total={len(y_train)}")
    print(f"  After  → {after}   |  total={len(y_res)}")
    print(f"  Class dist after SMOTE >50K: {y_res.mean():.1%}")
    return X_res, y_res