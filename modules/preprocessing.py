"""
modules/preprocessing.py
=========================
Tiền xử lý dữ liệu (Data Preprocessing) — nguồn duy nhất cho toàn bộ pipeline.

Module này phục vụ cả ba pipeline:
  - Classical ML  (02_preprocessing.ipynb, 03_classical_pipeline.ipynb)
  - Deep Learning / MLP  (04_deep_learning.ipynb — phần step-by-step)
  - TabNet  (04_deep_learning.ipynb — phần preprocess_for_tabnet)

Cấu trúc
--------
  Phần 1 — Làm sạch cơ bản
      drop_columns, drop_missing_values, map_target_variable

  Phần 2 — Classical preprocessing (sklearn Pipeline / ColumnTransformer)
      build_classical_preprocessor, compare_preprocessing_configs,
      select_best_preprocessor

  Phần 3 — Deep Learning preprocessing (step-by-step)
      apply_ohe, scale_numeric, split_data, apply_smote

  Phần 4 — Feature engineering dùng chung cho DL
      group_rare_categories, apply_quantile_binning

  Phần 5 — Pipeline tự động cho DL
      preprocess_for_dl, preprocess_for_tabnet

  Phần 6 — Helpers nội bộ
      _compute_balanced_class_weights, _encode_categorical_for_tabnet

Cách dùng
---------
    import modules.preprocessing as prep

    # Classical
    preprocessor = prep.build_classical_preprocessor('config_2', num_features, cat_features)

    # Deep Learning (step-by-step)
    from modules.preprocessing import (
        drop_columns, drop_missing_values, map_target_variable,
        apply_ohe, scale_numeric, split_data, apply_smote,
    )

    # TabNet
    prep_tabnet = prep.preprocess_for_tabnet(df_clean, target_col='income')
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


# ══════════════════════════════════════════════════════════════════════════════
# Phần 1 — Làm sạch cơ bản
# ══════════════════════════════════════════════════════════════════════════════

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Xóa các cột không cần thiết khỏi DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Danh sách tên cột cần xóa.

    Returns
    -------
    pd.DataFrame
    """
    df = df.drop(columns=columns, errors='ignore')
    print(f"Dropped columns: {columns}")
    return df


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa '?' về NaN rồi xóa toàn bộ hàng có missing value.

    Dùng cho Deep Learning pipeline (drop thay vì impute).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    initial_shape = df.shape
    df = df.replace('?', np.nan).dropna()
    print(f"Dropped rows with missing values: {initial_shape[0] - df.shape[0]}")
    return df


def map_target_variable(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Mã hóa biến mục tiêu dạng chuỗi thành nhị phân 0 / 1.
      '<=50K' → 0,  '>50K' → 1

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Tên cột target.

    Returns
    -------
    pd.DataFrame
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return df
    df[target_col] = df[target_col].map({'<=50K': 0, '>50K': 1})
    print(f"Mapped target variable '{target_col}' to binary (0 / 1).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Phần 2 — Classical preprocessing (sklearn Pipeline / ColumnTransformer)
# ══════════════════════════════════════════════════════════════════════════════

# Tập hợp các cấu hình preprocessing cho classical pipeline.
# Key: tên config, Value: dict mô tả tham số.
_CLASSICAL_CONFIGS = {
    'config_1_onehot_mostfreq_standard': dict(
        scaler=StandardScaler(),
        impute_strategy='most_frequent',
        fill_value=None,
    ),
    'config_2_onehot_constant_standard': dict(
        scaler=StandardScaler(),
        impute_strategy='constant',
        fill_value='Missing',
    ),
    'config_3_onehot_mostfreq_minmax': dict(
        scaler=MinMaxScaler(),
        impute_strategy='most_frequent',
        fill_value=None,
    ),
    'config_4_onehot_mostfreq_noscale': dict(
        scaler='passthrough',
        impute_strategy='most_frequent',
        fill_value=None,
    ),
}


def build_classical_preprocessor(
    config_name: str,
    num_features: list,
    cat_features: list,
) -> ColumnTransformer:
    """
    Tạo một ColumnTransformer theo cấu hình đã chọn.

    Parameters
    ----------
    config_name : str
        Một trong các key của _CLASSICAL_CONFIGS:
        'config_1_onehot_mostfreq_standard'
        'config_2_onehot_constant_standard'
        'config_3_onehot_mostfreq_minmax'
        'config_4_onehot_mostfreq_noscale'
    num_features : list
        Danh sách tên cột số.
    cat_features : list
        Danh sách tên cột categorical.

    Returns
    -------
    ColumnTransformer (chưa fit)
    """
    if config_name not in _CLASSICAL_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(_CLASSICAL_CONFIGS.keys())}"
        )

    cfg = _CLASSICAL_CONFIGS[config_name]
    imputer_kwargs = {'strategy': cfg['impute_strategy']}
    if cfg['fill_value'] is not None:
        imputer_kwargs['fill_value'] = cfg['fill_value']

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(**imputer_kwargs)),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', cfg['scaler'], num_features),
        ('cat', cat_pipeline, cat_features),
    ])

    return preprocessor


def get_all_classical_configs(
    num_features: list,
    cat_features: list,
) -> dict:
    """
    Tạo toàn bộ 4 cấu hình preprocessing classical để so sánh.

    Returns
    -------
    dict[str, ColumnTransformer]
    """
    return {
        name: build_classical_preprocessor(name, num_features, cat_features)
        for name in _CLASSICAL_CONFIGS
    }


def compare_preprocessing_configs(
    preprocessing_configs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pos_label: str = '>50K',
) -> pd.DataFrame:
    """
    So sánh các cấu hình preprocessing bằng Logistic Regression.

    Mỗi cấu hình được fit trên X_train rồi đánh giá trên X_test.
    Trả về bảng kết quả sắp xếp theo F1-score giảm dần.

    Parameters
    ----------
    preprocessing_configs : dict[str, ColumnTransformer]
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    pos_label : str
        Nhãn dương tính dùng để tính precision / recall / f1.

    Returns
    -------
    pd.DataFrame
        Cột: configuration, accuracy, precision, recall, f1_score,
             n_features_after_preprocessing.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    results = []
    for name, transformer in preprocessing_configs.items():
        X_tr = transformer.fit_transform(X_train)
        X_te = transformer.transform(X_test)

        model = LogisticRegression(
            random_state=42, max_iter=5000, solver='lbfgs'
        )
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        results.append({
            'configuration': name,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=pos_label),
            'recall':    recall_score(y_test, y_pred, pos_label=pos_label),
            'f1_score':  f1_score(y_test, y_pred, pos_label=pos_label),
            'n_features_after_preprocessing': X_tr.shape[1],
        })

    return (
        pd.DataFrame(results)
        .sort_values(by='f1_score', ascending=False)
        .reset_index(drop=True)
    )


def select_best_preprocessor(
    preprocessing_configs: dict,
    results_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Chọn cấu hình tốt nhất theo F1-score và fit lại trên toàn bộ X_train.

    Parameters
    ----------
    preprocessing_configs : dict[str, ColumnTransformer]
    results_df : pd.DataFrame
        Output của compare_preprocessing_configs.
    X_train, X_test : pd.DataFrame

    Returns
    -------
    (best_name, best_preprocessor, X_train_best, X_test_best)
    """
    best_name = results_df.iloc[0]['configuration']
    best_preprocessor = preprocessing_configs[best_name]

    X_train_best = best_preprocessor.fit_transform(X_train)
    X_test_best  = best_preprocessor.transform(X_test)

    print(f"Best config selected : {best_name}")
    print(f"X_train_best shape   : {X_train_best.shape}")
    print(f"X_test_best shape    : {X_test_best.shape}")

    return best_name, best_preprocessor, X_train_best, X_test_best


# ══════════════════════════════════════════════════════════════════════════════
# Phần 3 — Deep Learning preprocessing (step-by-step)
# ══════════════════════════════════════════════════════════════════════════════

def apply_ohe(
    df: pd.DataFrame,
    cat_cols: list,
    encoder: OneHotEncoder = None,
):
    """
    One-Hot Encoding các cột categorical.

    Nếu encoder=None: fit_transform trên df (dùng cho train).
    Nếu encoder được truyền vào: chỉ transform (dùng cho val/test).

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list
    encoder : OneHotEncoder hoặc None

    Returns
    -------
    (df_out, encoder)
    """
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[cat_cols])
    else:
        encoded_data = encoder.transform(df[cat_cols])

    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)

    df_out = df.drop(columns=cat_cols)
    df_out = pd.concat([df_out, encoded_df], axis=1)

    print(f"Applied OHE on {len(cat_cols)} columns → {len(encoded_cols)} new columns.")
    return df_out, encoder


def scale_numeric(
    df: pd.DataFrame,
    num_cols: list,
    scaler: StandardScaler = None,
):
    """
    Chuẩn hóa các cột số bằng StandardScaler.

    Nếu scaler=None: fit_transform (dùng cho train).
    Nếu scaler được truyền vào: chỉ transform (dùng cho val/test).

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list
    scaler : StandardScaler hoặc None

    Returns
    -------
    (df, scaler)
    """
    df = df.copy()
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
    random_state: int = 42,
):
    """
    Chia dữ liệu thành Train / Val / Test với stratify theo target.

    Thứ tự tách:
      1. Tách test ra khỏi toàn bộ (tỉ lệ test_size).
      2. Tách val ra khỏi phần còn lại (tỉ lệ tuyệt đối val_size).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    test_size : float, default=0.2
    val_size : float, default=0.1
        Tỉ lệ tuyệt đối của val so với toàn bộ dataset.
    random_state : int, default=42

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # val_size tương đối so với X_temp
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Class dist (train) >50K: {y_train.mean():.1%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str | float = 'auto',
    k_neighbors: int = 5,
    random_state: int = 42,
):
    """
    Áp dụng SMOTE (Synthetic Minority Over-sampling Technique) để cân bằng lớp
    trên tập train.

    Chỉ áp dụng trên tập train để tránh data leakage sang val/test.
    Gọi sau khi đã encode & scale (SMOTE cần không gian số liên tục).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix của tập train (đã encode & scale).
    y_train : pd.Series
    sampling_strategy : str or float, default='auto'
        'auto' → cân bằng minority lên bằng majority.
        float  → tỉ lệ minority/majority mong muốn (vd: 0.5 → 1:2).
    k_neighbors : int, default=5
        Số láng giềng gần nhất khi sinh mẫu tổng hợp.
    random_state : int, default=42

    Returns
    -------
    (X_res, y_res) : (pd.DataFrame, pd.Series)
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    before = y_train.value_counts().to_dict()
    after  = y_res.value_counts().to_dict()
    print(f"SMOTE applied (strategy='{sampling_strategy}', k={k_neighbors})")
    print(f"  Before → {before}  |  total={len(y_train)}")
    print(f"  After  → {after}   |  total={len(y_res)}")
    print(f"  Class dist after SMOTE >50K: {y_res.mean():.1%}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════════════════════
# Phần 4 — Feature engineering dùng chung cho DL
# ══════════════════════════════════════════════════════════════════════════════

def group_rare_categories(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list,
    threshold: float = 0.01,
):
    """
    Gom các category xuất hiện ít hơn ngưỡng threshold thành nhãn 'Other'.

    Fit trên train (tính tần suất), transform đồng nhất cho val/test
    để tránh data leakage.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
    cat_features : list
    threshold : float, default=0.01
        Tần suất tương đối tối thiểu để giữ lại một category.

    Returns
    -------
    (X_train, X_val, X_test, rare_mapping)
        rare_mapping : dict[str, set] — tập rare categories của mỗi cột.
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()
    rare_mapping = {}

    for col in cat_features:
        freq = X_train[col].value_counts(normalize=True)
        rare_cats = set(freq[freq < threshold].index)
        rare_mapping[col] = rare_cats

        if rare_cats:
            for split in [X_train, X_val, X_test]:
                split[col] = split[col].apply(
                    lambda v: 'Other' if v in rare_cats else v
                )

    n_removed = sum(len(v) for v in rare_mapping.values())
    print(f"Rare category grouping (threshold={threshold}):")
    for col, cats in rare_mapping.items():
        if cats:
            print(f"  {col}: {len(cats)} rare → 'Other'  "
                  f"(kept {X_train[col].nunique()} categories)")
    print(f"  Total rare categories grouped: {n_removed}")

    return X_train, X_val, X_test, rare_mapping


def apply_quantile_binning(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    binning_cols: tuple = ('age', 'hours-per-week'),
    n_bins: int = 4,
    drop_original_numeric: bool = True,
):
    """
    Tạo cột binning theo quantile dựa trên tập train để tránh leakage.

    Bin edges được tính từ train, sau đó áp dụng lại cho val/test.
    Biên ngoài cùng được mở rộng ra ±inf để xử lý giá trị ngoài khoảng train.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
    binning_cols : tuple
        Các cột số cần binning.
    n_bins : int, default=4
        Số lượng bin.
    drop_original_numeric : bool, default=True
        Nếu True, xóa cột gốc sau khi tạo cột bin.

    Returns
    -------
    (X_train, X_val, X_test, binning_info)
        binning_info : dict[str, list] — bin edges của mỗi cột.
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()
    binning_info = {}

    for col in binning_cols:
        if col not in X_train.columns:
            continue

        _, bins = pd.qcut(
            X_train[col], q=n_bins, retbins=True, duplicates='drop'
        )
        bins = bins.copy()
        bins[0]  = -np.inf
        bins[-1] = np.inf

        new_col = f'{col}_bin'
        for split in [X_train, X_val, X_test]:
            split[new_col] = pd.cut(
                split[col], bins=bins, include_lowest=True
            ).astype(str)

        binning_info[col] = bins.tolist()

        if drop_original_numeric:
            for split in [X_train, X_val, X_test]:
                split.drop(columns=[col], inplace=True)

    return X_train, X_val, X_test, binning_info


# ══════════════════════════════════════════════════════════════════════════════
# Phần 5 — Pipeline tự động cho DL
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_for_dl(
    df: pd.DataFrame,
    target_col: str = 'income',
    cols_to_drop: list = None,
    cat_impute_strategy: str = 'most_frequent',
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    use_binning: bool = False,
    binning_cols: tuple = ('age', 'hours-per-week'),
    n_bins: int = 4,
    drop_original_numeric_for_bins: bool = True,
    rare_threshold: float = 0.01,
) -> dict:
    """
    Pipeline tiền xử lý hoàn chỉnh cho MLP / Deep Learning.
    Sử dụng One-Hot Encoding thuần cho categorical features.

    Các bước:
      1. Chuẩn hóa '?' → NaN
      2. Loại bỏ cột dư thừa
      3. Tách X / y, encode target
      4. Train / Val / Test split (stratify)
      5. Impute categorical missing (fit trên train)
      6. Gom rare categories (fit trên train)
      7. (Tùy chọn) Quantile binning (fit trên train)
      8. ColumnTransformer: StandardScaler (num) + OHE (cat)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame gốc (chưa qua bước nào).
    target_col : str, default='income'
    cols_to_drop : list hoặc None
        Các cột cần xóa trước khi xử lý (vd: ['education', 'fnlwgt']).
    cat_impute_strategy : str, default='most_frequent'
    test_size : float, default=0.2
    val_size : float, default=0.2
        Tỉ lệ tuyệt đối của val so với toàn bộ dataset.
    random_state : int, default=42
    use_binning : bool, default=False
    binning_cols : tuple
    n_bins : int, default=4
    drop_original_numeric_for_bins : bool, default=True
    rare_threshold : float, default=0.01

    Returns
    -------
    dict với các keys:
        X_train, X_val, X_test   : np.ndarray float32
        y_train, y_val, y_test   : np.ndarray int64
        num_features, cat_features : list[str]
        preprocessor             : fitted ColumnTransformer
        cat_imputer              : SimpleImputer
        input_dim                : int
        y_dist                   : (n_class0, n_class1)
        rare_mapping, binning_info
    """
    data = df.copy()
    data.replace('?', np.nan, inplace=True)

    if cols_to_drop:
        data.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    X = data.drop(columns=[target_col])
    y = data[target_col].map({'<=50K': 0, '>50K': 1}).astype(int)
    y_0, y_1 = (y == 0).sum(), (y == 1).sum()

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")
    print("Encoding mode: OHE thuần")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=random_state, stratify=y_train_full,
    )
    print(f"\nTrain: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

    cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
    X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
    X_val[cat_features]   = cat_imputer.transform(X_val[cat_features])
    X_test[cat_features]  = cat_imputer.transform(X_test[cat_features])

    X_train, X_val, X_test, rare_mapping = group_rare_categories(
        X_train, X_val, X_test, cat_features=cat_features, threshold=rare_threshold
    )

    binning_info = {}
    if use_binning:
        X_train, X_val, X_test, binning_info = apply_quantile_binning(
            X_train, X_val, X_test,
            binning_cols=binning_cols, n_bins=n_bins,
            drop_original_numeric=drop_original_numeric_for_bins,
        )
        print(f"Binning enabled for: {list(binning_info.keys())}")

    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ])

    X_train_np = preprocessor.fit_transform(X_train, y_train).astype(np.float32)
    X_val_np   = preprocessor.transform(X_val).astype(np.float32)
    X_test_np  = preprocessor.transform(X_test).astype(np.float32)
    y_train_np = y_train.values.astype(np.int64)
    y_val_np   = y_val.values.astype(np.int64)
    y_test_np  = y_test.values.astype(np.int64)

    input_dim = X_train_np.shape[1]
    print(f"Input dimension: {input_dim}")
    print(f"Target distribution (train): {np.bincount(y_train_np)}")

    return {
        'X_train': X_train_np, 'X_val': X_val_np, 'X_test': X_test_np,
        'y_train': y_train_np, 'y_val': y_val_np, 'y_test': y_test_np,
        'num_features': num_features, 'cat_features': cat_features,
        'preprocessor': preprocessor, 'cat_imputer': cat_imputer,
        'input_dim': input_dim, 'y_dist': (y_0, y_1),
        'rare_mapping': rare_mapping, 'binning_info': binning_info,
    }


def preprocess_for_tabnet(
    df: pd.DataFrame,
    target_col: str = 'income',
    cols_to_drop: list = None,
    cat_impute_strategy: str = 'most_frequent',
    num_impute_strategy: str = 'median',
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    use_binning: bool = True,
    binning_cols: tuple = ('age', 'hours-per-week'),
    n_bins: int = 4,
    drop_original_numeric_for_bins: bool = True,
    rare_threshold: float = 0.01,
) -> dict:
    """
    Pipeline tiền xử lý dành riêng cho TabNet.

    Khác với preprocess_for_dl:
      - Categorical giữ dạng integer encoding (không OHE)
        để TabNet dùng embedding layer nội bộ.
      - Trả về metadata cat_idxs / cat_dims cho TabNet.
      - Binning được bật mặc định (use_binning=True): các cột số liên tục
        (mặc định là 'age' và 'hours-per-week') được chia thành n_bins bucket
        và chuyển thành biến categorical integer — khác với pipeline MLP vốn
        giữ nguyên các cột số này.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã qua bước làm sạch cơ bản (drop_columns, drop_missing_values,
        map_target_variable). Hàm sẽ tự xử lý imputation và encoding nội bộ.
    target_col : str, default='income'
    cols_to_drop : list hoặc None
    cat_impute_strategy : str, default='most_frequent'
    num_impute_strategy : str, default='median'
    test_size : float, default=0.2
    val_size : float, default=0.2
    random_state : int, default=42
    use_binning : bool, default=True
    binning_cols : tuple
    n_bins : int, default=4
    drop_original_numeric_for_bins : bool, default=True
    rare_threshold : float, default=0.01

    Returns
    -------
    dict với các keys:
        X_train, X_val, X_test   : np.ndarray float32
        y_train, y_val, y_test   : np.ndarray int64
        num_features, cat_features, ordered_features : list[str]
        cat_idxs                 : list[int]
        cat_dims                 : list[int]
        cat_maps                 : dict[str, dict]
        class_weights            : dict[int, float]
        cat_imputer, num_imputer : fitted SimpleImputer
        rare_mapping, binning_info
        input_dim                : int
    """
    data = df.copy()
    data.replace('?', np.nan, inplace=True)

    if cols_to_drop:
        data.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    X = data.drop(columns=[target_col])
    if pd.api.types.is_numeric_dtype(data[target_col]):
        y = data[target_col].astype(int)
    else:
        y = data[target_col].astype(str).str.strip().map({'<=50K': 0, '>50K': 1}).astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=random_state, stratify=y_train_full,
    )

    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

    if num_features:
        num_imputer = SimpleImputer(strategy=num_impute_strategy)
        X_train[num_features] = num_imputer.fit_transform(X_train[num_features])
        X_val[num_features]   = num_imputer.transform(X_val[num_features])
        X_test[num_features]  = num_imputer.transform(X_test[num_features])
    else:
        num_imputer = None

    if cat_features:
        cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
        X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
        X_val[cat_features]   = cat_imputer.transform(X_val[cat_features])
        X_test[cat_features]  = cat_imputer.transform(X_test[cat_features])
    else:
        cat_imputer = None

    X_train, X_val, X_test, rare_mapping = group_rare_categories(
        X_train, X_val, X_test, cat_features=cat_features, threshold=rare_threshold
    )

    binning_info = {}
    if use_binning:
        X_train, X_val, X_test, binning_info = apply_quantile_binning(
            X_train, X_val, X_test,
            binning_cols=binning_cols, n_bins=n_bins,
            drop_original_numeric=drop_original_numeric_for_bins,
        )

    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

    X_train_enc, X_val_enc, X_test_enc, cat_dims, cat_maps = \
        _encode_categorical_for_tabnet(X_train, X_val, X_test, cat_features)

    ordered_features = num_features + cat_features
    cat_idxs = [ordered_features.index(col) for col in cat_features]

    X_train_np = np.column_stack([
        X_train_enc[num_features].to_numpy(dtype=np.float32),
        X_train_enc[cat_features].to_numpy(dtype=np.float32),
    ])
    X_val_np = np.column_stack([
        X_val_enc[num_features].to_numpy(dtype=np.float32),
        X_val_enc[cat_features].to_numpy(dtype=np.float32),
    ])
    X_test_np = np.column_stack([
        X_test_enc[num_features].to_numpy(dtype=np.float32),
        X_test_enc[cat_features].to_numpy(dtype=np.float32),
    ])
    y_train_np = y_train.values.astype(np.int64)
    y_val_np   = y_val.values.astype(np.int64)
    y_test_np  = y_test.values.astype(np.int64)

    class_weights = _compute_balanced_class_weights(y_train_np)

    print("Encoding mode: TabNet integer categorical + internal embeddings")
    print(f"Num features : {len(num_features)} | Cat features: {len(cat_features)}")
    print(f"cat_idxs     : {cat_idxs}")
    print(f"cat_dims     : {cat_dims}")

    return {
        'X_train': X_train_np, 'X_val': X_val_np, 'X_test': X_test_np,
        'y_train': y_train_np, 'y_val': y_val_np, 'y_test': y_test_np,
        'num_features': num_features, 'cat_features': cat_features,
        'ordered_features': ordered_features,
        'cat_idxs': cat_idxs, 'cat_dims': cat_dims, 'cat_maps': cat_maps,
        'class_weights': class_weights,
        'cat_imputer': cat_imputer, 'num_imputer': num_imputer,
        'rare_mapping': rare_mapping, 'binning_info': binning_info,
        'input_dim': X_train_np.shape[1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phần 6 — Helpers nội bộ
# ══════════════════════════════════════════════════════════════════════════════

def _compute_balanced_class_weights(y: np.ndarray) -> dict:
    """
    Tính class weight cân bằng theo công thức: w_j = N / (k * n_j).

    Parameters
    ----------
    y : np.ndarray

    Returns
    -------
    dict[int, float]
    """
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    N, k = len(y), len(classes)
    return {int(cls): float(N / (k * cnt)) for cls, cnt in zip(classes, counts)}


def _encode_categorical_for_tabnet(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list,
):
    """
    Encode categorical sang integer theo vocabulary fit trên train.
    Giá trị không có trong vocabulary của train → unknown index.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
    cat_features : list

    Returns
    -------
    (X_train, X_val, X_test, cat_dims, cat_maps)
        cat_dims : list[int] — số lượng category (+ 1 unknown) mỗi cột.
        cat_maps : dict[str, dict] — mapping string → int mỗi cột.
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()
    cat_dims, cat_maps = [], {}

    for col in cat_features:
        vocab   = sorted(X_train[col].astype(str).unique().tolist())
        mapping = {v: i for i, v in enumerate(vocab)}
        unk_idx = len(mapping)
        cat_maps[col] = mapping
        cat_dims.append(len(mapping) + 1)

        X_train[col] = X_train[col].astype(str).map(mapping).fillna(unk_idx).astype(np.int64)
        X_val[col]   = X_val[col].astype(str).map(mapping).fillna(unk_idx).astype(np.int64)
        X_test[col]  = X_test[col].astype(str).map(mapping).fillna(unk_idx).astype(np.int64)

    return X_train, X_val, X_test, cat_dims, cat_maps