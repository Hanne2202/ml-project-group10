import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Thay thế '?' bằng NaN để chuẩn hóa missing value."""
    df_clean = df.copy()
    df_clean.replace('?', np.nan, inplace=True)
    print("Missing values after standardization:")
    print(df_clean.isnull().sum())
    return df_clean


def drop_redundant_features(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Loại bỏ các cột không cần thiết hoặc dư thừa thông tin."""
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    print("Shape after dropping:", df_clean.shape)
    print("Remaining columns:", df_clean.columns.tolist())
    return df_clean


def split_features_target(df: pd.DataFrame, target_col: str = 'income'):
    """Tách tập đặc trưng X và biến mục tiêu y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("\nTarget distribution:")
    print(y.value_counts())
    return X, y


def identify_feature_types(X: pd.DataFrame):
    """Xác định và trả về (num_features, cat_features)."""
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    print("Numerical features :", num_features)
    print("Categorical features:", cat_features)
    return num_features, cat_features


def get_preprocessing_configs(num_features: list, cat_features: list) -> dict:
    """
    Trả về dict các ColumnTransformer ứng với 4 cấu hình preprocessing:

    - config_1: StandardScaler + most_frequent imputer + OHE
    - config_2: StandardScaler + constant-'Missing' imputer + OHE
    - config_3: MinMaxScaler  + most_frequent imputer + OHE
    - config_4: no scaling   + most_frequent imputer + OHE
    """
    configs = {
        "config_1_onehot_mostfreq_standard": ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]),
        "config_2_onehot_constant_standard": ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]),
        "config_3_onehot_mostfreq_minmax": ColumnTransformer(transformers=[
            ('num', MinMaxScaler(), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]),
        "config_4_onehot_mostfreq_noscale": ColumnTransformer(transformers=[
            ('num', 'passthrough', num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ])
    }
    print("Available preprocessing configurations:")
    for name in configs:
        print(" -", name)
    return configs


def compare_preprocessing_configs(
    configs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pos_label: str = '>50K'
) -> pd.DataFrame:
    """
    So sánh các cấu hình preprocessing bằng Logistic Regression.

    Trả về DataFrame với accuracy, precision, recall, f1_score, n_features
    sắp xếp theo f1_score giảm dần.
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    rows = []
    for name, transformer in configs.items():
        X_tr = transformer.fit_transform(X_train)
        X_te = transformer.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=5000, solver='lbfgs')
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        rows.append({
            'configuration': name,
            'n_features': X_tr.shape[1],
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, pos_label=pos_label), 4),
            'recall': round(recall_score(y_test, y_pred, pos_label=pos_label), 4),
            'f1_score': round(f1_score(y_test, y_pred, pos_label=pos_label), 4),
        })

    results_df = (
        pd.DataFrame(rows)
        .sort_values(by='f1_score', ascending=False)
        .reset_index(drop=True)
    )
    print(results_df)
    return results_df


def select_best_config(configs: dict, results_df: pd.DataFrame, X_train, X_test):
    """
    Chọn cấu hình tốt nhất theo f1_score, fit lại preprocessor trên toàn bộ train set.

    Trả về (best_name, best_preprocessor, X_train_best, X_test_best).
    """
    best_name = results_df.iloc[0]['configuration']
    best_preprocessor = configs[best_name]

    X_train_best = best_preprocessor.fit_transform(X_train)
    X_test_best = best_preprocessor.transform(X_test)

    print(f"Best config selected : {best_name}")
    print("X_train_best shape   :", X_train_best.shape)
    print("X_test_best shape    :", X_test_best.shape)
    return best_name, best_preprocessor, X_train_best, X_test_best
