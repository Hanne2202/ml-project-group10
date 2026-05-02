"""
modules/eda.py
==============
Exploratory Data Analysis (EDA) — phân tích dữ liệu khám phá.

Module này được dùng chung cho cả classical pipeline (01_eda.ipynb)
và deep learning pipeline (04_deep_learning.ipynb).

Cách dùng
---------
    import modules.eda as eda

    df = eda.load_data(url)
    eda.dataset_overview(df)
    num_cols, cat_cols = eda.get_column_types(df)
    eda.check_missing_values(df)
    eda.inspect_categorical_values(df, cat_cols)
    eda.plot_target_distribution(df, target_col='income')
    eda.plot_categorical_by_target(df, cat_cols=['education', 'occupation'], target_col='income')
    eda.plot_numerical_by_target(df, num_cols=['age', 'hours-per-week'], target_col='income')
    eda.plot_correlation_heatmap(df, num_cols)
    eda.analyze_capital_feature(df, 'capital-gain')
    eda.check_education_redundancy(df)
    eda.plot_missing_summary(df)
    eda.plot_numerical_distributions(df, num_cols)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load & tổng quan dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

def load_data(source: str) -> pd.DataFrame:
    """
    Load CSV từ URL hoặc đường dẫn local.

    Parameters
    ----------
    source : str
        URL công khai hoặc đường dẫn file local.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(source)
    print("Dataset shape:", df.shape)
    print(df.head())
    return df


def dataset_overview(df: pd.DataFrame):
    """
    In thông tin tổng quan: kiểu dữ liệu, thống kê mô tả số và phân loại.

    Parameters
    ----------
    df : pd.DataFrame
    """
    print("=== DataFrame Info ===")
    df.info()
    print("\n=== Numerical Summary ===")
    print(df.describe())
    print("\n=== Categorical Summary ===")
    print(df.describe(include='object'))


def get_column_types(df: pd.DataFrame):
    """
    Phân loại cột thành numerical và categorical.

    Returns
    -------
    (num_cols, cat_cols) : tuple[list, list]
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Numerical columns :", num_cols)
    print("Categorical columns:", cat_cols)
    return num_cols, cat_cols


# ──────────────────────────────────────────────────────────────────────────────
# 2. Kiểm tra missing value
# ──────────────────────────────────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame):
    """
    Kiểm tra missing value theo hai dạng:
    - NaN chuẩn
    - Ký hiệu '?' ẩn trong cột categorical

    Parameters
    ----------
    df : pd.DataFrame
    """
    print("=== Missing values (NaN) ===")
    print(df.isnull().sum())
    print("\n=== '?' count per column ===")
    print((df == '?').sum())
    print("\n=== '?' percentage per column (%) ===")
    print(((df == '?').sum() / len(df) * 100).round(2))


def plot_missing_summary(df: pd.DataFrame):
    """
    Vẽ bar chart tổng hợp số lượng '?' trên từng cột.
    Chỉ hiển thị các cột có ít nhất 1 giá trị '?'.

    Parameters
    ----------
    df : pd.DataFrame
    """
    missing_counts = (df == '?').sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print("Không có giá trị '?' nào trong dataset.")
        return

    plt.figure(figsize=(8, 4))
    missing_counts.sort_values(ascending=True).plot(kind='barh')
    plt.title("Số lượng giá trị '?' theo cột")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()


def inspect_categorical_values(df: pd.DataFrame, cat_cols: list, top_n: int = 10):
    """
    In value_counts cho từng cột phân loại.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list
        Danh sách tên cột categorical.
    top_n : int
        Số lượng giá trị phổ biến nhất cần hiển thị.
    """
    for col in cat_cols:
        print(f"\nColumn: {col}  |  unique: {df[col].nunique()}")
        print(df[col].astype(str).str.strip().value_counts(dropna=False).head(top_n))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Phân phối biến mục tiêu
# ──────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'income'):
    """
    In thống kê và vẽ biểu đồ phân phối biến mục tiêu (target).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Tên cột target.
    """
    print("Unique values:", df[target_col].unique())
    print("\nClass counts:")
    print(df[target_col].value_counts())
    print("\nClass percentages (%):")
    print((df[target_col].value_counts(normalize=True) * 100).round(2))

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Phân tích đặc trưng theo target
# ──────────────────────────────────────────────────────────────────────────────

def plot_categorical_by_target(
    df: pd.DataFrame,
    cat_cols: list,
    target_col: str = 'income'
):
    """
    Vẽ stacked bar chart chuẩn hóa (theo hàng) cho từng cột categorical
    để so sánh phân phối giữa các nhóm target.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list
        Danh sách cột categorical cần phân tích.
    target_col : str
        Tên cột target.
    """
    for col in cat_cols:
        print(f"\n=== {col} by {target_col} (proportion) ===")
        crosstab_norm = pd.crosstab(df[col], df[target_col], normalize='index')
        print(crosstab_norm.round(3))

        crosstab_norm.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'{col} by {target_col} (Normalized)')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.show()


def plot_numerical_by_target(
    df: pd.DataFrame,
    num_cols: list,
    target_col: str = 'income'
):
    """
    Vẽ boxplot cho các cột số theo target trong một figure.

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list
        Danh sách cột số cần phân tích.
    target_col : str
        Tên cột target.
    """
    print(df[num_cols].describe())

    n = len(num_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        sns.boxplot(data=df, x=target_col, y=col, ax=ax)
        ax.set_title(f'{col} by {target_col}')
    plt.tight_layout()
    plt.show()


def plot_numerical_distributions(df: pd.DataFrame, num_cols: list):
    """
    Vẽ histogram phân phối cho từng cột số.
    Hữu ích để quan sát skewness và outlier trước khi preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list
        Danh sách cột số.
    """
    n = len(num_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col].dropna(), bins=40, ax=ax, kde=True)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Phân tích đặc trưng đặc biệt
# ──────────────────────────────────────────────────────────────────────────────

def analyze_capital_feature(
    df: pd.DataFrame,
    col: str,
    target_col: str = 'income'
):
    """
    Phân tích phân phối và mối liên hệ với target cho capital-gain hoặc capital-loss.

    Các cột này thường có phân phối rất lệch (phần lớn bằng 0),
    nên cần phân tích riêng phần giá trị khác 0.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Tên cột cần phân tích ('capital-gain' hoặc 'capital-loss').
    target_col : str
        Tên cột target.
    """
    if col not in df.columns:
        print(f"Column '{col}' not found.")
        return

    zero_count = (df[col] == 0).sum()
    nonzero_pct = round((df[col] > 0).sum() / len(df) * 100, 2)

    print(f"=== {col} ===")
    print(df[col].describe())
    print(f"\nZero values : {zero_count} ({round(zero_count / len(df) * 100, 2)}%)")
    print(f"Non-zero    : {(df[col] > 0).sum()} ({nonzero_pct}%)")
    print(f"\nProportion of non-zero {col} by {target_col}:")
    print(pd.crosstab(df[target_col], df[col] > 0, normalize='index').round(3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot toàn bộ
    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f'Boxplot of {col}')
    axes[0].set_ylabel(col)

    # Histogram chỉ cho giá trị khác 0
    nonzero_vals = df.loc[df[col] > 0, col]
    if len(nonzero_vals) > 0:
        sns.histplot(nonzero_vals, bins=50, ax=axes[1])
        axes[1].set_title(f'Non-zero {col} Distribution')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def check_education_redundancy(df: pd.DataFrame):
    """
    Kiểm tra xem 'education' và 'educational-num' có mang cùng thông tin không.

    Nếu mỗi mức education tương ứng đúng một giá trị educational-num,
    hai cột này là redundant và chỉ cần giữ lại một.

    Parameters
    ----------
    df : pd.DataFrame
    """
    if 'education' not in df.columns or 'educational-num' not in df.columns:
        print("Columns 'education' or 'educational-num' not found.")
        return
    mapping = df.groupby('education')['educational-num'].unique().reset_index()
    print("Mapping between education and educational-num:")
    print(mapping)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Ma trận tương quan
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list):
    """
    Vẽ heatmap tương quan (Pearson) giữa các cột số.

    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list
        Danh sách cột số dùng để tính tương quan.
    """
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.show()