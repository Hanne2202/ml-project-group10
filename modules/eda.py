import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(source: str) -> pd.DataFrame:
    """Load CSV từ URL hoặc đường dẫn local."""
    df = pd.read_csv(source)
    print("Dataset shape:", df.shape)
    print(df.head())
    return df


def dataset_overview(df: pd.DataFrame):
    """In info, thống kê mô tả số và phân loại của dataset."""
    print("=== DataFrame Info ===")
    df.info()
    print("\n=== Numerical Summary ===")
    print(df.describe())
    print("\n=== Categorical Summary ===")
    print(df.describe(include='object'))


def get_column_types(df: pd.DataFrame):
    """Trả về (num_cols, cat_cols) – danh sách cột số và phân loại."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Numerical columns :", num_cols)
    print("Categorical columns:", cat_cols)
    return num_cols, cat_cols


def check_missing_values(df: pd.DataFrame):
    """Kiểm tra NaN chuẩn và ký hiệu '?' ẩn trong dataset."""
    print("=== Missing values (NaN) ===")
    print(df.isnull().sum())
    print("\n=== '?' count per column ===")
    print((df == '?').sum())


def inspect_categorical_values(df: pd.DataFrame, cat_cols: list, top_n: int = 10):
    """In value_counts cho từng cột phân loại trong cat_cols."""
    for col in cat_cols:
        print(f"\nColumn: {col}  |  unique: {df[col].nunique()}")
        print(df[col].astype(str).str.strip().value_counts(dropna=False).head(top_n))


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'income'):
    """In thống kê và vẽ biểu đồ phân phối biến mục tiêu."""
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


def plot_categorical_by_target(df: pd.DataFrame, cat_cols: list, target_col: str = 'income'):
    """Vẽ stacked bar chart chuẩn hóa cho từng cột phân loại theo target."""
    for col in cat_cols:
        crosstab_norm = pd.crosstab(df[col], df[target_col], normalize='index')
        crosstab_norm.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'{col} by {target_col} (Normalized)')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.show()


def plot_numerical_by_target(df: pd.DataFrame, num_cols: list, target_col: str = 'income'):
    """Vẽ boxplot cho tất cả các cột số theo target trong một figure."""
    n = len(num_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        sns.boxplot(data=df, x=target_col, y=col, ax=ax)
        ax.set_title(f'{col} by {target_col}')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list):
    """Vẽ heatmap tương quan giữa các cột số."""
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.show()


def analyze_capital_feature(df: pd.DataFrame, col: str, target_col: str = 'income'):
    """
    Phân tích phân phối và mối liên hệ với target cho một capital feature
    (capital-gain hoặc capital-loss).
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

    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f'Boxplot of {col}')
    axes[0].set_ylabel(col)

    # Histogram for non-zero values
    nonzero_vals = df.loc[df[col] > 0, col]
    if len(nonzero_vals) > 0:
        sns.histplot(nonzero_vals, bins=50, ax=axes[1])
        axes[1].set_title(f'Non-zero {col} Distribution')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def check_education_redundancy(df: pd.DataFrame):
    """Kiểm tra xem 'education' và 'educational-num' có mang cùng thông tin không."""
    if 'education' not in df.columns or 'educational-num' not in df.columns:
        print("Columns 'education' or 'educational-num' not found.")
        return
    mapping = df.groupby('education')['educational-num'].unique().reset_index()
    print("Mapping between education and educational-num:")
    print(mapping)
