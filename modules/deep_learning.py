import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ───────────────────────────── 1. Preprocessing ─────────────────────────────

def group_rare_categories(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list,
    threshold: float = 0.01,
):
    """
    Gom các category xuất hiện ít (< threshold) thành 'Other'.
    Fit trên train, transform giống nhau cho val/test.

    Returns
    -------
    (X_train, X_val, X_test, rare_mapping)
        rare_mapping : dict[str, set]  – tập hợp rare categories cho mỗi cột
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    rare_mapping = {}
    n_train = len(X_train)

    for col in cat_features:
        freq = X_train[col].value_counts(normalize=True)
        rare_cats = set(freq[freq < threshold].index)
        rare_mapping[col] = rare_cats

        if rare_cats:
            X_train[col] = X_train[col].apply(lambda v: "Other" if v in rare_cats else v)
            X_val[col] = X_val[col].apply(lambda v: "Other" if v in rare_cats else v)
            X_test[col] = X_test[col].apply(lambda v: "Other" if v in rare_cats else v)

    n_removed = sum(len(v) for v in rare_mapping.values())
    print(f"Rare category grouping (threshold={threshold}):")
    for col, cats in rare_mapping.items():
        if cats:
            print(f"  {col}: {len(cats)} rare → 'Other'  (kept {X_train[col].nunique()} categories)")
    print(f"  Total rare categories grouped: {n_removed}")

    return X_train, X_val, X_test, rare_mapping


def apply_quantile_binning(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    binning_cols: tuple = ("age", "hours-per-week"),
    n_bins: int = 4,
    drop_original_numeric: bool = True,
):
    """
    Tạo các cột binning theo quantile dựa trên train để tránh leakage.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    binning_info = {}

    for col in binning_cols:
        if col not in X_train.columns:
            continue

        # Lấy bin edges từ train rồi áp dụng lại cho val/test.
        _, bins = pd.qcut(
            X_train[col],
            q=n_bins,
            retbins=True,
            duplicates="drop",
        )

        # Mở rộng biên để val/test ngoài khoảng train vẫn rơi vào bin hợp lệ.
        bins = bins.copy()
        bins[0] = -np.inf
        bins[-1] = np.inf

        new_col = f"{col}_bin"
        X_train[new_col] = pd.cut(X_train[col], bins=bins, include_lowest=True).astype(str)
        X_val[new_col] = pd.cut(X_val[col], bins=bins, include_lowest=True).astype(str)
        X_test[new_col] = pd.cut(X_test[col], bins=bins, include_lowest=True).astype(str)
        binning_info[col] = bins.tolist()

        if drop_original_numeric:
            X_train.drop(columns=[col], inplace=True)
            X_val.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)

    return X_train, X_val, X_test, binning_info


def _compute_balanced_class_weights(y: np.ndarray):
    """
    Tính class weight theo công thức: w_j = N / (k * n_j).
    """
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    N = len(y)
    k = len(classes)
    weights = {int(cls): float(N / (k * cnt)) for cls, cnt in zip(classes, counts)}
    return weights


def _encode_categorical_for_tabnet(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list,
):
    """
    Encode categorical sang integer theo mapping fit trên train.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    cat_dims = []
    cat_maps = {}

    for col in cat_features:
        train_vals = X_train[col].astype(str)
        vocab = sorted(train_vals.unique().tolist())
        mapping = {v: i for i, v in enumerate(vocab)}
        unk_idx = len(mapping)
        cat_maps[col] = mapping
        cat_dims.append(len(mapping) + 1)  # +1 cho unknown

        X_train[col] = train_vals.map(mapping).fillna(unk_idx).astype(np.int64)
        X_val[col] = X_val[col].astype(str).map(mapping).fillna(unk_idx).astype(np.int64)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(unk_idx).astype(np.int64)

    return X_train, X_val, X_test, cat_dims, cat_maps


def preprocess_for_dl(
    df: pd.DataFrame,
    target_col: str = "income",
    cols_to_drop: list = None,
    cat_impute_strategy: str = "most_frequent",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    use_binning: bool = False,
    binning_cols: tuple = ("age", "hours-per-week"),
    n_bins: int = 4,
    drop_original_numeric_for_bins: bool = True,
    rare_threshold: float = 0.01,
):
    """
    Pipeline tiền xử lý hoàn chỉnh cho Deep Learning.
    Sử dụng One-Hot Encoding thuần cho categorical features.

    Returns
    -------
    dict với keys:
        X_train, X_val, X_test   : np.ndarray float32
        y_train, y_val, y_test   : np.ndarray int64
        num_features, cat_features : list[str]
        preprocessor             : fitted ColumnTransformer
        cat_imputer              : SimpleImputer
        input_dim                : int
    """

    # 1. Chuẩn hóa missing value
    data = df.copy()
    data.replace("?", np.nan, inplace=True)

    # 2. Loại bỏ cột dư thừa
    if cols_to_drop:
        data.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    # 3. Tách X / y
    X = data.drop(columns=[target_col])
    y = data[target_col].map({"<=50K": 0, ">50K": 1}).astype(int)

    y_0 = (y == 0).sum()
    y_1 = (y == 1).sum()

    # 4. Xác định kiểu cột
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")
    print(f"Encoding mode: OHE thuần")

    # 5. Split: train → (train + val) / test, rồi train → train / val
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=random_state, stratify=y_train_full,
    )

    print(f"\nTrain: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

    # 6. Impute categorical missing
    cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
    X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
    X_val[cat_features] = cat_imputer.transform(X_val[cat_features])
    X_test[cat_features] = cat_imputer.transform(X_test[cat_features])

    # 6.1 Gom category hiếm trên categorical để giảm sparsity.
    X_train, X_val, X_test, rare_mapping = group_rare_categories(
        X_train, X_val, X_test, cat_features=cat_features, threshold=rare_threshold
    )

    # 6.2 Feature engineering theo quantile binning.
    binning_info = {}
    if use_binning:
        X_train, X_val, X_test, binning_info = apply_quantile_binning(
            X_train,
            X_val,
            X_test,
            binning_cols=binning_cols,
            n_bins=n_bins,
            drop_original_numeric=drop_original_numeric_for_bins,
        )
        print(f"Binning enabled for: {list(binning_info.keys())}")

    # Cập nhật kiểu cột sau các bước feature engineering.
    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # 7. ColumnTransformer: StandardScaler (num) + OHE (cat)
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    X_train_np = preprocessor.fit_transform(X_train, y_train).astype(np.float32)
    X_val_np = preprocessor.transform(X_val).astype(np.float32)
    X_test_np = preprocessor.transform(X_test).astype(np.float32)
    y_train_np = y_train.values.astype(np.int64)
    y_val_np = y_val.values.astype(np.int64)
    y_test_np = y_test.values.astype(np.int64)

    input_dim = X_train_np.shape[1]
    print(f"Input dimension: {input_dim}")
    print(f"Target distribution (train): {np.bincount(y_train_np)}")

    return {
        "X_train": X_train_np, "X_val": X_val_np, "X_test": X_test_np,
        "y_train": y_train_np, "y_val": y_val_np, "y_test": y_test_np,
        "num_features": num_features, "cat_features": cat_features,
        "preprocessor": preprocessor, "cat_imputer": cat_imputer,
        "input_dim": input_dim, "y_dist": (y_0, y_1),
        "rare_mapping": rare_mapping, "binning_info": binning_info,
    }


def preprocess_for_tabnet(
    df: pd.DataFrame,
    target_col: str = "income",
    cols_to_drop: list = None,
    cat_impute_strategy: str = "most_frequent",
    num_impute_strategy: str = "median",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    use_binning: bool = True,
    binning_cols: tuple = ("age", "hours-per-week"),
    n_bins: int = 4,
    drop_original_numeric_for_bins: bool = True,
    rare_threshold: float = 0.01,
):
    """
    Preprocessing dành riêng cho TabNet: categorical giữ dạng integer encoding,
    đồng thời trả về metadata cat_idxs/cat_dims để TabNet dùng embedding nội bộ.
    """
    data = df.copy()
    data.replace("?", np.nan, inplace=True)

    if cols_to_drop:
        data.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    X = data.drop(columns=[target_col])
    y = data[target_col].map({"<=50K": 0, ">50K": 1}).astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    if num_features:
        num_imputer = SimpleImputer(strategy=num_impute_strategy)
        X_train[num_features] = num_imputer.fit_transform(X_train[num_features])
        X_val[num_features] = num_imputer.transform(X_val[num_features])
        X_test[num_features] = num_imputer.transform(X_test[num_features])
    else:
        num_imputer = None

    if cat_features:
        cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
        X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
        X_val[cat_features] = cat_imputer.transform(X_val[cat_features])
        X_test[cat_features] = cat_imputer.transform(X_test[cat_features])
    else:
        cat_imputer = None

    X_train, X_val, X_test, rare_mapping = group_rare_categories(
        X_train, X_val, X_test, cat_features=cat_features, threshold=rare_threshold
    )

    binning_info = {}
    if use_binning:
        X_train, X_val, X_test, binning_info = apply_quantile_binning(
            X_train,
            X_val,
            X_test,
            binning_cols=binning_cols,
            n_bins=n_bins,
            drop_original_numeric=drop_original_numeric_for_bins,
        )

    # Cập nhật feature lists sau feature engineering.
    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    X_train_enc, X_val_enc, X_test_enc, cat_dims, cat_maps = _encode_categorical_for_tabnet(
        X_train, X_val, X_test, cat_features=cat_features
    )

    ordered_features = num_features + cat_features
    cat_idxs = [ordered_features.index(col) for col in cat_features]

    X_train_np = np.column_stack(
        [X_train_enc[num_features].to_numpy(dtype=np.float32), X_train_enc[cat_features].to_numpy(dtype=np.float32)]
    )
    X_val_np = np.column_stack(
        [X_val_enc[num_features].to_numpy(dtype=np.float32), X_val_enc[cat_features].to_numpy(dtype=np.float32)]
    )
    X_test_np = np.column_stack(
        [X_test_enc[num_features].to_numpy(dtype=np.float32), X_test_enc[cat_features].to_numpy(dtype=np.float32)]
    )

    y_train_np = y_train.values.astype(np.int64)
    y_val_np = y_val.values.astype(np.int64)
    y_test_np = y_test.values.astype(np.int64)

    class_weights = _compute_balanced_class_weights(y_train_np)

    print("Encoding mode: TabNet integer categorical + internal embeddings")
    print(f"Num features: {len(num_features)} | Cat features: {len(cat_features)}")
    print(f"cat_idxs: {cat_idxs}")
    print(f"cat_dims: {cat_dims}")

    return {
        "X_train": X_train_np,
        "X_val": X_val_np,
        "X_test": X_test_np,
        "y_train": y_train_np,
        "y_val": y_val_np,
        "y_test": y_test_np,
        "num_features": num_features,
        "cat_features": cat_features,
        "ordered_features": ordered_features,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "cat_maps": cat_maps,
        "class_weights": class_weights,
        "cat_imputer": cat_imputer,
        "num_imputer": num_imputer,
        "rare_mapping": rare_mapping,
        "binning_info": binning_info,
        "input_dim": X_train_np.shape[1],
    }


# ───────────────────────────── 2. Dataset / DataLoader ──────────────────────

class TabularDataset(Dataset):
    """PyTorch Dataset cho dữ liệu bảng đã được encode sẵn."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(prep: dict, batch_size: int = 256, num_workers: int = 0):
    """
    Tạo DataLoader cho train / val / test từ output của preprocess_for_dl.
    """
    train_ds = TabularDataset(prep["X_train"], prep["y_train"])
    val_ds = TabularDataset(prep["X_val"], prep["y_val"])
    test_ds = TabularDataset(prep["X_test"], prep["y_test"])

    # shuffle=True để mỗi epoch có batch khác nhau, giúp mô hình tổng quát hơn
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders created  |  batch_size={batch_size}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}  |  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ───────────────────────────── 3. Model ─────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron cho binary classification trên dữ liệu bảng.

    Parameters
    ----------
    input_dim    : Số đặc trưng đầu vào.
    hidden_dims  : Danh sách kích thước từng hidden layer, mặc định [128, 64].
    dropout      : Tỷ lệ dropout sau mỗi hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Dùng OrderedDict để các lớp được thêm vào đúng thứ tự và có tên rõ ràng
        layers = OrderedDict()
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers[f"linear_{i}"] = nn.Linear(prev_dim, h_dim)
            layers[f"bn_{i}"] = nn.BatchNorm1d(h_dim)
            layers[f"relu_{i}"] = nn.ReLU()
            layers[f"drop_{i}"] = nn.Dropout(dropout)

            # Cập nhật đầu vào cho lớp tiếp theo
            prev_dim = h_dim

        layers["output"] = nn.Linear(prev_dim, 1)  # 1 output: P(income > 50K) (raw logit)

        self.net = nn.Sequential(layers)

    def forward(self, x):
        # self.net trả về shape (batch, 1) → squeeze để thành (batch,)
        return self.net(x).squeeze(1)  # shape (batch,)


# ───────────────────────────── 4. Training ──────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Chạy 1 epoch training, trả về (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Xoá gradients cũ trước khi backpropagation
        optimizer.zero_grad()

        # Lan truyền tiến (forward pass) để tính logits (chưa qua sigmoid)
        logits = model(X_batch)                        # shape (batch,)

        # Tính loss với BCEWithLogitsLoss (tự tích hợp sigmoid + binary cross-entropy)
        loss = criterion(logits, y_batch.float())

        # Lan truyền ngược (backward pass) để tính gradients
        loss.backward()

        # Cập nhật trọng số với optimizer
        optimizer.step()

        # Cộng dồn loss và tính accuracy trên batch
        total_loss += loss.item() * len(y_batch)

        # Dự đoán nhãn: sigmoid(logits) > 0.5 → logits > 0
        preds = (logits > 0).long()                    # sigmoid(0)=0.5 → ngưỡng 0.5
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate trên một DataLoader, trả về (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch.float())

        total_loss += loss.item() * len(y_batch)
        preds = (logits > 0).long()
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,             # Learning rate cho optimizer (Adam)
    weight_decay: float = 1e-4,   # Kĩ thuật regularization L2 giúp tránh overfitting bằng cách phạt các trọng số lớn
    patience: int = 10,           # Số epoch liên tiếp không cải thiện val_loss để dừng sớm
    device: str = None,
):
    """
    Vòng huấn luyện chính với Early Stopping theo val_loss.

    Returns
    -------
    history : dict  (train_loss, val_loss, train_acc, val_acc theo từng epoch)
    """

    # Đây model lên device (GPU nếu có) trước khi tạo criterion/optimizer để đảm bảo pos_weight cũng được chuyển đúng
    model = model.to(device)

    # Xử lý class imbalance bằng pos_weight cho BCEWithLogitsLoss


    y_train_all = []
    # Duyệt toàn bộ train_loader để lấy tất cả y_train
    for _, y_batch in train_loader:
        y_train_all.append(y_batch)
    # Kết hợp tất cả batch lại thành một tensor duy nhất
    y_train_all = torch.cat(y_train_all)

    class_weights = _compute_balanced_class_weights(y_train_all.cpu().numpy())
    w0 = class_weights.get(0, 1.0)
    w1 = class_weights.get(1, 1.0)
    pos_weight = torch.tensor(w1 / w0, dtype=torch.float32)
    print(
        f"Balanced class weights (train): {class_weights} | "
        f"BCE pos_weight (w1/w0): {pos_weight.item():.4f}"
    )

    # Định nghĩa hàm Loss: BCEWithLogitsLoss tích hợp sẵn Sigmoid vào bên trong, và sử dụng pos_weight để xử lý imbalance
    # Biến logits đầu ra của model sẽ được sigmoid chuyển thành xác suất, sau đó tính binary cross-entropy loss với y_batch
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Định nghĩa optimizer: Adam với learning rate và weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: giảm lr khi val_loss không cải thiện sau 'patience' epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"Training on {device}  |  epochs={epochs}  |  lr={lr}  |  patience={patience}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(
                f"Epoch {epoch:3d}/{epochs}  |  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{marker}"
            )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    # Khôi phục trọng số tốt nhất
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return history


# ───────────────────────────── 5. Evaluation ────────────────────────────────

@torch.no_grad()
def predict(model, loader, device=None):
    """Trả về (y_true, y_pred, y_proba) từ DataLoader."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_true, all_pred, all_proba = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)                         # shape (batch,)
        proba = torch.sigmoid(logits)                   # P(income > 50K)
        preds = (proba > 0.5).long()
        all_true.append(y_batch.numpy())
        all_pred.append(preds.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba)


def evaluate_model(model, test_loader, device=None):
    """In classification report và confusion matrix trên test set."""
    y_true, y_pred, y_proba = predict(model, test_loader, device)

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["<=50K", ">50K"]))

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
    }
    print("Summary:", metrics)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – MLP (Deep Learning)")
    plt.tight_layout()
    plt.show()

    return metrics, y_true, y_pred, y_proba


# ───────────────────────────── 6. Visualization ─────────────────────────────

def plot_learning_curves(history: dict):
    """Vẽ loss và accuracy theo epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ───────────────────────────── 8. TabNet ────────────────────────────────────

def train_tabnet_model(
    prep: dict,
    n_d: int = 32,
    n_a: int = 32,
    n_steps: int = 3,
    gamma: float = 1.3,
    lambda_sparse: float = 1e-3,
    lr: float = 2e-2,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    virtual_batch_size: int = 128,
    custom_weights: dict = None,
    pretrainer=None,
    device: str = None,
):
    """
    Train TabNetClassifier trên dữ liệu đã tiền xử lý.
    Yêu cầu: pip install pytorch-tabnet

    Parameters
    ----------
    n_d, n_a  : chiều rộng các lớp decision/attention (nên bằng nhau)
    n_steps   : số bước attention (thường 3-10)

    Returns
    -------
    TabNetClassifier đã fit
    """
    from pytorch_tabnet.tab_model import TabNetClassifier

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tabnet = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        cat_idxs=prep.get("cat_idxs", []),
        cat_dims=prep.get("cat_dims", []),
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr},
        device_name=device,
        verbose=1,
        seed=42,
    )

    if custom_weights is None:
        custom_weights = prep.get("class_weights", _compute_balanced_class_weights(prep["y_train"]))

    print(f"TabNet class weights: {custom_weights}")

    tabnet.fit(
        X_train=prep["X_train"],
        y_train=prep["y_train"],
        eval_set=[(prep["X_val"], prep["y_val"])],
        eval_name=["val"],
        eval_metric=["logloss"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        weights=custom_weights,
        from_unsupervised=pretrainer,
    )

    print(f"TabNet training complete. Best epoch: {tabnet.best_epoch}")
    return tabnet


def pretrain_tabnet(
    prep: dict,
    pretrain_epochs: int = 80,
    batch_size: int = 256,
    virtual_batch_size: int = 128,
    pretraining_ratio: float = 0.8,
    n_d: int = 32,
    n_a: int = 32,
    n_steps: int = 3,
    gamma: float = 1.3,
    lambda_sparse: float = 1e-3,
    lr: float = 2e-2,
    device: str = None,
):
    """
    Self-supervised pretraining cho TabNet bằng masked reconstruction.
    """
    from pytorch_tabnet.pretraining import TabNetPretrainer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrainer = TabNetPretrainer(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        cat_idxs=prep.get("cat_idxs", []),
        cat_dims=prep.get("cat_dims", []),
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr},
        seed=42,
        device_name=device,
        verbose=1,
    )

    X_unlabeled = np.vstack([prep["X_train"], prep["X_val"], prep["X_test"]])

    pretrainer.fit(
        X_train=X_unlabeled,
        eval_set=[prep["X_val"]],
        eval_name=["val"],
        max_epochs=pretrain_epochs,
        patience=10,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        pretraining_ratio=pretraining_ratio,
    )

    print(f"TabNet pretraining complete. Best epoch: {pretrainer.best_epoch}")
    return pretrainer


def evaluate_tabnet(tabnet, prep: dict):
    """
    Đánh giá TabNetClassifier trên test set.

    Returns
    -------
    (metrics, y_true, y_pred, y_proba)
    """
    y_pred = tabnet.predict(prep["X_test"])
    y_proba = tabnet.predict_proba(prep["X_test"])[:, 1]
    y_true = prep["y_test"]

    print("=== Classification Report (TabNet) ===")
    print(classification_report(y_true, y_pred, target_names=["<=50K", ">50K"]))

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
    }
    print("Summary:", metrics)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – TabNet")
    plt.tight_layout()
    plt.show()

    return metrics, y_true, y_pred, y_proba
