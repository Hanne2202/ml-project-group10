"""
Deep Learning pipeline cho dữ liệu bảng (tabular).

Bao gồm:
- Preprocessing riêng cho DL (LabelEncoder cho categorical, scaling, encode target)
- Tạo Dataset / DataLoader cho PyTorch
- Định nghĩa mô hình MLP
- Hàm train / evaluate
- Vẽ learning curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)


# ───────────────────────────── 1. Preprocessing ─────────────────────────────

def preprocess_for_dl(
    df: pd.DataFrame,
    target_col: str = "income",
    cols_to_drop: list = None,
    cat_impute_strategy: str = "most_frequent",
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Pipeline tiền xử lý hoàn chỉnh cho Deep Learning.

    Các bước:
    1. Chuẩn hóa missing value ('?' → NaN)
    2. Loại bỏ cột dư thừa
    3. Tách X / y, encode target → 0/1
    4. Xác định cột số / phân loại
    5. Train / val / test split (stratified)
    6. Impute missing cho categorical
    7. Label-encode từng cột categorical
    8. StandardScaler cho numerical

    Returns
    -------
    dict với keys:
        X_train, X_val, X_test   : np.ndarray float32
        y_train, y_val, y_test   : np.ndarray int64
        num_features, cat_features : list[str]
        label_encoders           : dict[str, LabelEncoder]
        scaler                   : StandardScaler
        cat_imputer              : SimpleImputer
        input_dim                : int
    """
    if cols_to_drop is None:
        cols_to_drop = ["education", "fnlwgt"]

    # 1. Chuẩn hóa missing value
    data = df.copy()
    data.replace("?", np.nan, inplace=True)

    # 2. Loại bỏ cột dư thừa
    data.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    # 3. Tách X / y
    X = data.drop(columns=[target_col])
    y = data[target_col].map({"<=50K": 0, ">50K": 1}).astype(int)

    # 4. Xác định kiểu cột
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")

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

    # 7. Label-encode categorical (fit trên train, transform trên val/test)
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Xử lý category chưa gặp ở train → gán thêm class mới
        for split in [X_val, X_test]:
            split[col] = split[col].astype(str).map(
                lambda v, _le=le: _le.transform([v])[0]
                if v in _le.classes_
                else len(_le.classes_)
            )
        label_encoders[col] = le

    # 8. Scale numerical
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_val[num_features] = scaler.transform(X_val[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # Chuyển sang numpy float32
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
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
        "label_encoders": label_encoders, "scaler": scaler,
        "cat_imputer": cat_imputer, "input_dim": input_dim,
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

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = OrderedDict()
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers[f"linear_{i}"] = nn.Linear(prev_dim, h_dim)
            layers[f"bn_{i}"] = nn.BatchNorm1d(h_dim)
            layers[f"relu_{i}"] = nn.ReLU()
            layers[f"drop_{i}"] = nn.Dropout(dropout)
            prev_dim = h_dim

        layers["output"] = nn.Linear(prev_dim, 2)  # 2 classes

        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x)


# ───────────────────────────── 4. Training ──────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Chạy 1 epoch training, trả về (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
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
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = None,
):
    """
    Vòng huấn luyện chính với Early Stopping theo val_loss.

    Returns
    -------
    history : dict  (train_loss, val_loss, train_acc, val_acc theo từng epoch)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Xử lý class imbalance bằng weighted loss
    y_train_all = []
    for _, y_batch in train_loader:
        y_train_all.append(y_batch)
    y_train_all = torch.cat(y_train_all)
    class_counts = torch.bincount(y_train_all).float()
    class_weights = (1.0 / class_counts) * class_counts.sum() / len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        logits = model(X_batch)
        proba = torch.softmax(logits, dim=1)[:, 1]
        all_true.append(y_batch.numpy())
        all_pred.append(logits.argmax(1).cpu().numpy())
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


# ───────────────────────────── 7. Architecture Search ───────────────────────

def compare_architectures(
    prep: dict,
    configs: list[dict] = None,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
    batch_size: int = 256,
):
    """
    Huấn luyện và so sánh nhiều cấu hình MLP khác nhau.

    Parameters
    ----------
    prep      : Output của preprocess_for_dl.
    configs   : Danh sách dict, mỗi dict gồm:
                  - name (str): tên cấu hình
                  - hidden_dims (list[int])
                  - dropout (float)
                  - lr (float)
                  - weight_decay (float)
                Nếu None, dùng bộ configs mặc định.
    epochs    : Số epoch tối đa.
    patience  : Early stopping patience.
    device    : 'cuda' hoặc 'cpu'.
    batch_size: Batch size cho DataLoader.

    Returns
    -------
    results : list[dict]  – mỗi dict chứa config, metrics (val & test), model, history.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if configs is None:
        configs = [
            {"name": "Small [64]",             "hidden_dims": [64],             "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
            {"name": "Medium [128, 64]",       "hidden_dims": [128, 64],        "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
            {"name": "Large [256, 128, 64]",   "hidden_dims": [256, 128, 64],   "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
            {"name": "Deep [128, 64, 32]",     "hidden_dims": [128, 64, 32],    "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
            {"name": "Wide [256, 128]",        "hidden_dims": [256, 128],       "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
            {"name": "Dropout-low [128, 64]",  "hidden_dims": [128, 64],        "dropout": 0.15,"lr": 1e-3, "weight_decay": 1e-4},
            {"name": "Dropout-high [128, 64]", "hidden_dims": [128, 64],        "dropout": 0.5, "lr": 1e-3, "weight_decay": 1e-4},
        ]

    train_loader, val_loader, test_loader = create_dataloaders(prep, batch_size=batch_size)

    results = []
    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n{'='*70}")
        print(f"[{i}/{len(configs)}]  Config: {name}")
        print(f"{'='*70}")

        model = MLP(
            input_dim=prep["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg.get("dropout", 0.3),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        history = train_model(
            model, train_loader, val_loader,
            epochs=epochs,
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            patience=patience,
            device=device,
        )

        # Val metrics
        y_true_val, y_pred_val, _ = predict(model, val_loader, device)
        val_f1 = f1_score(y_true_val, y_pred_val)
        val_acc = accuracy_score(y_true_val, y_pred_val)

        # Test metrics
        y_true_test, y_pred_test, _ = predict(model, test_loader, device)
        test_f1 = f1_score(y_true_test, y_pred_test)
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_prec = precision_score(y_true_test, y_pred_test)
        test_rec = recall_score(y_true_test, y_pred_test)

        results.append({
            "name": name,
            "config": cfg,
            "n_params": n_params,
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "test_acc": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
            "test_precision": round(test_prec, 4),
            "test_recall": round(test_rec, 4),
            "best_val_loss": round(min(history["val_loss"]), 4),
            "model": model,
            "history": history,
        })

        print(f"  → Val  acc={val_acc:.4f}  f1={val_f1:.4f}")
        print(f"  → Test acc={test_acc:.4f}  f1={test_f1:.4f}")

    return results


def show_comparison_table(results: list[dict]):
    """Hiển thị bảng so sánh các cấu hình."""
    rows = []
    for r in results:
        rows.append({
            "Config": r["name"],
            "Params": f"{r['n_params']:,}",
            "Val Acc": r["val_acc"],
            "Val F1": r["val_f1"],
            "Test Acc": r["test_acc"],
            "Test F1": r["test_f1"],
            "Test Prec": r["test_precision"],
            "Test Recall": r["test_recall"],
        })
    df = pd.DataFrame(rows)
    return df


def plot_comparison(results: list[dict]):
    """Vẽ biểu đồ so sánh metrics giữa các cấu hình."""
    names = [r["name"] for r in results]
    val_f1 = [r["val_f1"] for r in results]
    test_f1 = [r["test_f1"] for r in results]
    test_acc = [r["test_acc"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # F1 comparison
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, val_f1, w, label="Val F1", alpha=0.8)
    axes[0].bar(x + w/2, test_f1, w, label="Test F1", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 Score – So sánh các cấu hình")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Learning curves overlay (val_loss)
    for r in results:
        epochs = range(1, len(r["history"]["val_loss"]) + 1)
        axes[1].plot(epochs, r["history"]["val_loss"], label=r["name"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Val Loss – So sánh các cấu hình")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Best config
    best = max(results, key=lambda r: r["val_f1"])
    print(f"\n★ Cấu hình tốt nhất (theo Val F1): {best['name']}  "
          f"(val_f1={best['val_f1']}, test_f1={best['test_f1']})")
    return best
