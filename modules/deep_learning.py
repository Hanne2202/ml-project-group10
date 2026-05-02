"""
modules/deep_learning.py
========================
Deep Learning Pipeline — Dataset, MLP, training loop, TabNet.

Phần preprocessing đã được chuyển sang modules/preprocessing.py.
Module này chỉ chứa:
  - Dataset / DataLoader
  - MLP model
  - Training / evaluation loop
  - TabNet wrapper

Cách dùng
---------
    import modules.deep_learning as dl
    import modules.preprocessing as prep

    # Preprocessing (step-by-step cho MLP)
    prep_data = { 'X_train': ..., 'X_val': ..., ... }
    train_loader, val_loader, test_loader = dl.create_dataloaders(prep_data)

    # Preprocessing tự động cho TabNet
    prep_tabnet = prep.preprocess_for_tabnet(df_clean, target_col='income')
    tabnet = dl.train_tabnet_model(prep_tabnet)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# Import helper nội bộ từ preprocessing để tính class weight
from modules.preprocessing import _compute_balanced_class_weights


# ─────────────────────────── 1. Dataset / DataLoader ────────────────────────

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

    Parameters
    ----------
    prep : dict
        Chứa X_train, X_val, X_test, y_train, y_val, y_test (np.ndarray).
    batch_size : int, default=256
    num_workers : int, default=0

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_ds = TabularDataset(prep["X_train"], prep["y_train"])
    val_ds   = TabularDataset(prep["X_val"],   prep["y_val"])
    test_ds  = TabularDataset(prep["X_test"],  prep["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders created  |  batch_size={batch_size}")
    print(f"  Train batches: {len(train_loader)}  |  Val: {len(val_loader)}  |  Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────── 2. MLP Model ───────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron cho binary classification trên dữ liệu bảng.

    Kiến trúc: Input → [Linear → BatchNorm → ReLU → Dropout] × n → Output (1 logit)

    Parameters
    ----------
    input_dim   : Số đặc trưng đầu vào.
    hidden_dims : Danh sách kích thước từng hidden layer, mặc định [128, 64].
    dropout     : Tỷ lệ dropout sau mỗi hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = OrderedDict()
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers[f"linear_{i}"] = nn.Linear(prev_dim, h_dim)
            layers[f"bn_{i}"]     = nn.BatchNorm1d(h_dim)
            layers[f"relu_{i}"]   = nn.ReLU()
            layers[f"drop_{i}"]   = nn.Dropout(dropout)
            prev_dim = h_dim

        layers["output"] = nn.Linear(prev_dim, 1)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape (batch,)


# ─────────────────────────────── 3. Training ────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Chạy 1 epoch training, trả về (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds   = (logits > 0).long()
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate trên một DataLoader, trả về (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch.float())

        total_loss += loss.item() * len(y_batch)
        preds   = (logits > 0).long()
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

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
    use_pos_weight: bool = True,
):
    """
    Vòng huấn luyện chính với Early Stopping theo val_loss.

    Parameters
    ----------
    use_pos_weight : bool, default=True
        Có tính pos_weight để bù class imbalance không.
        Đặt False khi train data đã được cân bằng bằng SMOTE.

    Returns
    -------
    history : dict  (train_loss, val_loss, train_acc, val_acc theo epoch)
    """
    import copy
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if use_pos_weight:
        y_train_all = torch.cat([yb for _, yb in train_loader])
        class_counts = torch.bincount(y_train_all).float()
        pos_weight   = (class_counts[0] / class_counts[1]).to(device)
        print(f"BCE pos_weight (w1/w0): {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("use_pos_weight=False: BCEWithLogitsLoss không trọng số.")
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss    = float("inf")
    best_state       = None
    epochs_no_improve = 0

    print(f"Training on {device}  |  epochs={epochs}  |  lr={lr}  |  patience={patience}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = copy.deepcopy(model.state_dict())
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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


# ─────────────────────────────── 4. Evaluation ──────────────────────────────

@torch.no_grad()
def predict(model, loader, device=None):
    """Trả về (y_true, y_pred, y_proba) từ DataLoader."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_true, all_pred, all_proba = [], [], []
    for X_batch, y_batch in loader:
        X_batch  = X_batch.to(device)
        logits   = model(X_batch)
        proba    = torch.sigmoid(logits)
        preds    = (proba > 0.5).long()
        all_true.append(y_batch.numpy())
        all_pred.append(preds.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba)


def evaluate_model(model, test_loader, device=None):
    """
    In classification report, confusion matrix và trả về metrics trên test set.

    Returns
    -------
    (metrics, y_true, y_pred, y_proba)
    """
    y_true, y_pred, y_proba = predict(model, test_loader, device)

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["<=50K", ">50K"]))

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1_score":  round(f1_score(y_true, y_pred), 4),
    }
    print("Summary:", metrics)

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – MLP")
    plt.tight_layout()
    plt.show()

    return metrics, y_true, y_pred, y_proba


# ─────────────────────────────── 5. Visualization ───────────────────────────

def plot_learning_curves(history: dict):
    """Vẽ loss và accuracy theo epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────── 6. TabNet ──────────────────────────────────

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
    Train TabNetClassifier trên dữ liệu đã tiền xử lý bởi preprocess_for_tabnet.

    Parameters
    ----------
    prep : dict — output của prep.preprocess_for_tabnet()
    n_d, n_a    : chiều rộng các lớp decision / attention (nên bằng nhau)
    n_steps     : số bước attention
    pretrainer  : TabNetPretrainer đã fit (từ pretrain_tabnet), hoặc None

    Returns
    -------
    TabNetClassifier đã fit
    """
    from pytorch_tabnet.tab_model import TabNetClassifier

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tabnet = TabNetClassifier(
        n_d=n_d, n_a=n_a, n_steps=n_steps,
        gamma=gamma, lambda_sparse=lambda_sparse,
        cat_idxs=prep.get("cat_idxs", []),
        cat_dims=prep.get("cat_dims", []),
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr},
        device_name=device, verbose=1, seed=42,
    )

    if custom_weights is None:
        custom_weights = prep.get(
            "class_weights",
            _compute_balanced_class_weights(prep["y_train"])
        )

    print(f"TabNet class weights: {custom_weights}")

    tabnet.fit(
        X_train=prep["X_train"], y_train=prep["y_train"],
        eval_set=[(prep["X_val"], prep["y_val"])],
        eval_name=["val"], eval_metric=["logloss"],
        max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, virtual_batch_size=virtual_batch_size,
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
    n_d: int = 32, n_a: int = 32,
    n_steps: int = 3, gamma: float = 1.3,
    lambda_sparse: float = 1e-3,
    lr: float = 2e-2,
    device: str = None,
):
    """
    Self-supervised pretraining cho TabNet bằng masked reconstruction.

    Parameters
    ----------
    prep : dict — output của prep.preprocess_for_tabnet()

    Returns
    -------
    TabNetPretrainer đã fit
    """
    from pytorch_tabnet.pretraining import TabNetPretrainer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrainer = TabNetPretrainer(
        n_d=n_d, n_a=n_a, n_steps=n_steps,
        gamma=gamma, lambda_sparse=lambda_sparse,
        cat_idxs=prep.get("cat_idxs", []),
        cat_dims=prep.get("cat_dims", []),
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr},
        seed=42, device_name=device, verbose=1,
    )

    X_unlabeled = np.vstack([prep["X_train"], prep["X_val"], prep["X_test"]])
    pretrainer.fit(
        X_train=X_unlabeled,
        eval_set=[prep["X_val"]], eval_name=["val"],
        max_epochs=pretrain_epochs, patience=10,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        pretraining_ratio=pretraining_ratio,
    )

    print(f"TabNet pretraining complete. Best epoch: {pretrainer.best_epoch}")
    return pretrainer


def evaluate_tabnet(tabnet, prep: dict):
    """
    Đánh giá TabNetClassifier trên test set.

    Parameters
    ----------
    tabnet : TabNetClassifier đã fit
    prep   : dict — output của prep.preprocess_for_tabnet()

    Returns
    -------
    (metrics, y_true, y_pred, y_proba)
    """
    y_pred  = tabnet.predict(prep["X_test"])
    y_proba = tabnet.predict_proba(prep["X_test"])[:, 1]
    y_true  = prep["y_test"]

    print("=== Classification Report (TabNet) ===")
    print(classification_report(y_true, y_pred, target_names=["<=50K", ">50K"]))

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1_score":  round(f1_score(y_true, y_pred), 4),
    }
    print("Summary:", metrics)

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – TabNet")
    plt.tight_layout()
    plt.show()

    return metrics, y_true, y_pred, y_proba