import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import optuna
from sklearn.metrics import f1_score
import copy

try:
    from modules.deep_learning import (
        MLP, create_dataloaders, train_one_epoch, evaluate, predict, evaluate_model,
        train_tabnet_model, evaluate_tabnet, pretrain_tabnet,
    )
except ImportError:
    from deep_learning import (
        MLP, create_dataloaders, train_one_epoch, evaluate, predict, evaluate_model,
        train_tabnet_model, evaluate_tabnet, pretrain_tabnet,
    )


# ─────────────────────────────── Optuna Objective ───────────────────────────

def _optuna_objective(trial, prep, epochs, patience, device):
    """Optuna objective function: train MLP, return val F1."""

    # --- Search space ---
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = []
    for i in range(n_layers):
        dim = trial.suggest_int(f"hidden_dim_{i}", 32, 512, step=32)
        hidden_dims.append(dim)

    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    # --- Dataloaders ---
    train_loader, val_loader, _ = create_dataloaders(prep, batch_size=batch_size)

    # --- Model ---
    model = MLP(input_dim=prep["input_dim"], hidden_dims=hidden_dims, dropout=dropout)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # --- Loss (class imbalance) ---
    y_train_all = []
    for _, yb in train_loader:
        y_train_all.append(yb)
    y_train_all = torch.cat(y_train_all)
    class_counts = torch.bincount(y_train_all).float()
    pos_weight = (class_counts[0] / class_counts[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer ---
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_state = model.state_dict().copy()
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Pruning: report val F1 mỗi epoch
        model.eval()
        y_true_v, y_pred_v, _ = predict(model, val_loader, device)
        val_f1 = f1_score(y_true_v, y_pred_v)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    y_true_v, y_pred_v, _ = predict(model, val_loader, device)
    return f1_score(y_true_v, y_pred_v)


# ─────────────────────────────── Run Search ─────────────────────────────────

def run_optuna_search(
    prep: dict,
    n_trials: int = 50,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
):
    """
    Chạy Optuna study để tìm hyperparameters tốt nhất cho MLP.

    Returns
    -------
    optuna.Study
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name="MLP-Adult-Income",
    )

    study.optimize(
        lambda trial: _optuna_objective(trial, prep, epochs, patience, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'='*70}")
    print(f"  OPTUNA SEARCH COMPLETED  ({len(study.trials)} trials)")
    print(f"{'='*70}")
    print(f"  Best Val F1: {study.best_value:.4f}")
    print(f"  Best Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study


# ────────────────────────────── Train Best Model ────────────────────────────

def train_best_optuna_model(
    study,
    prep: dict,
    epochs: int = 50,
    patience: int = 10,
    device: str = None,
):
    """
    Train lại mô hình tốt nhất từ Optuna study với đánh giá đầy đủ.

    Đọc best_params từ study, rebuild kiến trúc MLP tốt nhất,
    train từ đầu với đầy đủ epochs rồi đánh giá trên test set.

    Returns
    -------
    (model, history, metrics)
    """
    params = study.best_params

    n_layers = params["n_layers"]
    hidden_dims = [params[f"hidden_dim_{i}"] for i in range(n_layers)]
    dropout = params["dropout"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    optimizer_name = params["optimizer"]

    print(f"Training best model: hidden_dims={hidden_dims}, dropout={dropout}")
    print(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}, batch_size={batch_size}, optimizer={optimizer_name}")

    train_loader, val_loader, test_loader = create_dataloaders(prep, batch_size=batch_size)
    model = MLP(input_dim=prep["input_dim"], hidden_dims=hidden_dims, dropout=dropout)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    y_train_all = []
    for _, yb in train_loader:
        y_train_all.append(yb)
    y_train_all = torch.cat(y_train_all)
    class_counts = torch.bincount(y_train_all).float()
    pos_weight = (class_counts[0] / class_counts[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"Training on {device}  |  epochs={epochs}  |  patience={patience}")
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
            # best_state = model.state_dict().copy()
            best_state = copy.deepcopy(model.state_dict())
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

    metrics, y_true, y_pred, y_proba = evaluate_model(model, test_loader, device=device)

    return model, history, metrics


# ─────────────────────────────── Visualization ──────────────────────────────

def plot_optuna_results(study):
    """Trực quan hóa kết quả Optuna: optimization history + param importances."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("Không có trial hoàn thành để vẽ.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1. Optimization history
    values = [t.value for t in trials]
    best_so_far = np.maximum.accumulate(values)
    axes[0].plot(range(1, len(values) + 1), values, "o-", alpha=0.5, markersize=4, label="Trial F1")
    axes[0].plot(range(1, len(values) + 1), best_so_far, "r-", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Val F1")
    axes[0].set_title("Optuna Optimization History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Param importances
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        imp_vals = list(importances.values())
        y_pos = np.arange(len(params))
        axes[1].barh(y_pos, imp_vals, align="center", alpha=0.8)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(params, fontsize=9)
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Hyperparameter Importances")
        axes[1].grid(True, alpha=0.3, axis="x")
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Cannot compute importances:\n{e}",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Hyperparameter Importances (unavailable)")

    plt.tight_layout()
    plt.show()


# ───────────────────────────── TabNet + Optuna ─────────────────────────────

def _optuna_tabnet_objective(trial, prep_tabnet, max_epochs, patience, device, use_pretraining):
    """Optuna objective function: train TabNet, return val F1."""
    width = trial.suggest_categorical("width", [8, 16, 32, 64])
    n_steps = trial.suggest_int("n_steps", 3, 5)
    gamma = trial.suggest_float("gamma", 1.0, 1.5)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True)
    lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    virtual_batch_size = trial.suggest_categorical("virtual_batch_size", [64, 128, 256])

    pretrainer = None
    if use_pretraining:
        pretrain_epochs = trial.suggest_int("pretrain_epochs", 20, 60, step=10)
        pretrain_ratio = trial.suggest_float("pretraining_ratio", 0.6, 0.9)
        pretrainer = pretrain_tabnet(
            prep_tabnet,
            pretrain_epochs=pretrain_epochs,
            batch_size=batch_size,
            virtual_batch_size=min(virtual_batch_size, batch_size),
            pretraining_ratio=pretrain_ratio,
            n_d=width,
            n_a=width,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            lr=lr,
            device=device,
        )

    tabnet = train_tabnet_model(
        prep_tabnet,
        n_d=width,
        n_a=width,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=min(virtual_batch_size, batch_size),
        custom_weights=prep_tabnet.get("class_weights"),
        pretrainer=pretrainer,
        device=device,
    )

    y_pred_val = tabnet.predict(prep_tabnet["X_val"])
    y_true_val = prep_tabnet["y_val"]
    val_f1 = f1_score(y_true_val, y_pred_val)
    return val_f1


def run_optuna_search_tabnet(
    prep_tabnet: dict,
    n_trials: int = 20,
    max_epochs: int = 80,
    patience: int = 10,
    device: str = None,
    use_pretraining: bool = True,
):
    """
    Chạy Optuna study để tìm hyperparameters tốt nhất cho TabNet.

    Returns
    -------
    optuna.Study
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    study_name = "TabNet-Adult-Income-Pretrain" if use_pretraining else "TabNet-Adult-Income"
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=study_name,
    )

    study.optimize(
        lambda trial: _optuna_tabnet_objective(
            trial,
            prep_tabnet,
            max_epochs,
            patience,
            device,
            use_pretraining,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'='*70}")
    print(f"  TABNET OPTUNA SEARCH COMPLETED  ({len(study.trials)} trials)")
    print(f"{'='*70}")
    print(f"  Best Val F1: {study.best_value:.4f}")
    print("  Best Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study


def train_best_optuna_tabnet_model(
    study,
    prep_tabnet: dict,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = None,
    use_pretraining: bool = True,
):
    """
    Train lại TabNet tốt nhất từ Optuna study với đánh giá đầy đủ trên test set.

    Returns
    -------
    (tabnet_model, metrics)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    params = study.best_params
    width = params["width"]
    n_steps = params["n_steps"]
    gamma = params["gamma"]
    lambda_sparse = params["lambda_sparse"]
    lr = params["lr"]
    batch_size = params["batch_size"]
    virtual_batch_size = min(params["virtual_batch_size"], batch_size)

    pretrainer = None
    if use_pretraining and "pretrain_epochs" in params:
        pretrainer = pretrain_tabnet(
            prep_tabnet,
            pretrain_epochs=params["pretrain_epochs"],
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            pretraining_ratio=params.get("pretraining_ratio", 0.8),
            n_d=width,
            n_a=width,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            lr=lr,
            device=device,
        )

    tabnet_model = train_tabnet_model(
        prep_tabnet,
        n_d=width,
        n_a=width,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        custom_weights=prep_tabnet.get("class_weights"),
        pretrainer=pretrainer,
        device=device,
    )

    metrics, _, _, _ = evaluate_tabnet(tabnet_model, prep_tabnet)
    return tabnet_model, metrics
