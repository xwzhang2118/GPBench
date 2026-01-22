import os
import time
import psutil
import random
import torch
import numpy as np
import optuna

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned

from base_DeepCCR_class import DeepCCR

def run_nested_cv_with_early_stopping(
    data,
    label,
    nsnp,
    num_classes,
    learning_rate,
    batch_size,
    patience,
    epoch=1000
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    acc_list, pre_list, rec_list, f1_list = [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        x_tr = torch.from_numpy(X_tr).float().unsqueeze(1)
        y_tr = torch.from_numpy(y_tr).long()
        x_val = torch.from_numpy(X_val).float().unsqueeze(1)
        y_val = torch.from_numpy(y_val).long()
        x_te = torch.from_numpy(X_test).float().unsqueeze(1)
        y_te = torch.from_numpy(y_test).long()

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr), batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            TensorDataset(x_val, y_val), batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(x_te, y_te), batch_size, shuffle=False
        )

        model = DeepCCR(
            input_seq_len=nsnp,
            num_classes=num_classes
        )

        model.train_model(
            train_loader,
            valid_loader,
            epoch,
            learning_rate,
            patience,
            device
        )

        y_pred = model.predict(test_loader, device)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        if np.isnan(f1):
            print(f"Fold {fold} produced NaN F1, pruning trial.")
            raise TrialPruned()

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0
        )
        fold_cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: "
            f"ACC={acc:.4f}, PRE={pre:.4f}, REC={rec:.4f}, F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, "
            f"GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB"
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return {
        "acc": np.mean(acc_list),
        "pre": np.mean(pre_list),
        "rec": np.mean(rec_list),
        "f1": np.mean(f1_list),
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(data, label, nsnp):
    set_seed(42)

    label = label.astype(int)
    num_classes = len(np.unique(label))
    print("Number of classes:", num_classes)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience = trial.suggest_int("patience", 3, 15)

        try:
            metrics = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                num_classes=num_classes,
                learning_rate=lr,
                batch_size=batch_size,
                patience=patience
            )
        except TrialPruned:
            return float("-inf")
        return metrics["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == "__main__":
    main()
