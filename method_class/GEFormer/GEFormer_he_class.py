import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned
from gMLP_class import GEFormer

def run_nested_cv_with_early_stopping(
    data, label, nsnp,
    learning_rate, patience, batch_size,
    epoch=1000
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    num_classes = len(np.unique(label))

    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        x_train = torch.from_numpy(X_train_sub).float().to(device)
        y_train = torch.from_numpy(y_train_sub).long().to(device)
        x_valid = torch.from_numpy(X_valid).float().to(device)
        y_valid = torch.from_numpy(y_valid).long().to(device)
        x_test  = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).long().to(device)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            TensorDataset(x_valid, y_valid), batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test_tensor), batch_size, shuffle=False
        )

        model = GEFormer(nsnp=nsnp, num_classes=num_classes)
        model.train_model(
            train_loader, valid_loader,
            epoch, learning_rate, patience, device
        )

        logits = model.predict(test_loader)
        y_pred = np.argmax(logits, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        if np.isnan(f1) or f1 <= 0:
            print(f"Fold {fold} resulted in NaN or zero F1, pruning trial")
            raise TrialPruned()

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: "
            f"ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, CPU={fold_cpu_mem:.2f}MB"
        )

    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

    return np.mean(all_f1) if all_f1 else 0.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Hyperparameter(data, label, nsnp):
    set_seed(42)

    def objective(trial):
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 0.1, log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        )
        patience = trial.suggest_int("patience", 1, 10)

        try:
            f1_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=learning_rate,
                patience=patience,
                batch_size=batch_size
            )
        except TrialPruned:
            return float("-inf")

        return f1_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params