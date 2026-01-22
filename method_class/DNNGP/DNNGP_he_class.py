import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned
from base_dnngp_class import DNNGP

def run_nested_cv_with_early_stopping(
        data, label, nsnp,
        learning_rate, dropout1, dropout2,
        weight_decay, patience,
        batch_size=64, epoch=1000):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    num_classes = len(np.unique(label))

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        x_train_tensor = torch.from_numpy(X_train_sub).float().to(device)
        y_train_tensor = torch.from_numpy(y_train_sub).long().to(device)
        x_valid_tensor = torch.from_numpy(X_valid).float().to(device)
        y_valid_tensor = torch.from_numpy(y_valid).long().to(device)
        x_test_tensor = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).long().to(device)

        x_train_tensor = x_train_tensor.unsqueeze(1)
        x_valid_tensor = x_valid_tensor.unsqueeze(1)
        x_test_tensor  = x_test_tensor.unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            TensorDataset(x_valid_tensor, y_valid_tensor),
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(x_test_tensor, y_test_tensor),
            batch_size=batch_size, shuffle=False
        )

        model = DNNGP(nsnp, dropout1, dropout2, output_dim=num_classes).to(device)
        model.loss_fn = torch.nn.CrossEntropyLoss()

        model.train_model(
            train_loader, valid_loader,
            epoch, learning_rate, weight_decay, patience, device
        )

        logits = model.predict(test_loader)
        y_pred = torch.argmax(torch.tensor(logits), dim=1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

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

    return np.mean(all_f1)

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

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        patience = trial.suggest_int("patience", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        dropout1 = trial.suggest_float("dropout1", 0.0, 0.9, step=0.1)
        dropout2 = trial.suggest_float("dropout2", 0.0, 0.9, step=0.1)
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])

        try:
            f1 = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=lr,
                dropout1=dropout1,
                dropout2=dropout2,
                weight_decay=weight_decay,
                patience=patience,
                batch_size=batch_size
            )
        except TrialPruned:
            return float("-inf")

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("successfully")
    return study.best_params


if __name__ == "__main__":
    main()
