import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned
from base_deepgs_class import DeepGS

def run_nested_cv_classification( data, label, nsnp, learning_rate, momentum, weight_decay,
    patience, batch_size, num_round=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    num_classes = len(np.unique(label))

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"\n===== Fold {fold} =====")
        fold_start_time = time.time()
        process = psutil.Process(os.getpid())

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        # tensor
        x_train = torch.from_numpy(X_train_sub).float().unsqueeze(1).to(device)
        y_train = torch.from_numpy(y_train_sub).long().to(device)
        x_valid = torch.from_numpy(X_valid).float().unsqueeze(1).to(device)
        y_valid = torch.from_numpy(y_valid).long().to(device)
        x_test  = torch.from_numpy(X_test).float().unsqueeze(1).to(device)
        y_test  = torch.from_numpy(y_test).long().to(device)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            TensorDataset(x_valid, y_valid),
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=batch_size, shuffle=False
        )

        model = DeepGS(nsnp, num_classes=num_classes)
        model.loss_fn = torch.nn.CrossEntropyLoss()

        model.train_model(
            train_loader, valid_loader,
            num_round, learning_rate,
            momentum, weight_decay,
            patience, device
        )

        y_pred = model.predict(test_loader)

        if y_pred.ndim == 2:
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            y_pred_class = y_pred

        acc = accuracy_score(y_test.cpu().numpy(), y_pred_class)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test.cpu().numpy(),
            y_pred_class,
            average="macro",
            zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: "
            f"ACC={acc:.4f}, "
            f"PREC={prec:.4f}, "
            f"REC={rec:.4f}, "
            f"F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, "
            f"CPU={cpu_mem:.2f}MB"
        )

    print("\n===== Final Results =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

    return np.mean(all_f1)

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
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        momentum = trial.suggest_float("momentum", 0.1, 0.9, step=0.1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        weight_decay = trial.suggest_categorical(
            "weight_decay", [1e-4, 1e-3, 1e-2, 1e-1]
        )
        patience = trial.suggest_int("patience", 10, 100, step=10)

        try:
            f1 = run_nested_cv_classification(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                patience=patience,
                batch_size=batch_size
            )
        except Exception as e:
            raise TrialPruned()

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best params:", study.best_params)
    return study.best_params
