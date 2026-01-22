import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from AlexNet_206_class import AlexNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned 

def run_nested_cv_with_early_stopping(data, label, nsnp, num_classes, learning_rate, patience, batch_size, num_round=300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    
    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()
    
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        x_train_tensor = torch.from_numpy(X_train_sub).float().to(device)
        y_train_tensor = torch.from_numpy(y_train_sub).long().to(device)
        x_valid_tensor = torch.from_numpy(X_valid).float().to(device)
        y_valid_tensor = torch.from_numpy(y_valid).long().to(device)
        x_test_tensor = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).long().to(device)
        
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size, shuffle=False)

        model = AlexNet(num_classes=num_classes)
        model.train_model(train_loader, valid_loader, num_round, learning_rate, patience, device)
        y_pred = model.predict(test_loader)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        if np.isnan(f1) or f1 <= 0:
            print(f"Fold {fold} resulted in NaN or zero F1, pruning the trial...")
            raise TrialPruned()
        
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, '
              f'Time={fold_time:.2f}s, CPU={fold_cpu_mem:.2f}MB')
        
    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
        
    return float(np.mean(all_f1)) if all_f1 else 0.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(data, label, nsnp, num_classes):
    set_seed(42)
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience = trial.suggest_int("patience", 10, 100, step=10)
        try:
            f1_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                num_classes=num_classes,
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

if __name__ == "__main__":
    main()
