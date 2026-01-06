import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from base_dnngp import DNNGP
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data  import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned 

def run_nested_cv_with_early_stopping(data, label, nsnp, learning_rate, 
                                      dropout1, dropout2, weight_decay, patience, batch_size=64, epoch=1000):
    device = torch.device("cuda:0")
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_mse, all_mae, all_r2, all_pcc = [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        x_train_tensor = torch.from_numpy(X_train_sub).float().to(device)
        y_train_tensor = torch.from_numpy(y_train_sub).float().to(device)
        x_valid_tensor = torch.from_numpy(X_valid).float().to(device)
        y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
        x_test_tensor = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).float().to(device)
        
        x_train_tensor = x_train_tensor.unsqueeze(1)
        x_valid_tensor = x_valid_tensor.unsqueeze(1)
        x_test_tensor  = x_test_tensor.unsqueeze(1)

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size, shuffle=False)

        model = DNNGP(nsnp, dropout1, dropout2)
        model.train_model(train_loader, valid_loader, epoch, learning_rate, weight_decay, patience, device)
        y_pred = model.predict(test_loader)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        if np.isnan(pcc):
            print(f"Fold {fold} resulted in NaN PCC, pruning the trial...")
            raise TrialPruned()

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
                f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')
        
    return np.mean(all_pcc) if all_pcc else 0.0

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
        lr = trial.suggest_float("learning_rate", 1e-4, 0.1)
        patience = trial.suggest_int("patience", 10, 100, step=10)
        batch_size = trial.suggest_categorical("batch_size",[64])
        dropout1 = trial.suggest_float("dropout1", 0.0, 0.9, step=0.1)
        dropout2 = trial.suggest_float("dropout2", 0.0, 0.9, step=0.1)
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])

        try:
            corr_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=lr,
                patience=patience,
                dropout1=dropout1,
                dropout2=dropout2,
                weight_decay=weight_decay
            )

        except TrialPruned:
            return float("-inf")
        return corr_score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == "__main__":
    main()
