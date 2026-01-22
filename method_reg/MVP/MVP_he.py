import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned
from base_MVP import MVP


def run_nested_cv_with_early_stopping(data, label, n_features, learning_rate, patience, batch_size, nb_filters, num_round=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()
        
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        x_train_tensor = torch.from_numpy(X_train_sub).float().unsqueeze(2).unsqueeze(3)  # (N, n_features, 1, 1)
        y_train_tensor = torch.from_numpy(y_train_sub).float().unsqueeze(1)  # (N, 1)
        
        x_valid_tensor = torch.from_numpy(X_valid).float().unsqueeze(2).unsqueeze(3)
        y_valid_tensor = torch.from_numpy(y_valid).float().unsqueeze(1)  # (N, 1)
        
        x_test_tensor = torch.from_numpy(X_test).float().unsqueeze(2).unsqueeze(3)
        y_test_tensor = torch.from_numpy(y_test).float()
        
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        valid_loader = DataLoader(valid_data, batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_data, batch_size, shuffle=False,
                                num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

        model = MVP(input_size=n_features, nb_filters=nb_filters)
        model.train_model(train_loader, valid_loader, num_round, learning_rate, patience, device)
        y_pred = model.predict(test_loader)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        if np.isnan(pcc) or np.isinf(pcc):
            print(f"Fold {fold} resulted in NaN/Inf PCC, pruning the trial...")
            raise TrialPruned()

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, '
              f'Time={fold_time:.2f}s, CPU={fold_cpu_mem:.2f}MB')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            del model
        
    print("\n===== CV Summary =====")
    print(f"PCC : {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"MAE : {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"MSE : {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"R2  : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
        
    return float(np.mean(all_pcc)) if all_pcc else 0.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(data, label, n_features):
    set_seed(42)
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        patience = trial.suggest_int("patience", 1, 10)
        nb_filters = trial.suggest_categorical("nb_filters", [8, 16, 32])
        try:
            corr_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                n_features=n_features,
                learning_rate=learning_rate,
                patience=patience,
                batch_size=batch_size,
                nb_filters=nb_filters
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
