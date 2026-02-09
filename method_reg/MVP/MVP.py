import os
import time
import psutil
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from base_MVP import MVP
import MVP_Hyperparameters
import pynvml


def parse_args():
    parser = argparse.ArgumentParser(description="MVP regression")
    parser.add_argument('--methods', type=str, default='MVP/')
    parser.add_argument('--species', type=str, default='Wheat/')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--nb_filters', type=int, default=32)
    return parser.parse_args()


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    return xData, yData, nsample, nsnp, names


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_mem_by_pid(pid, handle=None):
    if handle is None:
        return 0.0
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in procs:
            if p.pid == pid:
                return p.usedGpuMemory / 1024**2
        return 0.0
    except Exception:
        return 0.0


def run_nested_cv(args, data, label, n_features, device, gpu_handle=None):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
    
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        x_train_tensor = torch.from_numpy(X_train_sub).float().unsqueeze(2).unsqueeze(3)
        y_train_tensor = torch.from_numpy(y_train_sub).float().unsqueeze(1)
        
        x_valid_tensor = torch.from_numpy(X_valid).float().unsqueeze(2).unsqueeze(3)
        y_valid_tensor = torch.from_numpy(y_valid).float().unsqueeze(1)
        
        x_test_tensor = torch.from_numpy(X_test).float().unsqueeze(2).unsqueeze(3)
        y_test_tensor = torch.from_numpy(y_test).float()

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        valid_loader = DataLoader(valid_data, args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

        model = MVP(input_size=n_features, nb_filters=args.nb_filters)
        model.train_model(train_loader, valid_loader, args.epochs, args.learning_rate, args.patience, device)
        y_pred = model.predict(test_loader)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = get_gpu_mem_by_pid(os.getpid(), gpu_handle)
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, '
              f'Time={fold_time:.2f}s, GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    gpu_handle = None
    try:
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Warning: GPU monitoring initialization failed: {e}")
        gpu_handle = None
    
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_species =['Cotton/']

    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        for j in range(len(names)):
            args.phe = names[j]
            print("starting run " + args.methods + args.species + args.phe)
            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            best_params = MVP_Hyperparameters.Hyperparameter(X, label, nsnp)
            args.learning_rate = best_params['learning_rate']
            args.batch_size = best_params['batch_size']
            args.patience = best_params['patience']
            args.nb_filters = best_params['nb_filters']
            start_time = time.time() 
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            run_nested_cv(args, data=X, label=label, n_features=nsnp, device=device, gpu_handle=gpu_handle)
            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully") 
