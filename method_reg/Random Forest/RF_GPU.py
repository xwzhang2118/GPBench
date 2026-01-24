import os
import time
import psutil
import pynvml
import argparse
import random
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import RF_Hyperparameters
import cupy as cp


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='RandomForest/', help='Method name')
    parser.add_argument('--species', type=str, default='Cattle/', help='Species name')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of CPU cores to use (-1 for all cores)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU acceleration')
    args = parser.parse_args()
    return args

def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    return xData, yData, nsample, nsnp, names

def get_gpu_mem_by_pid(pid):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    import cuml
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation with Random Forest...")
       
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()
        
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train_scaled = y_train.astype(np.float32)
        y_test_scaled = y_test.astype(np.float32)

        x_train_gpu = cp.asarray(x_train)
        x_test_gpu = cp.asarray(x_test)
        y_train_gpu = cp.asarray(y_train_scaled)
        
        model = cuRandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_streams=1
        )

        model.fit(x_train_gpu, y_train_gpu)
        
        y_test_preds = model.predict(x_test_gpu)
        y_test_preds = cp.asnumpy(y_test_preds)
        y_test_scaled_cpu = cp.asnumpy(cp.asarray(y_test_scaled))

        y_test_original = y_test_scaled_cpu.reshape(-1)
        y_pred = y_test_preds.reshape(-1)

        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        pcc, _ = pearsonr(y_test_original, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = get_gpu_mem_by_pid(os.getpid())
        fold_cpu_mem = process.memory_info().rss / 1024**2
        
        acceleration_status = "GPU" if GPU_AVAILABLE else f"CPU({args.n_jobs} cores)"
        print(f'Fold {fold}[{acceleration_status}]: Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')
        
        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_test_preds})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    acceleration_status = "GPU" if GPU_AVAILABLE else f"CPU({args.n_jobs} cores)"
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    all_species =['Cattle/', 'Chicken/', 'Chickpea/', 'Cotton/', 'Loblolly_Pine/',
                   'Maize/', 'Millet/', 'Mouse/', 'Pig/', 'Rapeseed/', 
                   'Rice/', 'Soybean/', 'Wheat/','Yeast/']
    
    args = parse_args()
    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        
        for j in range(len(names)):
            args.phe = names[j]
            print(f"Starting run: {args.methods}{args.species}{args.phe}")

            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            best_params = RF_Hyperparameters.main(X, label)
            args.n_estimators = best_params['n_estimators']
            args.max_depth = best_params['max_depth']
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            run_nested_cv(args, data=X, label=label)
            
            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully")