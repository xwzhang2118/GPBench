import os
import time
import psutil
import argparse
import random
import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import subprocess
import threading
import queue
import LightGBM_Hyperparameters

class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.5):
        self.gpu_id = gpu_id
        self.interval = interval
        self.max_memory = 0
        self.current_memory = 0
        self.monitoring = False
        self.pid = os.getpid()
        self.queue = queue.Queue()
    
    def _get_gpu_memory_by_pid(self):
        try:
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-compute-apps=pid,used_memory,gpu_bus_id',
                '--format=csv,nounits,noheader'
            ], timeout=5)
            
            lines = result.decode('utf-8').strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    pid = int(parts[0].strip())
                    if pid == self.pid:
                        mem_str = parts[1].strip()
                        mem_value = ''.join(filter(str.isdigit, mem_str))
                        if mem_value:
                            return int(mem_value)
            return 0
        except Exception as e:
            print(f"GPU memory query error: {e}")
            return 0
    
    def _monitor_loop(self):
        while self.monitoring:
            try:
                mem = self._get_gpu_memory_by_pid()
                self.current_memory = mem
                if mem > self.max_memory:
                    self.max_memory = mem
                time.sleep(self.interval)
            except Exception as e:
                print(f"Monitor loop error: {e}")
                break
    
    def start(self):
        self.max_memory = 0
        self.current_memory = 0
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        return self.max_memory

gpu_monitor = GPUMonitor(gpu_id=0, interval=0.2)

def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM GPU Benchmark")
    parser.add_argument('--methods', type=str, default='LightGBM/', help='Method name')
    parser.add_argument('--species', type=str, default='', help='Dataset name')
    parser.add_argument('--phe', type=str, default='', help='Phenotype')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_leaves', type=int, default=10)
    parser.add_argument('--min_data_in_leaf', type=int, default=1)
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--lambda_l1', type=float, default=0.1)
    parser.add_argument('--lambda_l2', type=float, default=0.1)
    parser.add_argument('--min_gain_to_split', type=float, default=0.1)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_freq', type=int, default=1)
    parser.add_argument('--num_boost_round', type=int, default=100)
    parser.add_argument('--objective', type=str, default='regression')
    parser.add_argument('--device_type', type=str, default='gpu')
    parser.add_argument('--early_stopping_rounds', type=int, default=50)
    return parser.parse_args()

def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print(f"Number of samples: {nsample}, SNPs: {nsnp}")
    return xData, yData, nsample, nsnp, names

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()

    params = {
        'objective': args.objective,
        'metric': 'rmse',
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,
        'min_data_in_leaf': args.min_data_in_leaf,
        'max_depth': args.max_depth,
        'lambda_l1': args.lambda_l1,
        'lambda_l2': args.lambda_l2,
        'min_gain_to_split': args.min_gain_to_split,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': args.bagging_freq,
        'num_boost_round': args.num_boost_round,
        'device_type': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_threads': 8,
        'verbose': -1
    }

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"\n===== Running fold {fold} =====")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        gpu_monitor.start()
        time.sleep(0.5)

        cpu_mem_before = process.memory_info().rss / 1024**2

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        train_set = lgb.Dataset(X_train, label=y_train)
        test_set = lgb.Dataset(X_test, label=y_test)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[test_set]
        )

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = gpu_monitor.stop()   
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')

        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Total Time : {time.time() - time_star:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    all_species =['Cattle/', 'Chicken/', 'Chickpea/', 'Cotton/', 'Loblolly_Pine/',
                   'Maize/', 'Millet/', 'Mouse/', 'Pig/', 'Rapeseed/', 
                   'Rice/', 'Soybean/', 'Wheat/','Yeast/']
    
    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        for j in range(len(names)):
            args.phe = names[j]
            print(f"Starting run: {args.methods}{args.species}{args.phe}")
            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))

            best_params = LightGBM_Hyperparameters.main(X, label)
            args.learning_rate = best_params['learning_rate']
            args.num_leaves = best_params['num_leaves']
            args.min_data_in_leaf = best_params['min_data_in_leaf']
            args.max_depth = best_params['max_depth']
            args.lambda_l1 = best_params['lambda_l1']
            args.lambda_l2 = best_params['lambda_l2']
            args.min_gain_to_split = best_params['min_gain_to_split']
            args.feature_fraction = best_params['feature_fraction']
            args.bagging_fraction = best_params['bagging_fraction']
            args.bagging_freq = best_params['bagging_freq']
            start_time = time.time()
            run_nested_cv(args, data=X, label=label)
            elapsed_time = time.time() - start_time

            print(f"running time: {elapsed_time:.2f} s")
            print("✅ Successfully finished.\n")
