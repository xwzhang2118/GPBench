import os
import time
import psutil
import swanlab
import argparse
import random
import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
import subprocess
import threading
import queue

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import LightGBM_he_class

class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.2):
        self.gpu_id = gpu_id
        self.interval = interval
        self.max_memory = 0
        self.current_memory = 0
        self.monitoring = False
        self.pid = os.getpid()

    def _get_gpu_memory_by_pid(self):
        try:
            result = subprocess.check_output([
                'nvidia-smi',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,nounits,noheader'
            ])
            lines = result.decode().strip().split('\n')
            for line in lines:
                pid, mem = line.split(',')
                if int(pid.strip()) == self.pid:
                    return int(mem.strip())
            return 0
        except Exception:
            return 0

    def _monitor_loop(self):
        while self.monitoring:
            mem = self._get_gpu_memory_by_pid()
            self.current_memory = mem
            self.max_memory = max(self.max_memory, mem)
            time.sleep(self.interval)

    def start(self):
        self.max_memory = 0
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        self.thread.join(timeout=1)
        return self.max_memory
gpu_monitor = GPUMonitor()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='LightGBM/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--min_data_in_leaf', type=int, default=20)
    parser.add_argument('--max_depth', type=int, default=-1)
    parser.add_argument('--lambda_l1', type=float, default=0.0)
    parser.add_argument('--lambda_l2', type=float, default=0.0)
    parser.add_argument('--min_gain_to_split', type=float, default=0.0)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_freq', type=int, default=1)
    parser.add_argument('--num_boost_round', type=int, default=200)

    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    X = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    Y = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    return X, Y

def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    le = LabelEncoder()
    y_all = le.fit_transform(label)
    n_classes = len(np.unique(y_all))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    process = psutil.Process(os.getpid())

    params = {
        'objective': 'binary' if n_classes == 2 else 'multiclass',
        'metric': 'binary_logloss' if n_classes == 2 else 'multi_logloss',
        'num_class': n_classes if n_classes > 2 else None,
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
        'device_type': 'gpu',
        'gpu_device_id': 0,
        'num_threads': 8,
        'verbose': -1
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(data, y_all)):
        print(f"\n===== Fold {fold} =====")
        start_time = time.time()

        gpu_monitor.start()
        cpu_mem_before = process.memory_info().rss / 1024**2

        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        test_set = lgb.Dataset(X_test, label=y_test)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[test_set]
        )

        y_prob = model.predict(X_test)
        if n_classes == 2:
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - start_time
        gpu_mem = gpu_monitor.stop()
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, "
            f"F1={f1:.4f}, Time={fold_time:.2f}s, "
            f"GPU={gpu_mem:.2f}MB, CPU={cpu_mem:.2f}MB"
        )

        pd.DataFrame({
            "Y_test": le.inverse_transform(y_test),
            "Y_pred": le.inverse_transform(y_pred)
        }).to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    all_species = ['Human/Amd/', 'Human/BC/', "Horse/"]

    for species in all_species:
        args.species = species
        X, Y = load_data(args)
        label = Y[:, 0]

        best_params = LightGBM_he_class.main(X, label)
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
        run_nested_cv(args, X, label)
        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f}s")
        print("successfully")
