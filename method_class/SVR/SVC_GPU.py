import os
import time
import psutil
import argparse
import random
import torch
import numpy as np
import pandas as pd
import swanlab
import pynvml

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import cupy as cp
    from cuml.svm import SVC as cuSVC
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.svm import SVC
    CUML_AVAILABLE = False

import SVC_GPU_he


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='SVR/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--gamma', type=str, default='scale')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--use_gpu',default=True)

    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(args):
    X = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    Y = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    print(f"Samples: {X.shape[0]}, SNPs: {X.shape[1]}")
    return X, Y

def get_gpu_mem_by_pid(pid, handle):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def run_cv(args, X, y, handle=None):
    result_dir = os.path.join(args.result_dir, args.methods + args.species)
    os.makedirs(result_dir, exist_ok=True)

    use_gpu = args.use_gpu and CUML_AVAILABLE
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    le = LabelEncoder()
    y_all = le.fit_transform(y)
    np.save(os.path.join(result_dir, 'label_mapping.npy'), le.classes_)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_all)):
        fold_start = time.time()
        print(f"\n===== Fold {fold} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        if use_gpu:
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
            
            model = cuSVC(
                C=args.C,
                kernel=args.kernel,
                gamma=args.gamma,
                degree=args.degree,
                probability=True
            )
            model.fit(X_train_gpu, y_train_gpu)
            y_pred = model.predict(X_test_gpu)
            y_pred = cp.asnumpy(y_pred)
        else:
            model = SVC(
                C=args.C,
                kernel=args.kernel,
                gamma=args.gamma,
                degree=args.degree,
                probability=True
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        gpu_mem = get_gpu_mem_by_pid(os.getpid(), handle) if (use_gpu and handle) else 0.0  # 修正：检查handle
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: "
            f"ACC={acc:.4f}, "
            f"PREC={prec:.4f}, "
            f"REC={rec:.4f}, "
            f"F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, " 
            f"GPU={gpu_mem:.2f}MB, "
            f"CPU={cpu_mem:.2f}MB"
        )

        df = pd.DataFrame({
            "Y_test": le.inverse_transform(y_test),
            "Y_pred": le.inverse_transform(y_pred)
        })
        df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    # ===== Summary =====
    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")


if __name__ == "__main__":
    set_seed(42)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    args = parse_args()
    all_species = ['Human/Amd/', 'Human/BC/',"Horse/"]

    for species in all_species:
        args.species = species
        X, Y = load_data(args)        
        label = Y[:, 0]
        best_params = SVC_GPU_he.main(X, label)
        args.C = best_params['C']
        args.kernel = best_params['kernel']
        args.gamma = best_params['gamma']
        args.degree = best_params['degree']
        
        start_time = time.time()
        run_cv(args, X, label, handle)
        
        elapsed_time = time.time() - start_time
        print(f"Total running time: {elapsed_time:.2f} s")
        print("Successfully finished!")

        if CUML_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
