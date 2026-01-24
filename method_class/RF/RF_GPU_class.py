import os
import time
import psutil
import pynvml
import argparse
import random
import torch
import pandas as pd
import numpy as np
import swanlab
import cupy as cp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import RF_GPU_he_class

try:
    from cuml.ensemble import RandomForestClassifier as cuRFClassifier
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    GPU_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='RF/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--use_gpu', type=bool, default=True)
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args):
    X = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))['arr_0']
    Y = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))['arr_0']
    return X, Y


def get_gpu_mem_by_pid(pid):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def run_cv(args, X, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    le = LabelEncoder()
    y_all = le.fit_transform(label)
    num_classes = len(np.unique(y_all))
    np.save(os.path.join(result_dir, "label_mapping.npy"), le.classes_)

    print(f"Classes: {le.classes_} (n={num_classes})")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    process = psutil.Process(os.getpid())
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_all)):
        print(f"\n===== Fold {fold} =====")
        fold_start = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        X_train = cp.asarray(X_train, dtype=cp.float32)
        X_test = cp.asarray(X_test, dtype=cp.float32)
        y_train = cp.asarray(y_train, dtype=cp.int32)

        model = cuRFClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = cp.asnumpy(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        gpu_mem = get_gpu_mem_by_pid(os.getpid()) if args.use_gpu else 0.0
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, "
            f"F1={f1:.4f}, Time={fold_time:.2f}s, "
            f"GPU={gpu_mem:.2f}MB, CPU={cpu_mem:.2f}MB"
        )

        pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred)
        }).to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

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
        print(f"\n▶ Running {args.methods}{args.species}")
        label = Y[:, 0]
        label = np.nan_to_num(label, nan=np.nanmedian(label))

        best_params = RF_GPU_he_class.main(X, label)
        args.n_estimators = best_params['n_estimators']
        args.max_depth = best_params['max_depth']

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())

        start_time = time.time()
    
        run_cv(args, X, label)
        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f}s")

        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
