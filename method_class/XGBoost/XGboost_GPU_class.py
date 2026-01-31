import os
import time
import torch
import psutil
import argparse
import random
import xgboost as xgb
import numpy as np
import pandas as pd
import pynvml
import swanlab

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import XGboost_GPU_he_class

# =======================
# Argument parser
# =======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='XGBoost/')
    parser.add_argument('--species', type=str, default='Horse/')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--reg_alpha', type=float, default=0)
    parser.add_argument('--reg_lambda', type=float, default=1)

    parser.add_argument('--use_gpu', action='store_true')
    return parser.parse_args()

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_cv(args, X, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    le = LabelEncoder()
    y_all = le.fit_transform(label)
    np.save(os.path.join(result_dir, 'label_mapping.npy'), le.classes_)
    num_classes = len(np.unique(y_all))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ===== GPU / CPU =====
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        tree_method = "hist"
        device = "cuda:0"
        print("🚀 Using GPU XGBoost")
    else:
        tree_method = "hist"
        device = "cpu"
        print("⚠ Using CPU XGBoost")

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_all)):
        print(f"\n===== Fold {fold} =====")
        fold_start = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # ===== Objective =====
        if num_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        model = xgb.XGBClassifier(
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
            num_class=num_classes if num_classes > 2 else None,
            tree_method=tree_method,
            device=device,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # ===== Prediction =====
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)

        # ===== Metrics =====
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        gpu_mem = get_gpu_mem_by_pid(os.getpid(), handle) if use_gpu else 0.0
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, "
            f"F1={f1:.4f},  Time={fold_time:.2f}s"
        )

        df = pd.DataFrame({
            "Y_test": le.inverse_transform(y_test),
            "Y_pred": le.inverse_transform(y_pred)
        })
        df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    args = parse_args()

    all_species = ["Human/Sim/"] 
    for species in all_species:
        args.species = species
        X, Y = load_data(args)
        print(f"\n=== Running {args.methods}{args.species}{args.phe} ===")
        label = Y[:, 0]

        best_params = XGboost_GPU_he_class.Hyperparameter(X, label)
        args.learning_rate = best_params['learning_rate']
        args.n_estimators = best_params['n_estimators']
        args.max_depth = best_params['max_depth']
        args.subsample = best_params['subsample']
        args.colsample_bytree = best_params['colsample_bytree']
        args.gamma = best_params['gamma']
        args.reg_alpha = best_params['reg_alpha']
        args.reg_lambda = best_params['reg_lambda']
        args.min_child_weight = best_params['min_child_weight']
        
        start_time = time.time()
        run_cv(args, X, label)
        
        elapsed_time = time.time() - start_time
        print(f"Total running time: {elapsed_time:.2f} s")
        print("✔ Finished successfully")
