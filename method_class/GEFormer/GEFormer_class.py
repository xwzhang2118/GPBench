import os
import torch
import swanlab
import argparse
import psutil
import time
import random
import numpy as np
import pandas as pd
import pynvml
import GEFormer_he_class

from gMLP_class import GEFormer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='GEFormer/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    x = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    y = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]
    return x, y, x.shape[0], x.shape[1], names


def get_gpu_mem_by_pid(pid):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def run_nested_cv(args, data, label, nsnp, device):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    total_start = time.time()

    num_classes = len(np.unique(label))

    for fold, (train_idx, test_idx) in enumerate(kf.split(data, label)):
        fold_start = time.time()
        process = psutil.Process(os.getpid())

        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        x_train = torch.from_numpy(X_train_sub).float().to(device)
        y_train = torch.from_numpy(y_train_sub).long().to(device)
        x_valid = torch.from_numpy(X_valid).float().to(device)
        y_valid = torch.from_numpy(y_valid).long().to(device)
        x_test  = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).long().to(device)

        train_loader = DataLoader(TensorDataset(x_train, y_train), args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_valid, y_valid), args.batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(x_test, y_test_tensor), args.batch_size, shuffle=False)

        model = GEFormer(nsnp=nsnp, num_classes=num_classes).to(device)
        model.train_model(
            train_loader, valid_loader,
            args.epoch, args.learning_rate, args.patience, device
        )

        logits = model.predict(test_loader)
        y_pred = np.argmax(logits, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        gpu_mem = get_gpu_mem_by_pid(os.getpid())
        cpu_mem = process.memory_info().rss / 1024**2
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(
            f"Fold {fold}: ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, "
            f"F1={f1:.4f}, Time={fold_time:.2f}s, "
            f"GPU={gpu_mem:.2f}MB, CPU={cpu_mem:.2f}MB"
        )

        pd.DataFrame({"Y_test": y_test, "Y_pred": y_pred}).to_csv(
            os.path.join(result_dir, f"fold{fold}.csv"), index=False
        )

    total_time = time.time() - total_start
    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {total_time:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_species =  ["Human/Sim/"]

    for species in all_species:
        args.species = species
        X, Y, _, nsnp, _ = load_data(args)
        label_raw = np.nan_to_num(Y[:, 0])
        le = LabelEncoder()
        label = le.fit_transform(label_raw)
        num_classes = len(le.classes_)

        best_params = GEFormer_he_class.Hyperparameter(X, label, nsnp)
        args.learning_rate = best_params['learning_rate']
        args.batch_size = best_params['batch_size']
        args.patience = best_params['patience']
        start_time = time.time() 
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())
        run_nested_cv(args, X, label, nsnp, device)

        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f}s")
        print("Successfully finished:", args.species, args.phe)
