import os
import time
import psutil
import swanlab
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from base_CropARNet_class import SimpleSNPModel
import CropARNet_he_class
import pynvml

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='CropARNet/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=50)
    return parser.parse_args()

def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    print("Number of samples:", xData.shape[0])
    print("Number of SNPs:", xData.shape[1])
    return xData, yData, xData.shape[0], xData.shape[1], names

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_mem_by_pid(pid, handle):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def run_nested_cv(args, data, label, nsnp, num_classes, device, handle=None, le=None):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    time_star = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"\nRunning fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        x_train = torch.from_numpy(X_tr).float().to(device)
        y_train_t = torch.from_numpy(y_tr).long().to(device)
        x_valid = torch.from_numpy(X_val).float().to(device)
        y_valid_t = torch.from_numpy(y_val).long().to(device)
        x_test  = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).long().to(device)

        train_loader = DataLoader(TensorDataset(x_train, y_train_t), args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_valid, y_valid_t), args.batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(x_test, y_test_t), args.batch_size, shuffle=False)

        model = SimpleSNPModel(num_snps = nsnp, num_classes=num_classes)
        model.train_model(
            train_loader,
            valid_loader,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            args.patience,
            device
        )

        y_pred = model.predict(test_loader)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        gpu_mem = get_gpu_mem_by_pid(os.getpid(), handle) if handle else 0.0
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: ACC={acc:.4f}, PREC={prec:.4f}, "
            f"REC={rec:.4f}, F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, GPU={gpu_mem:.2f}MB, CPU={cpu_mem:.2f}MB"
        )

        pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred)
        }).to_csv(
            os.path.join(result_dir, f"fold{fold}.csv"), index=False
        )

        torch.cuda.empty_cache()

    print("\n===== Cross-validation summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_species = ['Human/Amd/', 'Human/BC/', "Horse/"]

    for sp in all_species:
        args.species = sp
        X, Y, _, nsnp, names = load_data(args)

        print("Starting:", args.methods + args.species)
        if Y.ndim == 1:
            label = Y
        else:
            label = Y[:, 0]
        
        label = np.nan_to_num(label, nan=np.nanmean(label))
        
        le = LabelEncoder()
        label = le.fit_transform(label)
        num_classes = len(np.unique(label))
        
        result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
        os.makedirs(result_dir, exist_ok=True)
        np.save(os.path.join(result_dir, 'label_mapping.npy'), le.classes_)

        best_params = CropARNet_he_class.main(X, label, nsnp)
        args.learning_rate = best_params["learning_rate"]
        args.batch_size = best_params["batch_size"]
        args.weight_decay = best_params["weight_decay"]
        args.patience = best_params["patience"]
        start_time = time.time() 
        torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())

        run_nested_cv(args, X, label, nsnp, num_classes, device, handle, le)
        elapsed_time = time.time() - start_time
        print(f"运行时间: {elapsed_time:.2f} 秒")
        print("successfully")
