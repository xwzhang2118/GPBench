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
from base_deepgs_class import DeepGS
import DeepGS_he_class
import pynvml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='DeepGS/')
    parser.add_argument('--species', type=str, default='Wheat/')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--num_round', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=50)
    return parser.parse_args()

def load_data(args):
    X = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    Y = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]
    print("Samples:", X.shape[0], "SNPs:", X.shape[1])
    return X, Y, X.shape[0], X.shape[1], names

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

def run_nested_cv(args, data, label, nsnp, num_classes, device, gpu_handle=None):

    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    cv_start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(data, label)):
        fold_start = time.time()
        process = psutil.Process(os.getpid())
        print(f"\n===== Fold {fold} =====")

        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=42
        )

        x_tr = torch.from_numpy(X_tr).float().unsqueeze(1).to(device)
        x_val = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
        x_te = torch.from_numpy(X_test).float().unsqueeze(1).to(device)

        y_tr = torch.from_numpy(y_tr).long().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        y_te = torch.from_numpy(y_test).long().to(device)

        train_loader = DataLoader(TensorDataset(x_tr, y_tr), args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_val, y_val), args.batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(x_te, y_te), args.batch_size, shuffle=False)

        model = DeepGS(nsnp, num_classes=num_classes)

        model.train_model(
            train_loader,
            valid_loader,
            args.num_round,
            args.learning_rate,
            args.momentum,
            args.weight_decay,
            args.patience,
            device
        )

        y_pred = model.predict(test_loader)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        gpu_mem = get_gpu_mem_by_pid(os.getpid(), gpu_handle)
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: "
            f"ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, GPU={gpu_mem:.2f}MB, CPU={cpu_mem:.2f}MB"
        )

        pd.DataFrame({
            "Y_test": y_test,
            "Y_pred": y_pred
        }).to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

        torch.cuda.empty_cache()

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

    all_species =  ["Human/Sim/"]

    for species in all_species:
        args.species = species
        X, Y, nsamples, nsnp, names = load_data(args)

        print("Starting:", args.methods + args.species)
        label_raw = np.nan_to_num(Y[:, 0])
        le = LabelEncoder()
        label = le.fit_transform(label_raw)
        num_classes = len(le.classes_)

        best_params = DeepGS_he_class.main(X, label, nsnp)
        args.learning_rate = best_params['learning_rate']
        args.batch_size = best_params['batch_size']
        args.momentum = best_params['momentum']
        args.weight_decay = best_params['weight_decay']
        args.patience = best_params['patience']

        start_time = time.time()
        run_nested_cv(args, X, label, nsnp, num_classes, device, gpu_handle)
        elapsed_time = time.time() - start_time
        print(f"Total running time: {elapsed_time:.2f}s")
        print("Successfully finished:", args.species, args.phe)
