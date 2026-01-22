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
from AlexNet_206_class import AlexNet
from torch.utils.data import DataLoader, TensorDataset
import SoyDNGP_he_class
import pynvml

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='SoyDNGP/', help='Random seed')
    parser.add_argument('--species', type=str, default='Chicken/', help='Species name')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/home/common/xwzhang/Project/Benchmark/data/')
    parser.add_argument('--result_dir', type=str, default='/home/common/xwzhang/Project/Benchmark/result/')
    
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    args = parser.parse_args()
    return args

def get_data(dataframe):
    data_matrix = np.array(dataframe)
    total_sample, total_snp = data_matrix.shape

    one_hot = np.zeros((total_sample, total_snp, 3), dtype=np.float32)
    one_hot[data_matrix == 2] = [1, 1, 0]
    one_hot[data_matrix == 1] = [1, 0, 1]
    one_hot[data_matrix == 0] = [0, 1, 1]

    target_snp = 206 * 206
    if total_snp != target_snp:
        print(f"⚠ SNP Number {total_snp} != {target_snp}")
        new_one_hot = np.zeros((total_sample, target_snp, 3), dtype=np.float32)
        copy_len = min(total_snp, target_snp)
        new_one_hot[:, :copy_len] = one_hot[:, :copy_len]
        one_hot = new_one_hot

    one_hot = one_hot.reshape(total_sample, 206, 206, 3)
    return one_hot


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    xData = get_data(xData)
    return xData, yData, nsample, nsnp, names

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_mem_by_pid(pid):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0


def run_nested_cv(args, data, label, nsnp, num_classes, device):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    time_star = time.time()
    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )
        
        x_train_tensor = torch.from_numpy(X_train_sub).float().to(device)
        y_train_tensor = torch.from_numpy(y_train_sub).long().to(device)
        x_valid_tensor = torch.from_numpy(X_valid).float().to(device)
        y_valid_tensor = torch.from_numpy(y_valid).long().to(device)
        x_test_tensor = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).long().to(device)

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=False)

        model = AlexNet(num_classes=num_classes)
        model.train_model(train_loader, valid_loader, args.epochs, args.learning_rate, args.patience, device)
        y_pred = model.predict(test_loader)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = get_gpu_mem_by_pid(os.getpid())
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, '
              f'Time={fold_time:.2f}s, GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')
       
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average ACC: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Average PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"Average REC: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"Average F1 : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")


if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()  
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    args = parse_args()
    all_species = ['Horse/',"Human/Amd/",'Human/BC/']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        print("Starting:", args.methods + args.species)
        
        label_raw = np.nan_to_num(Y[:, 0])
        le = LabelEncoder()
        label = le.fit_transform(label_raw)
        num_classes = len(le.classes_)
        
        best_params = SoyDNGP_he_class.main(X, label, nsnp, num_classes)
        args.learning_rate = best_params['learning_rate']
        args.batch_size = best_params['batch_size']
        args.patience = best_params['patience']
        
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())

        run_nested_cv(args, data=X, label=label, nsnp=nsnp, num_classes=num_classes, device=device)

        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f}s")
        print("successfully")