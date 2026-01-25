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
from base_G2PDeep_class import G2PDeep, ModelHyperparams
from torch.utils.data import DataLoader, TensorDataset
import G2PDeep_he_class
import pynvml


def parse_args():
    parser = argparse.ArgumentParser(description="G2PDeep classification")
    parser.add_argument('--methods', type=str, default='G2PDeep/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()


def process_snp_data(data: np.array) -> np.array:
    nb_classes = 4
    onehot_x = np.empty(
        shape=(data.shape[0], data.shape[1], nb_classes),
        dtype=np.float32
    )

    for i in range(data.shape[0]):
        _data = pd.to_numeric(data[i], errors='coerce')
        _targets = np.array(_data).reshape(-1).astype(np.int64)
        onehot_x[i] = np.eye(nb_classes)[_targets]

    return onehot_x


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    xData[xData == -9] = 0
    xData = process_snp_data(xData)
    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    return xData, yData, nsample, nsnp, names


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

        x_train_tensor = torch.from_numpy(X_train_sub).float()
        y_train_tensor = torch.from_numpy(y_train_sub).long()
        x_valid_tensor = torch.from_numpy(X_valid).float()
        y_valid_tensor = torch.from_numpy(y_valid).long()
        x_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(
            train_data, args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        valid_loader = DataLoader(
            valid_data, args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = DataLoader(
            test_data, args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

        hp = ModelHyperparams()
        model = G2PDeep(nsnp=nsnp, num_classes=num_classes, hyperparams=hp).to(device)
        model.train_model(train_loader, valid_loader, args.epoch, args.lr, args.patience, device)
        y_pred = model.predict(test_loader, device)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = get_gpu_mem_by_pid(os.getpid(), gpu_handle)
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

    all_species = ["Human/Sim/"]
    for species in all_species:
        args.species = species
        X, Y, nsamples, nsnp, names = load_data(args)

        print("Starting:", args.methods + args.species)
        label_raw = np.nan_to_num(Y[:, 0])
        le = LabelEncoder()
        label = le.fit_transform(label_raw)
        num_classes = len(le.classes_)

        best_params = G2PDeep_he_class.main(X, label, nsnp, num_classes)
        args.lr = best_params['learning_rate']
        args.patience = best_params['patience']
        args.batch_size = best_params['batch_size']

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())
        run_nested_cv(args, data=X, label=label, nsnp=nsnp, num_classes=num_classes, device=device, gpu_handle=gpu_handle)

        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f}s")
        print("successfully")
