import os
import time
import psutil
import swanlab
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from base_DeepCCR_class import DeepCCR
import DeepCCR_he_class

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='DeepCCR/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()

def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    print("Samples:", xData.shape[0])
    print("SNPs:", xData.shape[1])
    return xData, yData, xData.shape[0], xData.shape[1], names

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_nested_cv(args, data, label, nsnp, num_classes, device):

    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_f1, all_prec, all_rec = [], [], [], []
    time_start = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        # Tensor
        x_tr = torch.from_numpy(X_tr).float().unsqueeze(1)
        y_tr = torch.from_numpy(y_tr).long()
        x_val = torch.from_numpy(X_val).float().unsqueeze(1)
        y_val = torch.from_numpy(y_val).long()
        x_te = torch.from_numpy(X_test).float().unsqueeze(1)
        y_te = torch.from_numpy(y_test).long()

        train_loader = DataLoader(TensorDataset(x_tr, y_tr), args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_val, y_val), args.batch_size)
        test_loader  = DataLoader(TensorDataset(x_te, y_te), args.batch_size)

        model = DeepCCR(input_seq_len=nsnp, num_classes=num_classes)
        model.train_model(
            train_loader, valid_loader,
            args.epoch, args.lr, args.patience, device
        )

        y_pred = model.predict(test_loader, device)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro')

        all_acc.append(acc)
        all_f1.append(f1)
        all_prec.append(prec)
        all_rec.append(rec)

        fold_time = time.time() - fold_start
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: ACC={acc:.4f}, F1={f1:.4f}, "
            f"Prec={prec:.4f}, Rec={rec:.4f}, "
            f"Time={fold_time:.2f}s"
        )

        pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        }).to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - time_start

    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {total_time:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_species = ['Horse/', 'Human/Amd/', 'Human/BC/'] 

    for species in all_species:
        args.species = species
        X, Y, nsamples, nsnp, names = load_data(args)
        print("Starting:", args.methods + args.species)
        
        label = Y[:, 0]
        label_series = pd.Series(label)
        if label_series.isna().any():
            mode_val = label_series.mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else label_series.dropna().iloc[0] if not label_series.dropna().empty else 0
            label = label_series.fillna(fill_val).values
        else:
            label = label_series.values

        le = LabelEncoder()
        label = le.fit_transform(label)
        num_classes = len(np.unique(label))

        result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
        os.makedirs(result_dir, exist_ok=True)
        np.save(os.path.join(result_dir, 'label_mapping.npy'), le.classes_)

        best_params = DeepCCR_he_class.main(X, label, nsnp)
        args.lr = best_params['learning_rate']
        args.patience = best_params['patience']
        args.batch_size = best_params['batch_size']
        start_time = time.time()
        run_nested_cv(args, X, label, nsnp, num_classes, device)
        elapsed = time.time() - start_time

        print(f"Total running time: {elapsed:.2f}s")
        print("Successfully finished\n")
