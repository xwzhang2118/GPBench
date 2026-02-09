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

from base_dnngp_class import DNNGP
import DNNGP_he_class

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='DNNGP/')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout1', type=float, default=0.5)
    parser.add_argument('--dropout2', type=float, default=0.5)
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

def run_nested_cv(args, data, label, nsnp, device, le):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    num_classes = len(np.unique(label))

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    time_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(data, label)):
        print(f"\n===== Fold {fold} =====")
        fold_start = time.time()
        process = psutil.Process(os.getpid())

        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        # tensor
        x_train = torch.from_numpy(X_train_sub).float().unsqueeze(1).to(device)
        y_train = torch.from_numpy(y_train_sub).long().to(device)
        x_valid = torch.from_numpy(X_valid).float().unsqueeze(1).to(device)
        y_valid = torch.from_numpy(y_valid).long().to(device)
        x_test  = torch.from_numpy(X_test).float().unsqueeze(1).to(device)
        y_test  = torch.from_numpy(y_test).long().to(device)

        train_loader = DataLoader(TensorDataset(x_train, y_train), args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(x_valid, y_valid), args.batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(x_test, y_test), args.batch_size, shuffle=False)
        model = DNNGP(nsnp, args.dropout1, args.dropout2, output_dim=num_classes).to(device)

        model.train_model(
            train_loader,
            valid_loader,
            args.epoch,
            args.lr,
            args.weight_decay,
            args.patience,
            device
        )

        y_pred = model.predict(test_loader)
        y_pred_cls = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test.cpu().numpy(), y_pred_cls)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test.cpu().numpy(), y_pred_cls,
            average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        print(
            f"ACC={acc:.4f}, PREC={prec:.4f}, "
            f"REC={rec:.4f}, F1={f1:.4f}, "
            f"Time={time.time()-fold_start:.2f}s, "
            f"CPU={process.memory_info().rss/1024**2:.2f}MB"
        )

        pd.DataFrame({
            "Y_test": le.inverse_transform(y_test.cpu().numpy()),
            "Y_pred": le.inverse_transform(y_pred_cls)
        }).to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== CV Summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_species =  ["Human/Sim/"]

    for species in all_species:
        args.species = species
        X, Y, nsamples, nsnp, names = load_data(args)
        print("Starting:", args.methods + args.species)

        label = Y[:, 0]
        label = np.nan_to_num(label, nan=np.nanmean(label))
        le = LabelEncoder()
        label = le.fit_transform(label)
        num_classes = len(np.unique(label))

        best_params = DNNGP_he_class.Hyperparameter(X, label, nsnp)
        args.lr = best_params['learning_rate']
        args.weight_decay = best_params['weight_decay']
        args.patience = best_params['patience']
        args.dropout1 = best_params['dropout1']
        args.dropout2 = best_params['dropout2']
        start_time = time.time() 
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())

        run_nested_cv(args, X, label, nsnp, device, le)
        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f} s")
        print("successfully")