import os
import time
import psutil
import swanlab
import argparse
import random
import torch
import numpy as np
import pandas as pd
import sys
from bayesAfromR import BayesA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='BayesA/', help='Model name')
    parser.add_argument('--species', type=str, default='Human/', help='Species name')
    parser.add_argument('--phe', type=str, default='', help='Phenotype name')
    parser.add_argument('--task', type=str, default='classification', choices=['regression','classification'], help='Task: regression or classification')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to data directory')
    parser.add_argument('--result_dir', type=str, default='result/', help='Path to result directory')
    return parser.parse_args()


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genotype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]
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


def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    le = LabelEncoder()
    label_all = le.fit_transform(label)

    np.save(os.path.join(result_dir, 'label_mapping.npy'), le.classes_)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for fold, (train_index, test_index) in enumerate(kf.split(data, label_all)):
        fold_start = time.time()
        print(f"\n===== Fold {fold} =====")
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label_all[train_index], label_all[test_index]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        classes = np.unique(Y_train)
        scores = np.zeros((len(classes), X_test.shape[0]))
        for idx, cls in enumerate(classes):
            y_train_bin = (Y_train == cls).astype(float)
            model_k = BayesA(task="regression")
            model_k.fit(X_train, y_train_bin)
            scores[idx, :] = model_k.predict(X_test)

        Y_pred = np.argmax(scores, axis=0)

        acc = accuracy_score(Y_test, Y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(Y_test, Y_pred)

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}: ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')

   
        # ========== 保存预测结果 ==========
        Y_test_orig = le.inverse_transform(Y_test)
        Y_pred_orig = le.inverse_transform(Y_pred)
        results_df = pd.DataFrame({'Y_test': Y_test_orig, 'Y_pred': Y_pred_orig})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average ACC: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Average PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"Average REC: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"Average F1 : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Total time : {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()
    args = parse_args()
    all_species = ["Human/Sim/"]
    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        args.phe = names
        print("Starting run " + args.methods + args.species)
        label = Y[:, 0]

        if args.task == 'classification':
            s = pd.Series(label)
            fill_val = s.mode().iloc[0] if not s.dropna().empty else 0
            label = np.nan_to_num(label, nan=fill_val)

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())

        run_nested_cv(args, data=X, label=label)

        elapsed_time = time.time() - start_time
        print(f"Total running time: {elapsed_time:.2f} s")
        print("Successfully finished!")
