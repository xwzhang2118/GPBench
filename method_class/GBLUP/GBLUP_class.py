import os
import time
import psutil
import swanlab
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
ro.r('library(rrBLUP)')


def gblup_classification(X_train, y_train_bin, X_test):

    # Pass data to R
    ro.globalenv['X_train'] = X_train
    ro.globalenv['y_train_bin'] = y_train_bin
    ro.globalenv['X_test'] = X_test

    r_code = """
    library(rrBLUP)
    
    n_train <- nrow(X_train)
    m <- ncol(X_train)

    # Step1: allele frequencies
    p <- colMeans(X_train) / 2
    p <- pmax(pmin(p, 0.99), 0.01)

    # Step2: VanRaden standardized genotype
    Z_train <- sweep(X_train, 2, 2*p, "-") / sqrt(2*p*(1-p))
    Z_train[is.na(Z_train)] <- 0

    Z_test <- sweep(X_test, 2, 2*p, "-") / sqrt(2*p*(1-p))
    Z_test[is.na(Z_test)] <- 0

    # Step3: Genomic relationship matrix (VanRaden method 2)
    denom <- sum(2*p*(1-p))
    G <- Z_train %*% t(Z_train) / denom
    G <- G + diag(1e-6, n_train)  # stability

    # Step4: REML GBLUP
    fit <- mixed.solve(y = y_train_bin, K = G, SE = FALSE)

    # Extract variance components and fixed effect
    Vu <- fit$Vu
    Ve <- fit$Ve
    mu <- as.numeric(fit$beta)  # intercept
    h2 <- Vu / (Vu + Ve)

    # Step5: GBLUP prediction for test set
    y_centered <- y_train_bin - mu
    A <- G + (Ve / Vu) * diag(n_train)  # G + λ I

    G_test_train <- Z_test %*% t(Z_train) / denom
    u_test <- G_test_train %*% solve(A, y_centered)  # strictly correct formula

    y_pred_score <- mu + u_test
    y_pred_score
    """

    y_pred_score = np.array(ro.r(r_code)).flatten()
    return y_pred_score


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='GBLUP/', help='Method name') 
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
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


def run_nested_cv(args, data, label, process):
    result_dir = os.path.join(args.result_dir, args.methods + args.species)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation (GBLUP Classification with R)...")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    le = LabelEncoder()
    label_all = le.fit_transform(label)
    np.save(os.path.join(result_dir, 'label_mapping.npy'), le.classes_)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, label_all)):
        fold_start = time.time()
        print(f"===== Fold {fold} =====")
        X_train, X_test = data[train_idx], data[test_idx]
        Y_train, Y_test = label_all[train_idx], label_all[test_idx]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        classes = np.unique(Y_train)
        scores = np.zeros((len(classes), X_test.shape[0]))
        for idx, cls in enumerate(classes):
            y_train_bin = (Y_train == cls).astype(float)
            scores[idx, :] = gblup_classification(X_train, y_train_bin, X_test)

        Y_pred = np.argmax(scores, axis=0)

        acc = accuracy_score(Y_test, Y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='macro', zero_division=0)
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}: ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, '
              f'Time={fold_time:.2f}s, GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')

        Y_test_orig = le.inverse_transform(Y_test)
        Y_pred_orig = le.inverse_transform(Y_pred)
        results_df = pd.DataFrame({'Y_test': Y_test_orig, 'Y_pred': Y_pred_orig})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average ACC: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Average PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"Average REC: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"Average F1 : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()
    args = parse_args()
    process = psutil.Process(os.getpid())

    all_species = ['Human/Amd/', 'Human/BC/',"Horse/"]
    for sp in all_species:
        args.species = sp
        X, Y, nsamples, nsnp, names = load_data(args)
        print("Starting run " + args.methods + args.species)
        label = Y[:, 0]
        s = pd.Series(label)
        fill_val = s.mode().iloc[0] if not s.dropna().empty else 0
        label = np.nan_to_num(label, nan=fill_val)

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        run_nested_cv(args, data=X, label=label, process=process)

        elapsed_time = time.time() - start_time
        print(f"Total running time: {elapsed_time:.2f} s")
        print("Successfully finished!")