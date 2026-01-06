import os
import time
import psutil
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

ro.r('library(MASS)')

def gblup_r_vanraden(X_train, y_train, X_test):
    """
    VanRaden GBLUP 
    """
    ro.globalenv['X_train'] = X_train
    ro.globalenv['y_train'] = y_train
    ro.globalenv['X_test'] = X_test

    r_code = """
    n_train <- nrow(X_train)
    m <- ncol(X_train)

    # Step1: allele freq
    p <- colMeans(X_train) / 2
    p <- pmax(pmin(p, 0.99), 0.01)

    # Step2: standardized genotype
    Z_train <- sweep(X_train, 2, 2*p, "-") / sqrt(2*p*(1-p))
    Z_train[is.na(Z_train)] <- 0

    Z_test <- sweep(X_test, 2, 2*p, "-") / sqrt(2*p*(1-p))
    Z_test[is.na(Z_test)] <- 0

    # Step3: genomic relationship matrix
    G <- Z_train %*% t(Z_train) / m
    G <- G + diag(1e-6, n_train)

    # Step4: variance components
    Vu <- var(y_train)
    Ve <- Vu * 0.5
    lam <- Ve / Vu

    # Step5: BLUP solution
    A <- G + lam * diag(n_train)
    r <- y_train - mean(y_train)
    alpha <- solve(A, r)

    # Step6: predict GEBV
    G_test_train <- Z_test %*% t(Z_train) / m
    u_test <- G_test_train %*% alpha
    y_pred <- mean(y_train) + u_test

    y_pred
    """
    y_pred = np.array(ro.r(r_code)).flatten()
    return y_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='GBLUP/', help='Method name') 
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to data directory')
    parser.add_argument('--result_dir', type=str, default='result/', help='Path to result directory')
    args = parser.parse_args()
    return args


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
    torch.manual_seed(torch.tensor(seed))
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_nested_cv(args, data, label, process):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation with GBLUP (R VanRaden)...")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(label)):
        print(f"===== Fold {fold} =====")
        fold_start_time = time.time()
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        # === run strict GBLUP via R ===
        y_pred = gblup_r_vanraden(X_train, y_train, X_test)

        pcc = pearsonr(y_test, y_pred)[0]
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}: Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')

        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")


if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    args = parse_args()
    process = psutil.Process(os.getpid())
    all_species =['Cattle/', 'Chicken/', 'Chickpea/', 'Cotton/', 'Loblolly_Pine/',
                   'Maize/', 'Millet/', 'Mouse/', 'Pig/', 'Rapeseed/', 
                   'Rice/', 'Soybean/', 'Wheat/','Yeast/']

    for sp in all_species:
        args.species = sp
        X, Y, nsamples, nsnp, names = load_data(args)
        for i, phe in enumerate(names):
            args.phe = phe
            print("starting run " + args.methods + args.species + args.phe)
            label = Y[:, i]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()

            run_nested_cv(args, data=X, label=label, process=process)
            
            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully")
