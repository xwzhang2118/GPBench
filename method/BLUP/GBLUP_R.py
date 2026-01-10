import os
import time
import psutil
import swanlab
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# rpy2 导入
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

# 为 BLUP 求逆
ro.r('library(MASS)')

def gblup_r_vanraden_reml(X_train, y_train, X_test):
    """
    VanRaden I GBLUP with REML-estimated variance components
    Strictly correct prediction for test set.

    Args:
        X_train: numpy array, n_train x m markers
        y_train: numpy array, n_train x 1 phenotype
        X_test:  numpy array, n_test x m markers

    Returns:
        y_pred: numpy array, predicted GEBV for X_test
    """
    import numpy as np
    import rpy2.robjects as ro

    # Pass data to R
    ro.globalenv['X_train'] = X_train
    ro.globalenv['y_train'] = y_train
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
    fit <- mixed.solve(y = y_train, K = G, SE = FALSE)

    # Extract variance components and fixed effect
    Vu <- fit$Vu
    Ve <- fit$Ve
    mu <- as.numeric(fit$beta)  # <-- 转成标量，避免非兼容数组
    h2 <- Vu / (Vu + Ve)

    # Step5: GBLUP prediction for test set
    y_centered <- y_train - mu
    A <- G + (Ve / Vu) * diag(n_train)  # G + λ I

    G_test_train <- Z_test %*% t(Z_train) / denom
    u_test <- G_test_train %*% solve(A, y_centered)  # strictly correct formula

    y_pred <- mu + u_test
    y_pred
    """

    y_pred = np.array(ro.r(r_code)).flatten()
    return y_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='GBLUP_R/', help='Method name') 
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='/home/common/xwzhang/Project/Benchmark/data/', help='Path to data directory')
    parser.add_argument('--result_dir', type=str, default='/home/common/xwzhang/Project/Benchmark/result/', help='Path to result directory')
    args = parser.parse_args()
    return args


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))["arr_0"]
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
        y_pred = gblup_r_vanraden_reml(X_train, y_train, X_test)

        # 评价指标
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

        swanlab.log({
            "fold": fold,
            "corr": pcc,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "time_sec": fold_time,
            "gpu_mem_MB": fold_gpu_mem,
            "cpu_mem_MB": fold_cpu_mem,
        })

        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")

    swanlab.log({
        "final/mae_mean": np.mean(all_mae),
        "final/mae_std": np.std(all_mae),
        "final/mse_mean": np.mean(all_mse),
        "final/mse_std": np.std(all_mse),
        "final/r2_mean": np.mean(all_r2),
        "final/r2_std": np.std(all_r2),
        "final/corr_mean": np.mean(all_pcc),
        "final/corr_std": np.std(all_pcc),
    })


if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    args = parse_args()
    process = psutil.Process(os.getpid())
    all_species = ['Chicken/', 'Cotton/GSTP010/', 'Cattle/', 'Loblolly_Pine/',
                   'Millet/GSTP011/', 'Mouse/',  'Rapeseed/GSTP013/', 
                   'Rice/GSTP008/', 'Pig/','Maize/GSTP003/', 'Chickpea/GSTP012/', 
                   'Soybean/GSTP014/', 'Wheat/','Yeast/']

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

            swanlab.init(
                project="Benchmark_ST_Method_L40",
                entity="xwzhang",
                name=args.methods + args.species + args.phe,
                config={
                    "cv_splits": 10,
                }
            )

            run_nested_cv(args, data=X, label=label, process=process)
            
            elapsed_time = time.time() - start_time
            print(f"运行时间: {elapsed_time:.2f} 秒")
            swanlab.log({"final/all_time": elapsed_time})
            
            swanlab.finish()
            print("successfully")