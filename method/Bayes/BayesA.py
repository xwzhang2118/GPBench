import os
import time
import psutil
import argparse
import random
import torch
import numpy as np
import pandas as pd
from bayesAfromR import BayesA
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='BayesA/', help='Model name')
    parser.add_argument('--species', type=str, default='Cattle/', help='Species name')
    parser.add_argument('--phe', type=str, default='', help='Phenotype name')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to data directory')
    parser.add_argument('--result_dir', type=str, default='result/', help='Path to result directory')
    return parser.parse_args()


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    process = psutil.Process(os.getpid())

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        fold_start = time.time()
        print(f"\n===== Fold {fold} =====")
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label[train_index], label[test_index]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = BayesA(task="regression")
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        pcc, _ = pearsonr(Y_test, Y_pred)

        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}: Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')
        results_df = pd.DataFrame({'Y_test': Y_test, 'Y_pred': Y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)
        
    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Total time : {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()
    args = parse_args()

    all_species =['Cattle/', 'Chicken/', 'Chickpea/', 'Cotton/', 'Loblolly_Pine/',
                   'Maize/', 'Millet/', 'Mouse/', 'Pig/', 'Rapeseed/', 
                   'Rice/', 'Soybean/', 'Wheat/','Yeast/']
    
    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        for j in range(len(names)):
            args.phe = names[j]
            print("starting run " + args.methods + args.species + args.phe)
            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            run_nested_cv(args, data=X, label=label)
            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully")