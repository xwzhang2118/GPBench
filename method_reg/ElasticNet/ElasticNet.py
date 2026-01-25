import os
import time
import psutil
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ElasticNet_he

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='ElasticNet/', help='Method name')
    parser.add_argument('--species', type=str, default='', help='Dataset name')
    parser.add_argument('--phe', type=str, default='', help='Phenotype name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--alpha', type=float, default=0.5, help='Regularization strength')
    parser.add_argument('--l1_ratio', type=float, default=0.5, help='L1 ratio (0=Ridge, 1=Lasso)')
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


def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation with ElasticNet (sklearn)...")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
    
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()
        
        x_train = data[train_index]
        x_test = data[test_index]
        y_train = label[train_index]
        y_test = label[test_index]

        model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
        y_test_preds = model.predict(x_test)

        pcc, _ = pearsonr(y_test, y_test_preds)
        mse = mean_squared_error(y_test, y_test_preds)
        r2 = r2_score(y_test, y_test_preds)
        mae = mean_absolute_error(y_test, y_test_preds)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_cpu_mem = process.memory_info().rss / 1024**2
        
        print(f'Fold {fold}: Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, '
              f'Time={fold_time:.2f}s, CPU={fold_cpu_mem:.2f}MB')

        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_test_preds})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)
        
        del model, y_test_preds, x_train, x_test, y_train, y_test
    
    print("\n===== Cross-validation summary =====")
    print(f"Using sklearn ElasticNet")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Total Time: {time.time() - time_star:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    all_species =['Cotton/']
    
    for i in range(len(all_species)):
        args.species = all_species[i]
        X, Y, nsamples, nsnp, names = load_data(args)
        for j in range(len(names)):
            args.phe = names[j]
            print("starting run " + args.methods + args.species + args.phe)
            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            
            best_params = ElasticNet_he.main(X, label)
            args.alpha = best_params['alpha']
            args.l1_ratio = best_params['l1_ratio']
            
            start_time = time.time()
            run_nested_cv(args, data=X, label=label)
            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully") 