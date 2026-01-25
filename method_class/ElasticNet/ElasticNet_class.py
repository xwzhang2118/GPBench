import os
import time
import psutil
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import swanlab
import ElasticNet_he_class

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='ElasticNet/', help='Method name')
    parser.add_argument('--species', type=str, default='', help='Dataset name')
    parser.add_argument('--phe', type=str, default='', help='Phenotype name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
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
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def run_nested_cv(args, data, label):
    result_dir = os.path.join(args.result_dir, args.methods + args.species)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation with ElasticNet (LogisticRegression with elasticnet penalty)...")

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    time_star = time.time()
    
    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        x_train = data[train_index]
        x_test = data[test_index]
        y_train = label[train_index]
        y_test = label[test_index]

        model = LogisticRegression(
            penalty='elasticnet',
            C=args.C,
            l1_ratio=args.l1_ratio,
            solver='saga',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(x_train, y_train)
        y_test_preds = model.predict(x_test)

        acc = accuracy_score(y_test, y_test_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_test_preds, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_cpu_mem = process.memory_info().rss / 1024**2
        
        print(f'Fold {fold}: ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, '
              f'Time={fold_time:.2f}s, CPU={fold_cpu_mem:.2f}MB')
        
        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_test_preds})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

        del model, y_test_preds, x_train, x_test, y_train, y_test
    
    print("\n===== Cross-validation summary =====")
    print(f"Using sklearn LogisticRegression with elasticnet penalty")
    print(f"Average ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Average PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"Average REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"Average F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Total Time: {time.time() - time_star:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    all_species = ['Horse/', 'Human/BC/', 'Human/Amd/']
    for i in range(len(all_species)):
        args.species = all_species[i]
    
        X, Y, nsamples, nsnp, names = load_data(args)
        print("starting run " + args.methods + args.species)
        label_raw = np.nan_to_num(Y[:, 0])
        le = LabelEncoder()
        label = le.fit_transform(label_raw)
        num_classes = len(le.classes_)
        print(f"Number of classes: {num_classes}")

        best_params = ElasticNet_he_class.main(X, label)
        args.C = best_params['C']
        args.l1_ratio = best_params['l1_ratio']
        
        start_time = time.time()
        run_nested_cv(args, data=X, label=label)
        elapsed_time = time.time() - start_time
        print(f"Running time: {elapsed_time:.2f}s")
        print("successfully")
