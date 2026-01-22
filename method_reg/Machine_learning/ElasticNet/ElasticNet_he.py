import gc
import random
import time
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from optuna.exceptions import TrialPruned

def run_nested_cv_with_early_stopping(data, label, outer_cv, alpha, l1_ratio):
    best_corr_coefs = []
    best_maes = []
    best_r2s = []
    best_mses = []
    time_star = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data)):
        x_train = data[train_idx]
        x_test = data[test_idx]
        y_train = label[train_idx]
        y_test = label[test_idx]

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
        y_test_preds = model.predict(x_test)

        pcc, _ = pearsonr(y_test, y_test_preds)
        mse = mean_squared_error(y_test, y_test_preds)
        r2 = r2_score(y_test, y_test_preds)
        mae = mean_absolute_error(y_test, y_test_preds)
        
        best_corr_coefs.append(pcc)
        best_maes.append(mae)
        best_r2s.append(r2)
        best_mses.append(mse)

        print(f'Fold {fold + 1}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Corr={pcc:.4f}')
        del model, y_test_preds, x_train, x_test, y_train, y_test

    print("==== Final Results ====")
    print(f"MAE: {np.mean(best_maes):.4f} ± {np.std(best_maes):.4f}")
    print(f"MSE: {np.mean(best_mses):.4f} ± {np.std(best_mses):.4f}")
    print(f"R2 : {np.mean(best_r2s):.4f} ± {np.std(best_r2s):.4f}")
    print(f"Corr: {np.mean(best_corr_coefs):.4f} ± {np.std(best_corr_coefs):.4f}")

    print(f"Time: {time.time() - time_star:.2f}s")
    gc.collect()
    
    return np.mean(best_corr_coefs)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def main(data, label):
    set_seed(42)
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        l1_ratio = trial.suggest_categorical("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
        
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

        try:
            corr_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                outer_cv=outer_cv,
                alpha=alpha,
                l1_ratio=l1_ratio
            )
        except TrialPruned:
            return float("-inf")
        return corr_score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == '__main__':
    main()
