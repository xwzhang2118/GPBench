import gc
import random
import torch
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

try:
    import cupy as cp
    from cuml.svm import SVR as cuSVR
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

def run_nested_cv_with_early_stopping(data, label, outer_cv, C, epsilon, kernel, gamma, degree):
    best_corr_coefs = []
    best_maes = []
    best_r2s = []
    best_mses = []

    import time
    time_star = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data)):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        x_train_gpu = cp.asarray(x_train,  dtype=cp.float32)
        x_test_gpu = cp.asarray(x_test,  dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train.reshape(-1, 1),  dtype=cp.float32)
        y_test_gpu = cp.asarray(y_test.reshape(-1, 1), dtype=cp.float32)

        model = cuSVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)
        model.fit(x_train_gpu, y_train_gpu)

        y_test_preds = model.predict(x_test_gpu)

        y_test_preds = cp.asnumpy(y_test_preds).reshape(-1)
        y_test_scaled = cp.asnumpy(y_test_gpu).reshape(-1)
 
        mse = mean_squared_error(y_test_scaled, y_test_preds)
        r2 = r2_score(y_test_scaled, y_test_preds)
        mae = mean_absolute_error(y_test_scaled, y_test_preds)
        pcc, _ = pearsonr(y_test_scaled, y_test_preds)

        best_corr_coefs.append(pcc)
        best_maes.append(mae)
        best_r2s.append(r2)
        best_mses.append(mse)

        print(f'Fold {fold + 1}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Corr={pcc:.4f}')

        del model, x_train_gpu, x_test_gpu, y_train_gpu, y_test_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    print("==== Final Results ====")
    print(f"MAE: {np.mean(best_maes):.4f} ± {np.std(best_maes):.4f}")
    print(f"MSE: {np.mean(best_mses):.4f} ± {np.std(best_mses):.4f}")
    print(f"R2 : {np.mean(best_r2s):.4f} ± {np.std(best_r2s):.4f}")
    print(f"Corr: {np.mean(best_corr_coefs):.4f} ± {np.std(best_corr_coefs):.4f}")

    print(f"Time: {time.time() - time_star:.2f}s")
    return np.mean(best_corr_coefs)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(data, label):
    set_seed(42)
    
    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 1)
        epsilon = trial.suggest_uniform("epsilon", 0.01, 1)
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        degree = trial.suggest_int("degree", 1, 5)

        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

        corr_score = run_nested_cv_with_early_stopping(
            data=data,
            label=label,
            outer_cv=outer_cv,
            C=C,
            epsilon=epsilon,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
        )
        return corr_score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == '__main__':
    main()