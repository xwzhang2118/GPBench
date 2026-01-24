import random
import torch
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_nested_cv(data, label, params):
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_corr = []

    for train_idx, test_idx in kf.split(data):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        test_set = lgb.Dataset(X_test, label=y_test)

        model = lgb.train(
            params,
            train_set,
            valid_sets=[test_set],
            num_boost_round=100,
        )

        y_pred = model.predict(X_test)
        corr, _ = pearsonr(y_test, y_pred)
        all_corr.append(corr)
    return np.mean(all_corr)


def main(X, label):
    set_seed(42)
    torch.cuda.empty_cache()

    def objective(trial):
        params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'num_boost_round':trial.suggest_int('num_boost_round', 100, 1000),
        'device_type': 'gpu',
        'gpu_device_id': 1,
        'num_threads': 8,
        'verbose':-1, 
        }

        corr_scores = run_nested_cv(data=X, label=label, params=params)
        return np.mean(corr_scores) 

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) 

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params


if __name__ == "__main__":
    main()

