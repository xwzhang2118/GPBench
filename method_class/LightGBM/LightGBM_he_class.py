import random
import torch
import numpy as np
import lightgbm as lgb
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_cv_eval(data, label, params):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    le = LabelEncoder()
    y_all = le.fit_transform(label)
    n_classes = len(np.unique(y_all))

    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data, y_all)):
        print(f"===== Fold {fold+1} =====")

        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_test, label=y_test)


        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=100,
        )

        y_prob = model.predict(X_test)

        # ===== binary / multiclass safe =====
        if n_classes == 2:
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        print(
            f"Fold {fold+1}: "
            f"ACC={acc:.4f}, "
            f"PREC={prec:.4f}, "
            f"REC={rec:.4f}, "
            f"F1={f1:.4f}"
        )

    return (
        np.mean(accs),
        np.mean(precs),
        np.mean(recs),
        np.mean(f1s)
    )

def main(X, label):
    set_seed(42)
    torch.cuda.empty_cache()

    n_classes = len(np.unique(label))

    def objective(trial):
        params = {
            'objective': 'binary' if n_classes == 2 else 'multiclass',
            'metric': 'multi_logloss' if n_classes > 2 else 'binary_logloss',
            'num_class': n_classes if n_classes > 2 else None,

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

            'num_boost_round': trial.suggest_int('num_boost_round', 100, 1000),

            'device_type': 'gpu',
            'gpu_device_id': 1,
            'num_threads': 8,
            'verbosity': -1
        }

        acc, prec, rec, f1 = run_cv_eval(X, label, params)

        # ===== optimize macro-F1 =====
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best macro-F1:", study.best_value)

    return study.best_params


if __name__ == "__main__":
    main()
