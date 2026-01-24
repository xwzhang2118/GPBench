import os
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    CUML_AVAILABLE = False

def run_nested_cv_with_early_stopping(
        data, label, outer_cv,
        n_estimators, max_depth, use_gpu=True):

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    le = LabelEncoder()
    y_all = le.fit_transform(label)
    num_classes = len(np.unique(y_all))
    print(f"Classes: {le.classes_} (n={num_classes})")

    gpu_available = use_gpu and CUML_AVAILABLE and torch.cuda.is_available()

    import time
    time_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data, y_all)):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if gpu_available:
            X_train = cp.asarray(X_train)
            X_test = cp.asarray(X_test)
            y_train = cp.asarray(y_train)

            model = cuRandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_streams=1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if gpu_available:
            y_pred = cp.asnumpy(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        device = "GPU" if gpu_available else "CPU"
        print(
            f"Fold {fold + 1}[{device}]: "
            f"ACC={acc:.4f}, PREC={prec:.4f}, "
            f"REC={rec:.4f}, F1={f1:.4f}"
        )

        if gpu_available:
            cp.get_default_memory_pool().free_all_blocks()

    print("\n==== Final Results ====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {time.time() - time_start:.2f}s")
    return np.mean(all_f1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(data, label, use_gpu=True):
    set_seed(42)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)

        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

        score = run_nested_cv_with_early_stopping(
            data=data,
            label=label,
            outer_cv=outer_cv,
            n_estimators=n_estimators,
            max_depth=max_depth,
            use_gpu=use_gpu
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    return study.best_params

if __name__ == '__main__':
    main()
