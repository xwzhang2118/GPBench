import gc
import random
import time
import torch
import optuna
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import cupy as cp
    from cuml.svm import SVC as cuSVC
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.svm import SVC
    CUML_AVAILABLE = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_cv_eval(data, label, C, kernel, gamma, degree):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    le = LabelEncoder()
    y_all = le.fit_transform(label)

    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data, y_all)):
        print(f"===== Fold {fold+1} =====")
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        if CUML_AVAILABLE:
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
            
            model = cuSVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                probability=True
            )
            model.fit(X_train_gpu, y_train_gpu)
            y_pred = model.predict(X_test_gpu)
            y_pred = cp.asnumpy(y_pred)
        else:
            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                probability=True
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        print(f"Fold {fold+1}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}") 

        del model
        if CUML_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    return (np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s))

def main(data, label):
    set_seed(42)

    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 1)
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        degree = trial.suggest_int("degree", 1, 5)

        acc, prec, rec, f1 = run_cv_eval(data=data, label=label, C=C,
            kernel=kernel, gamma=gamma, degree=degree)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\n===== Best Parameters =====")
    print(study.best_params)
    print("Best macro-F1:", study.best_value)
    print("Successfully finished Optuna search")

    return study.best_params


if __name__ == "__main__":
    main()
