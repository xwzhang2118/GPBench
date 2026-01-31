import os
import random
import torch
import numpy as np
import argparse
import time
import optuna
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_nested_cv_with_early_stopping(
    data,
    label,
    outer_cv,
    learning_rate,
    n_estimators,
    max_depth,
    min_child_weight,
    subsample,
    colsample_bytree,
    gamma,
    reg_alpha,
    reg_lambda,
    use_gpu=True
):
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    # ===== Encode labels =====
    le = LabelEncoder()
    y_all = le.fit_transform(label)
    num_classes = len(np.unique(y_all))

    # ===== GPU / CPU =====
    gpu_available = torch.cuda.is_available() and use_gpu
    if gpu_available:
        tree_method = "hist"
        device = "cuda:0"
        print("🚀 使用 GPU 加速 XGBoost")
    else:
        tree_method = "hist"
        device = "cpu"
        print("⚠ 使用 CPU XGBoost")

    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data, y_all)):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        if num_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
            num_class_param = None
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
            num_class_param = num_classes

        model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
            num_class=num_class_param,
            early_stopping_rounds=50,
            random_state=42,
            tree_method=tree_method,
            device=device,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # ===== Prediction =====
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)

        # ===== Metrics =====
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        accel = "GPU" if gpu_available else "CPU"
        print(
            f"Fold {fold + 1}[{accel}]: "
            f"ACC={acc:.4f}, "
            f"PREC={prec:.4f}, "
            f"REC={rec:.4f}, "
            f"F1={f1:.4f}"
        )

    print("\n==== Final Results ====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    return np.mean(all_f1)

def Hyperparameter(data, label, use_gpu=True):
    set_seed(42)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }

        outer_cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=42
        )

        f1_mean = run_nested_cv_with_early_stopping(
            data=data,
            label=label,
            outer_cv=outer_cv,
            use_gpu=use_gpu,
            **params
        )
        return f1_mean

    study = optuna.create_study(direction="maximize")

    study.set_user_attr("gpu_available", torch.cuda.is_available())
    study.set_user_attr("using_gpu", use_gpu and torch.cuda.is_available())
    study.set_user_attr("xgboost_version", xgb.__version__)

    study.optimize(objective, n_trials=20)

    print("\n===== Optuna Result =====")
    print("Best F1:", study.best_value)
    print("Best params:", study.best_params)

    return study.best_params