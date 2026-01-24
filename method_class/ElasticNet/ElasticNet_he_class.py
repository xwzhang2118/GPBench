import gc
import random
import time
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from optuna.exceptions import TrialPruned

def run_nested_cv_with_early_stopping(data, label, outer_cv, C, l1_ratio):
    best_accs, best_precs, best_recs, best_f1s = [], [], [], []
    time_star = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data, label)):
        x_train = data[train_idx]
        x_test = data[test_idx]
        y_train = label[train_idx]
        y_test = label[test_idx]

        model = LogisticRegression(
            penalty='elasticnet',
            C=C,
            l1_ratio=l1_ratio,
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

        best_accs.append(acc)
        best_precs.append(prec)
        best_recs.append(rec)
        best_f1s.append(f1)

        print(f'Fold {fold + 1}: ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}')
        del model, y_test_preds, x_train, x_test, y_train, y_test

    print("==== Final Results ====")
    print(f"ACC: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"PREC: {np.mean(best_precs):.4f} ± {np.std(best_precs):.4f}")
    print(f"REC: {np.mean(best_recs):.4f} ± {np.std(best_recs):.4f}")
    print(f"F1: {np.mean(best_f1s):.4f} ± {np.std(best_f1s):.4f}")

    print(f"Time: {time.time() - time_star:.2f}s")
    gc.collect()

    mean_f1 = float(np.mean(best_f1s)) if best_f1s else 0.0
    if np.isnan(mean_f1) or mean_f1 <= 0:
        raise TrialPruned()

    return mean_f1

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def main(data, label):
    set_seed(42)
    
    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        l1_ratio = trial.suggest_categorical("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
        
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        try:
            f1_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                outer_cv=outer_cv,
                C=C,
                l1_ratio=l1_ratio
            )
        except TrialPruned:
            return float("-inf")
        return f1_score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == '__main__':
    main()
