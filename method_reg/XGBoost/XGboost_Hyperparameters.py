import os
import random
import torch
import numpy as np
import argparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import cupy as cp

def run_nested_cv_with_early_stopping(data, label, outer_cv, learning_rate, n_estimators, max_depth, min_child_weight, 
                                      subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, use_gpu=True):
    best_corr_coefs = []
    best_maes = []
    best_r2s = []
    best_mses = []

    # 检查 GPU 可用性
    gpu_available = torch.cuda.is_available() and use_gpu
    
    # 使用新的 XGBoost 2.0+ API
    if gpu_available:
        print("🚀 使用 GPU 加速 XGBoost (XGBoost 2.0+ API)")
        tree_method = 'hist'  # 使用 hist 算法
        device = 'cuda:0'     # 新的设备参数
    else:
        print("⚠ 使用 CPU 版本")
        tree_method = 'hist'
        device = 'cpu'

    import time
    time_star = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data)):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        # # 标准化数据
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)

        # # ==== y 标准化 ====
        # scaler_y = StandardScaler()
        # y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        # y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

        # x_train = cp.asarray(x_train)  # 转换为cupy数组（GPU内存）
        # y_train_scaled = cp.asarray(y_train_scaled)
        # x_test = cp.asarray(x_test)  # 转换为cupy数组（GPU内存）
        # y_test_scaled = cp.asarray(y_test_scaled)

        # 初始化XGBoost模型 - 使用新的GPU参数设置 (XGBoost 2.0+)
        model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            eval_metric='rmse',
            random_state=42,
            # ==== 新的 GPU 加速参数 (XGBoost 2.0+) ====
            tree_method=tree_method,      # 使用 hist 算法
            device=device,                # 新的设备参数：'cuda' 或 'cpu'
            n_jobs=-1,                   # 使用所有CPU核心
        )

        # 训练模型
        model.fit(x_train, 
                  y_train,
                eval_set=[(x_test, y_test)],
                verbose=False)

        # 预测
        y_test_preds = model.predict(x_test)
        # y_test_preds = scaler_y.inverse_transform(y_test_preds.reshape(-1, 1)).reshape(-1)
        # y_test_trues = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)
        y_test_preds = y_test_preds.reshape(-1)
        y_test_trues = y_test.reshape(-1)
        # 计算评价指标
        corr_coef = np.corrcoef(y_test_preds, y_test_trues)[0, 1]
        mae = mean_absolute_error(y_test_trues, y_test_preds)
        mse = mean_squared_error(y_test_trues, y_test_preds)
        r2 = r2_score(y_test_trues, y_test_preds)

        best_corr_coefs.append(corr_coef)
        best_maes.append(mae)
        best_r2s.append(r2)
        best_mses.append(mse)

        acceleration_status = "GPU" if gpu_available else "CPU"
        print(f'Fold {fold + 1}[{acceleration_status}]: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Corr={corr_coef:.4f}')

    print("==== Final Results ====")
    print(f"加速方式: {'GPU' if gpu_available else 'CPU'}")
    print(f"MAE: {np.mean(best_maes):.4f} ± {np.std(best_maes):.4f}")
    print(f"MSE: {np.mean(best_mses):.4f} ± {np.std(best_mses):.4f}")
    print(f"R2 : {np.mean(best_r2s):.4f} ± {np.std(best_r2s):.4f}")
    print(f"Corr: {np.mean(best_corr_coefs):.4f} ± {np.std(best_corr_coefs):.4f}")

    print(f"Time: {time.time() - time_star:.2f}s")
    return np.mean(best_corr_coefs)

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Hyperparameter(data, label, use_gpu=True):
    set_seed(42)
    
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.2)
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.05, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        gamma = trial.suggest_float("gamma", 0, 10)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10, log=True)
        reg_lambda = trial.suggest_float("reg_lambda",1e-3, 10, log=True)

        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

        corr_score = run_nested_cv_with_early_stopping(
            data=data,
            label=label,
            outer_cv=outer_cv,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            use_gpu=use_gpu
        )
        return corr_score

    # 运行Optuna超参数优化
    study = optuna.create_study(direction="maximize")
    
    # 添加GPU信息到study
    study.set_user_attr('gpu_available', torch.cuda.is_available())
    study.set_user_attr('using_gpu', use_gpu and torch.cuda.is_available())
    study.set_user_attr('xgboost_version', xgb.__version__)
    
    study.optimize(objective, n_trials=20)

    print("最佳参数:", study.best_params)
    print(f"优化完成 - 使用 {'GPU' if (use_gpu and torch.cuda.is_available()) else 'CPU'}")
    return study.best_params