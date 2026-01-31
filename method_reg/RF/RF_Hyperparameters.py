import os
import random
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from scipy.stats import pearsonr

# 尝试导入GPU加速版本
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    CUML_AVAILABLE = True
    print("✓ RAPIDS cuML 可用，将支持 GPU 加速")
except ImportError:
    CUML_AVAILABLE = False
    print("⚠ cuML 不可用，将使用 scikit-learn CPU 版本")

# 使用K折交叉验证并进行RandomForest训练
def run_nested_cv_with_early_stopping(data, label, outer_cv, n_estimators, max_depth, use_gpu=True):
    best_corr_coefs = []
    best_maes = []
    best_r2s = []
    best_mses = []

    # 检查GPU可用性
    gpu_available = use_gpu and CUML_AVAILABLE and torch.cuda.is_available()
    
    if gpu_available:
        print("🚀 使用 GPU 加速随机森林")
    else:
        print("⚠ 使用 CPU 版本 (scikit-learn)")

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

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train_scaled = y_train.astype(np.float32)
        y_test_scaled = y_test.astype(np.float32)
        
        # 将数据转换为 GPU 格式
        x_train_gpu = cp.asarray(x_train)
        x_test_gpu = cp.asarray(x_test)
        y_train_gpu = cp.asarray(y_train_scaled)

        model = cuRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            # min_samples_split=min_samples_split,
            # min_samples_leaf=min_samples_leaf,
            # max_features=max_features,
            random_state=42,
            n_streams=1  # 使用单个流以获得更好的性能
        )

        # 训练模型
        model.fit(x_train_gpu, y_train_gpu)

        # 预测
        y_test_preds = model.predict(x_test_gpu)
        
        # 将结果转换回 CPU
        y_test_preds = cp.asnumpy(y_test_preds)
        y_test_scaled_cpu = cp.asnumpy(cp.asarray(y_test_scaled))
        
        # # 反标准化
        # y_test_preds = scaler_y.inverse_transform(y_test_preds.reshape(-1, 1)).reshape(-1)
        # y_test_trues = scaler_y.inverse_transform(y_test_scaled_cpu.reshape(-1, 1)).reshape(-1)
        y_test_trues = y_test_scaled_cpu.reshape(-1)
        y_test_preds = y_test_preds.reshape(-1)

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
    acceleration_status = "GPU" if gpu_available else "CPU"
    print(f"加速方式: {acceleration_status}")
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

    # 目标函数，用于Optuna优化
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        # min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        # min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        # max_features = trial.suggest_float("max_features", 0.1, 1)
        
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        
        corr_score = run_nested_cv_with_early_stopping(
            data=data,
            label=label,
            outer_cv=outer_cv,
            n_estimators=n_estimators,
            max_depth=max_depth,
            # min_samples_split=min_samples_split,
            # min_samples_leaf=min_samples_leaf,
            # max_features=max_features,
            use_gpu=use_gpu
        )
        return corr_score

    # 运行Optuna超参数优化
    study = optuna.create_study(direction="maximize")
    
    # 添加GPU信息到study
    study.set_user_attr('gpu_available', torch.cuda.is_available())
    study.set_user_attr('using_gpu', use_gpu and torch.cuda.is_available())
    
    study.optimize(objective, n_trials=20)

    print("最佳参数:", study.best_params)
    print(f"优化完成 - 使用 {'GPU' if (use_gpu and torch.cuda.is_available()) else 'CPU'}")
    return study.best_params