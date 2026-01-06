import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing  import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data  import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned 
from utils.models_locally_connected import LCLModel
from utils.common import DataDimensions
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, valid_loader, optimizer,criterion, num_epochs, patience, device):
    model.to(device)
    best_loss = float('inf')
    best_state = None
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        # ---------- 验证 ----------
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        # ---------- Early stopping ----------
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 恢复最佳参数
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss
    
def predict(model, test_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)  
            y_pred.append(outputs.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred = np.squeeze(y_pred)
    return y_pred

def run_nested_cv_with_early_stopping(data, label, nsnp, learning_rate, patience, batch_size,epochs=1000):
    device = torch.device("cuda:0")
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        
        x_train_tensor = torch.from_numpy(X_train_sub).float().to(device)
        y_train_tensor = torch.from_numpy(y_train_sub).float().to(device)
        x_valid_tensor = torch.from_numpy(X_valid).float().to(device)
        y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
        x_test_tensor = torch.from_numpy(X_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).float().to(device)

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size, shuffle=False)

        # 初始化模型
        model = LCLModel(DataDimensions(channels=1, height=nsnp, width=1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()

        train_model(model, train_loader, valid_loader, optimizer,loss_fn, epochs, patience, device)
        y_pred = predict(model, test_loader, device)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        if np.isnan(pcc):
            print(f"Fold {fold} resulted in NaN PCC, pruning the trial...")
            raise TrialPruned()

        # 将结果添加到列表中
        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        # ====== 每折结束时统计 ======
        fold_time = time.time() - fold_start_time
        #fold_gpu_mem = #torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
                f'CPU={fold_cpu_mem:.2f}MB')
        
    return np.mean(all_pcc) if all_pcc else 0.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(data, label, nsnp):
    set_seed(42)
    
    def objective(trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4,0.1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience = trial.suggest_int("patience", 10, 100, step=10)
        try:
            corr_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=learning_rate,
                patience=patience,
                batch_size=batch_size
            )
        
        except TrialPruned:
            return float("-inf")
        return corr_score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("最佳参数:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == "__main__":
    main()
