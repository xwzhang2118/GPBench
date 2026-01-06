import os
import time
import psutil
import optuna
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data  import DataLoader, TensorDataset, random_split
from WheatGP_base import wheatGP_base
from torch.optim.lr_scheduler import StepLR
from optuna.exceptions import TrialPruned 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_genotype_data(genotype1d_array, features_per_group, n_groups=5, pad_value=0):
    N_samples, N_features = genotype1d_array.shape
    groups = []
    for i in range(n_groups):
        start = i * features_per_group
        end = start + features_per_group
        group = genotype1d_array[:, start:end]
        groups.append(group)
    groups = np.array(groups)
    tensors = [torch.tensor(groups[i, :], dtype=torch.float32) for i in range(5)]
    return tensors


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, scheduler, device):
    best_P = - float('inf')
    best_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []
    trigger_times = 0  
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = [b.to(device) for b in batch[:-1]]
            Y_train_batch = batch[-1].to(device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs.view(-1), Y_train_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            Y_preds = []
            Y_trues = []
            val_running_loss = 0.0
            for batch in val_loader:
                inputs = [b.to(device) for b in batch[:-1]]
                Y_val_batch = batch[-1].to(device)
                outputs = model(*inputs)
                Y_preds.append(outputs)
                Y_trues.append(Y_val_batch)
                loss = criterion(outputs.view(-1), Y_val_batch)
                val_running_loss += loss.item()
            epoch_val_loss = val_running_loss / len(val_loader)
            val_losses.append(epoch_val_loss)

            Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
            Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()
            pearson_r = np.corrcoef(Y_trues, Y_preds)[0, 1]

         # ---------- Early Stopping ----------
        if pearson_r > best_P:
            best_loss = epoch_val_loss
            best_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        
    return Y_trues, Y_preds

def predict(model, test_loader, device):
        with torch.no_grad():
            Y_preds = []
            Y_trues = []
            for batch in test_loader:
                inputs = [b.to(device) for b in batch[:-1]]
                Y_val_batch = batch[-1].to(device)
                outputs = model(*inputs)
                Y_preds.append(outputs)
                Y_trues.append(Y_val_batch)
            Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
            Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()
        return Y_preds, Y_trues


def run_nested_cv_with_early_stopping(data, label, nsnp, learning_rate, weight_decay, patience, batch_size, epochs=1000):
    print("Starting 10-fold cross-validation...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        features_per_group = nsnp // 5
        G1train, G2train, G3train, G4train, G5train = split_genotype_data(X_train, features_per_group)
        G1test, G2test, G3test, G4test, G5test = split_genotype_data(X_test, features_per_group)
        train_G = [G1train, G2train, G3train, G4train, G5train]
        test_G = [G1test, G2test, G3test, G4test, G5test]

        train_Y = torch.tensor(y_train, dtype=torch.float32)
        test_Y = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(*train_G, train_Y)
        test_dataset = TensorDataset(*test_G, test_Y)

        val_size = max(1, int(0.1 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = wheatGP_base(nsnp).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=90, gamma=0.1)
        train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience,scheduler, device)
        y_pred, y_test = predict(model,test_loader, device)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        
        if np.isnan(pcc):
            print(f"Fold {fold} resulted in NaN PCC, pruning the trial...")
            raise TrialPruned()

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')

    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")

    return np.mean(all_pcc) if all_pcc else 0.0



def main(data, label, nsnp):
    set_seed(42)
    
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.01)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
        patience = trial.suggest_int("patience", 1, 10)
        try:
            corr_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                batch_size=batch_size
            )
        except TrialPruned:
            return float("-inf")
        return corr_score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == "__main__":
    main()