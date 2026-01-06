import os
import time
import psutil
import argparse
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data  import DataLoader, TensorDataset, random_split
from WheatGP_base import wheatGP_base
from torch.optim.lr_scheduler import StepLR
import WheatGP_Hyperparameters

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='WheatGP/', help='Random seed')
    parser.add_argument('--species', type=str, default='Chicken/')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--step_size', type=int, default=90, help='Step size for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    args = parser.parse_args()
    return args

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

def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    return xData, yData, nsample, nsnp, names

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, scheduler, device):
    trigger_times = 0  
    best_loss = float('inf')
    best_P = float('-inf')
    train_losses = []
    val_losses = []
    best_state = None

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


def run_nested_cv(args, data, label, nsnp, device):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
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

        train_Y = torch.tensor(y_train, dtype=torch.float32).to(device)
        test_Y = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(*train_G, train_Y)
        test_dataset = TensorDataset(*test_G, test_Y)
        
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = wheatGP_base(nsnp).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=90, gamma=0.1)
        train_model(model,train_loader, val_loader, criterion, optimizer, args.epochs, args.patience, scheduler, device)
        y_pred, y_test = predict(model, test_loader, device)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results_df = pd.DataFrame({'Y_test': y_test, 'Y_pred': y_pred})
        results_df.to_csv(os.path.join(result_dir, f"fold{fold}.csv"), index=False)

    print("\n===== Cross-validation summary =====")
    print(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
    print(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print(f"Average R2 : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")


if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()  
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_species =['Cattle/', 'Chicken/', 'Chickpea/', 'Cotton/', 'Loblolly_Pine/',
                   'Maize/', 'Millet/', 'Mouse/', 'Pig/', 'Rapeseed/', 
                   'Rice/', 'Soybean/', 'Wheat/','Yeast/']
    
    for i in range(len(all_species)):
        args.species = all_species[i]
        args.device = device
        X, Y, nsamples, nsnp, names = load_data(args)
        for j in range(len(names)):
            args.phe = names[j]
            print("starting run " + args.methods + args.species + args.phe)
            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            best_params = WheatGP_Hyperparameters.main(X, label, nsnp)
            args.learning_rate = best_params['learning_rate']
            args.weight_decay = best_params['weight_decay']
            args.patience = best_params['patience']
            args.batch_size = best_params['batch_size']
            start_time = time.time() 
            torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())

            run_nested_cv(args, data=X, label=label, nsnp = nsnp, device = args.device)

            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully")