import time
import argparse
import pynvml
import psutil
import os
import torch
import random
import swanlab
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import KFold, train_test_split
import  sys
sys.path.append("..")
from utils.models_locally_connected import LCLModel
from utils.common import DataDimensions
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from EIR_Hyperparameters import main
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='EIR/', help='Random seed')
    parser.add_argument('--species', type=str, default='', help='Species name')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    args = parser.parse_args()
    return args


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    return xData, yData, nsample, nsnp, names

def get_gpu_mem_by_pid(pid):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def init():
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def one_hot_encode_ATCG(char):
    c = char
    if char==0:
        return [1,0,0,0]
    elif char==1:
        return [0,1,0,0]
    elif char==2:
        return [0,0,1,0]
    else:
        return [0,0,0,1]

def one_hot_seq(df:pd.DataFrame, nsnp:int):
    one_hot_df = df.applymap(lambda x:one_hot_encode_ATCG(x))
    one_hot_df = one_hot_df.values.tolist()
    one_hot_df = np.array(one_hot_df)
    one_hot_df = np.reshape(one_hot_df, (one_hot_df.shape[0],-1))
    tensor_data = torch.Tensor(one_hot_df)
    return tensor_data


def train_model(model, train_loader, valid_loader, optimizer, criterion, num_epochs, patience, device):

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

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss
    
def predict(model, test_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  
            y_pred.append(outputs.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred = np.squeeze(y_pred)
    return y_pred


def run_nested_cv(args, data, label, nsnp, device):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    data = data.reshape(data.shape[0], 1, nsnp, -1)

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
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

        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=False)

        model = LCLModel(DataDimensions(channels=1, height=nsnp, width=1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()

        train_model(model, train_loader, valid_loader, optimizer, loss_fn, args.epochs, args.patience, device)
        y_pred = predict(model, test_loader, device)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem =  get_gpu_mem_by_pid(os.getpid())
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


if __name__ == '__main__':
    start = time.time()
    set_seed(42)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    args = parse_args()
    all_species =['Cattle/', 'Chicken/', 'Chickpea/', 'Cotton/', 'Loblolly_Pine/',
                   'Maize/', 'Millet/', 'Mouse/', 'Pig/', 'Rapeseed/', 
                   'Rice/', 'Soybean/', 'Wheat/','Yeast/']
    
    for i in range(len(all_species)):
        args.species = all_species[i]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device  
        X, Y, nsamples, nsnp, names = load_data(args)
        for j in range(len(names)):
            args.phe = names[j]
            print("starting run " + args.methods + args.species + args.phe)
            label = Y[:, j]
            label = np.nan_to_num(label, nan=np.nanmean(label))
            best_params = main(X, label, nsnp)
            args.learning_rate = best_params['learning_rate']
            args.batch_size = best_params['batch_size']
            args.patience = best_params['patience']
            start_time = time.time() 
            torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())

            run_nested_cv(args, data=X, label=label, nsnp = nsnp, device = args.device)

            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully")


