import os
import time
import psutil
import argparse
import random
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import DL_GWAS_Hyperparameters
import keras
import pynvml
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.layers import *
from scipy.stats import pearsonr
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('--methods', type=str, default='DL_GWAS/', help='Random seed')
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    args = parser.parse_args()
    return args

def indices_to_one_hot(data,nb_classes):
	targets = np.array(data).reshape(-1)
	return np.eye(nb_classes)[targets]

def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, 'genetype.npz'))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, 'phenotype.npz'))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    xData = xData.astype(int)
    arr = np.empty(shape=(nsample, nsnp, 4))
    xData[xData == -9] = 0
    for i in range(0, nsample):
        arr[i] = indices_to_one_hot(pd.to_numeric(xData[i], downcast='signed'), 4)
    
    return arr, yData, nsample, nsnp, names

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_gpu_mem_by_pid(pid):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        if p.pid == pid:
            return p.usedGpuMemory / 1024**2
    return 0.0

def resnet(args, nsnp):
	
	inputs = Input(shape=(nsnp,4))
	
	x = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(inputs)
	x = Conv1D(10,20,padding='same',activation = 'linear', kernel_initializer = 'TruncatedNormal',kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(x)	
	x = Dropout(0.75)(x)
	shortcut = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(inputs)
	x = layers.add([shortcut,x])
	x = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(x)
	x = Dropout(0.75)(x)
	x = Flatten()(x)
	x = Dropout(0.75)(x)
	outputs = Dense(1,activation = 'linear',bias_regularizer = regularizers.l2(0.01),kernel_initializer = 'TruncatedNormal',name = 'out')(x)
	model = Model(inputs = inputs,outputs = outputs)
	model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),metrics=['mae'])
	return model

def isru(x):
    return x / (tf.sqrt(1 + 0.02 * tf.square(x)))
	
def run_nested_cv(args, data, label, nsnp):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience)
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        model = resnet(args, nsnp = nsnp)
        model.fit(X_train_sub, y_train_sub, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, y_valid),callbacks=[early_stopping],shuffle= True, verbose=0)
        y_pred = model.predict(X_test)

        y_pred = np.asarray(y_pred).flatten().astype(np.float64)
        y_test = np.asarray(y_test).flatten().astype(np.float64)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pcc, _ = pearsonr(y_test, y_pred)

        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = get_gpu_mem_by_pid(os.getpid())
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  Corr={pcc:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Time={fold_time:.2f}s, '
              f'GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB')
     
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
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
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
            best_params = DL_GWAS_Hyperparameters.main(X, label, nsnp)
            args.learning_rate = best_params['learning_rate']
            args.patience = best_params['patience']
            args.batch_size = best_params['batch_size']
            start_time = time.time()

            process = psutil.Process(os.getpid())
            run_nested_cv(args, data=X, label=label, nsnp = nsnp)

            elapsed_time = time.time() - start_time
            print(f"running time: {elapsed_time:.2f} s")
            print("successfully")