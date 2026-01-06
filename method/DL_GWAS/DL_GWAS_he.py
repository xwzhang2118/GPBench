import os
import time
import psutil
import random
import optuna
import torch
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.layers import *
from scipy.stats import pearsonr
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from optuna.exceptions import TrialPruned 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def resnet(nsnp, learning_rate):
    inputs = Input(shape=(nsnp, 4))
    x = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(inputs)
    x = Conv1D(10,20,padding='same',activation = 'linear', kernel_initializer = 'TruncatedNormal',kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.75)(x)
    shortcut = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(inputs)
    x = layers.add([shortcut,x])
    x = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(x)
    x = Dropout(0.75)(x)
    x = Flatten()(x)
    x = Dropout(0.75)(x)
    outputs = Dense(1,activation = 'linear', bias_regularizer = regularizers.l2(0.01),kernel_initializer = 'TruncatedNormal',name = 'out')(x)

    model = Model(inputs = inputs,outputs = outputs)
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['mae'])
    
    return model

def isru(x):
    return x / (tf.sqrt(1 + 0.02 * tf.square(x)))
	
def run_nested_cv_with_early_stopping(data, label, nsnp, learning_rate, batch_size, patience):
    print("Starting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    all_mse, all_mae, all_r2, all_pcc = [], [], [], []
    time_star = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        model = resnet(nsnp = nsnp, learning_rate = learning_rate)
        model.fit(X_train_sub, y_train_sub, batch_size=batch_size, epochs=1000, validation_data=(X_valid, y_valid),callbacks=[early_stopping],shuffle= True, verbose=0)
        y_pred = model.predict(X_test)
        
        y_pred = np.asarray(y_pred).flatten().astype(np.float64)
        y_test = np.asarray(y_test).flatten().astype(np.float64)

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
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience = trial.suggest_int("patience", 5, 10)
        try:
            corr_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=patience
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
