import os
import time
import psutil
import swanlab
import argparse
import random
import gc
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import pynvml
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import DL_GWAS_he_class

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="DL_GWAS classification")
    parser.add_argument("--methods", type=str, default="DL_GWAS/")
    parser.add_argument('--species', type=str, default='')
    parser.add_argument('--phe', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--result_dir', type=str, default='result/')
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def load_data(args):
    xData = np.load(os.path.join(args.data_dir, args.species, "genotype.npz"))["arr_0"]
    yData = np.load(os.path.join(args.data_dir, args.species, "phenotype.npz"))["arr_0"]
    names = np.load(os.path.join(args.data_dir, args.species, "phenotype.npz"))["arr_1"]

    nsample = xData.shape[0]
    nsnp = xData.shape[1]
    print("Number of samples: ", nsample)
    print("Number of SNPs: ", nsnp)
    xData = xData.astype(int)
    arr = np.empty(shape=(nsample, nsnp, 4), dtype=np.float32)
    xData[xData == -9] = 0
    for i in range(0, nsample):
        arr[i] = indices_to_one_hot(pd.to_numeric(xData[i], downcast="signed"), 4).astype(np.float32, copy=False)

    return arr, yData, nsample, nsnp, names


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_mem_by_pid(pid, handle=None):
    if handle is None:
        return 0.0
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in procs:
            if p.pid == pid:
                return p.usedGpuMemory / 1024**2
        return 0.0
    except Exception:
        return 0.0


def resnet(args, nsnp: int, num_classes: int):
    inputs = Input(shape=(nsnp, 4))

    x = Conv1D(
        10, 4, padding="same", activation="linear",
        kernel_initializer="TruncatedNormal",
        kernel_regularizer=regularizers.l2(0.1),
        bias_regularizer=regularizers.l2(0.01),
    )(inputs)
    x = Conv1D(
        10, 20, padding="same", activation="linear",
        kernel_initializer="TruncatedNormal",
        kernel_regularizer=regularizers.l2(0.1),
        bias_regularizer=regularizers.l2(0.01),
    )(x)
    x = Dropout(0.75)(x)

    shortcut = Conv1D(
        10, 4, padding="same", activation="linear",
        kernel_initializer="TruncatedNormal",
        kernel_regularizer=regularizers.l2(0.1),
        bias_regularizer=regularizers.l2(0.01),
    )(inputs)
    x = layers.add([shortcut, x])

    x = Conv1D(
        10, 4, padding="same", activation="linear",
        kernel_initializer="TruncatedNormal",
        kernel_regularizer=regularizers.l2(0.1),
        bias_regularizer=regularizers.l2(0.01),
    )(x)
    x = Dropout(0.75)(x)
    x = Flatten()(x)
    x = Dropout(0.75)(x)

    outputs = Dense(num_classes, activation="softmax", name="out")(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=["accuracy"],
    )
    return model


def run_nested_cv(args, data, label, nsnp: int, num_classes: int):
    result_dir = os.path.join(args.result_dir, args.methods + args.species + args.phe)
    os.makedirs(result_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    cv_start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        fold_start_time = time.time()
        process = psutil.Process(os.getpid())
        print(f"\n===== Fold {fold} =====")

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=42,
        )

        model = resnet(args, nsnp=nsnp, num_classes=num_classes)
        model.fit(
            X_train_sub,
            y_train_sub,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping],
            shuffle=True,
            verbose=0,
        )

        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_gpu_mem = get_gpu_mem_by_pid(os.getpid(), handle)
        fold_cpu_mem = process.memory_info().rss / 1024**2

        print(
            f"Fold {fold}: "
            f"ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, "
            f"Time={fold_time:.2f}s, GPU={fold_gpu_mem:.2f}MB, CPU={fold_cpu_mem:.2f}MB"
        )

        pd.DataFrame({"Y_test": y_test, "Y_pred": y_pred}).to_csv(
            os.path.join(result_dir, f"fold{fold}.csv"), index=False
        )

        del model
        keras.backend.clear_session()
        gc.collect()

    cv_time = time.time() - cv_start_time
    print("\n===== Cross-validation summary =====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {cv_time:.2f}s")

if __name__ == "__main__":
    set_seed(42)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    args = parse_args()

    all_species =  ["Human/Sim/"]
    for species in all_species:
        args.species = species
        X, Y, nsamples, nsnp, names = load_data(args)
        print("Starting:", args.methods + args.species)

        label_raw = np.nan_to_num(Y[:, 0])
        le = LabelEncoder()
        label = le.fit_transform(label_raw)
        num_classes = len(le.classes_)

        best_params = DL_GWAS_he_class.Hyperparameter(X, label, nsnp, num_classes)
        args.learning_rate = best_params["learning_rate"]
        args.batch_size = best_params["batch_size"]
        args.patience = best_params["patience"]

        start_time = time.time()
        run_nested_cv(args, data=X, label=label, nsnp=nsnp, num_classes=num_classes)
        elapsed_time = time.time() - start_time

        print(f"Total running time: {elapsed_time:.2f}s")
        print("Successfully finished:", args.species)

