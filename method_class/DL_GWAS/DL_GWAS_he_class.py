import os
import time
import psutil
import random
import optuna
import gc
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from optuna.exceptions import TrialPruned

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resnet(nsnp: int, num_classes: int, learning_rate: float):
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
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def run_nested_cv_with_early_stopping(
    data,
    label,
    nsnp: int,
    num_classes: int,
    learning_rate: float,
    batch_size: int,
    patience: int,
):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    all_acc = []
    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=42,
        )

        model = resnet(nsnp=nsnp, num_classes=num_classes, learning_rate=learning_rate)
        model.fit(
            X_train_sub,
            y_train_sub,
            batch_size=batch_size,
            epochs=1000,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping],
            shuffle=True,
            verbose=0,
        )

        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        if np.isnan(acc) or acc <= 0:
            try:
                model.stop_training = True
            except Exception:
                pass
            del model
            keras.backend.clear_session()
            gc.collect()
            raise TrialPruned()

        all_acc.append(acc)

        _ = process.memory_info().rss / 1024**2
        _ = time.time() - fold_start_time
        del model
        keras.backend.clear_session()
        gc.collect()

    return float(np.mean(all_acc)) if all_acc else 0.0


def Hyperparameter(data, label, nsnp: int, num_classes: int):
    set_seed(42)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience = trial.suggest_int("patience", 5, 15)

        try:
            acc_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                num_classes=num_classes,
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=patience,
            )
        except TrialPruned:
            return float("-inf")
        finally:
            keras.backend.clear_session()
            gc.collect()

        return acc_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    return study.best_params