import os
import time
import psutil
import random
import torch
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from optuna.exceptions import TrialPruned
from base_G2PDeep_class import G2PDeep, ModelHyperparams


def train_model(model, train_loader, valid_loader, optimizer, criterion, num_epochs, patience, device):
    model.to(device)
    best_loss = float('inf')
    best_state = None
    trigger_times = 0

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
        valid_loss /= len(valid_loader.dataset)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        cur_device = next(model.parameters()).device
        best_state = {k: v.to(cur_device) for k, v in best_state.items()}
        model.load_state_dict(best_state)
    return best_loss


def predict(model, test_loader, device):
    model.eval()
    model.to(device)
    y_pred_list = []
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_pred_list.append(preds.cpu())
    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    return y_pred


def run_nested_cv_with_early_stopping(data, label, nsnp, num_classes, learning_rate, patience, batch_size, epochs=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Starting 10-fold cross-validation...")
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"Running fold {fold}...")
        process = psutil.Process(os.getpid())
        fold_start_time = time.time()

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        x_train_tensor = torch.from_numpy(X_train_sub).float()
        y_train_tensor = torch.from_numpy(y_train_sub).long()
        x_valid_tensor = torch.from_numpy(X_valid).float()
        y_valid_tensor = torch.from_numpy(y_valid).long()
        x_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(
            train_data, batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        valid_loader = DataLoader(
            valid_data, batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        test_loader = DataLoader(
            test_data, batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

        hp = ModelHyperparams()
        model = G2PDeep(nsnp=nsnp, num_classes=num_classes, hyperparams=hp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_model(model, train_loader, valid_loader, optimizer, loss_fn, epochs, patience, device)
        y_pred = predict(model, test_loader, device)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        if np.isnan(f1) or f1 <= 0:
            print(f"Fold {fold} resulted in NaN or zero F1, pruning the trial...")
            raise TrialPruned()

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        fold_time = time.time() - fold_start_time
        fold_cpu_mem = process.memory_info().rss / 1024**2
        print(f'Fold {fold}:  ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}, '
              f'Time={fold_time:.2f}s, CPU={fold_cpu_mem:.2f}MB')

    print("\n===== Cross-validation summary =====")
    print(f"Average ACC: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Average PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"Average REC: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"Average F1 : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

    return float(np.mean(all_f1)) if all_f1 else 0.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(data, label, nsnp, num_classes):
    set_seed(42)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience = trial.suggest_int("patience", 10, 100, step=10)
        try:
            f1_score = run_nested_cv_with_early_stopping(
                data=data,
                label=label,
                nsnp=nsnp,
                num_classes=num_classes,
                learning_rate=learning_rate,
                patience=patience,
                batch_size=batch_size
            )
        except TrialPruned:
            return float("-inf")
        return f1_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params


if __name__ == "__main__":
    main()
