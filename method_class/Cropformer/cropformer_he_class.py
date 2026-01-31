import time
import torch
import numpy as np
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from lightning.pytorch import LightningModule
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(LightningModule):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=2, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, learning_rate=0.001):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, input_size)
        self.LayerNorm = nn.LayerNorm(input_size, eps=1e-12)
        self.relu = nn.ReLU()
        self.out = nn.Linear(input_size, output_dim)
        self.cnn = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        self.cnn = self.cnn.to(self.device)

        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        input_tensor = cnn_hidden
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, mixed_value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        return val_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def run_nested_cv_with_early_stopping(data, label, outer_cv, learning_rate, num_heads, dropout_prob, batch_size, hidden_dim,
                                      output_dim, kernel_size, patience, DEVICE):
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    time_star = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data)):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

        model = SelfAttention(num_heads, x_train.shape[1], hidden_dim, output_dim,
                              hidden_dropout_prob=0.5, kernel_size=kernel_size,
                              attention_probs_dropout_prob=dropout_prob).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.from_numpy(x_train).float().to(DEVICE)
        y_train_tensor = torch.from_numpy(y_train).long().to(DEVICE)
        x_test_tensor = torch.from_numpy(x_test).float().to(DEVICE)
        y_test_tensor = torch.from_numpy(y_test).long().to(DEVICE)

        train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=patience)
        best_f1 = -float('inf')

        for epoch in range(100):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = model.loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            y_test_preds, y_test_trues = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    y_pred = model(x_batch)
                    preds = torch.argmax(y_pred, dim=1)
                    y_test_preds.extend(preds.cpu().numpy())
                    y_test_trues.extend(y_batch.cpu().numpy())

            acc = accuracy_score(y_test_trues, y_test_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test_trues, y_test_preds, average="macro", zero_division=0)

            if f1 > best_f1:
                best_acc, best_prec, best_rec, best_f1 = acc, prec, rec, f1

            early_stopping(f1)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        all_acc.append(best_acc)
        all_prec.append(best_prec)
        all_rec.append(best_rec)
        all_f1.append(best_f1)
        print(f'Fold {fold+1}: ACC={best_acc:.4f}, PREC={best_prec:.4f}, REC={best_rec:.4f}, F1={best_f1:.4f}')

    print("==== Final Results ====")
    print(f"ACC : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"PREC: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
    print(f"REC : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
    print(f"F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Time: {time.time() - time_star:.2f}s")

    return all_f1

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Hyperparameter(X, label):
    set_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        heads = trial.suggest_int("heads", 1, 8)
        dropout = trial.suggest_float("dropout", 0.1, 0.9)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores = run_nested_cv_with_early_stopping(
            data=X,
            label=label.astype(int),
            outer_cv=outer_cv,
            learning_rate=lr,
            num_heads=heads,
            dropout_prob=dropout,
            batch_size=batch_size,
            hidden_dim=64,
            output_dim=len(np.unique(label)),
            kernel_size=3,
            patience=5,
            DEVICE=device
        )
        return np.mean(f1_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    return study.best_params