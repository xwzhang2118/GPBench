import time
import torch
import numpy as np
import torch.nn as nn
import random
import torch.optim  as optim 
from torch.utils.data  import DataLoader, TensorDataset 
from sklearn.preprocessing  import StandardScaler 
from lightning.pytorch import LightningModule
import optuna

from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.model_selection import KFold
from sklearn.metrics  import mean_absolute_error, mean_squared_error, r2_score

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
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, learning_rate=0.001):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = torch.nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

        self.learning_rate = learning_rate
        self.loss_fn = MSELoss()

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        self.cnn = self.cnn.to(self.device)

        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        input_tensor = cnn_hidden
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = mixed_query_layer
        key_layer = mixed_key_layer
        value_layer = mixed_value_layer

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

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
    best_corr_coefs = []
    best_maes = []
    best_r2s = []
    best_mses = []

    time_star = time.time()
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data)):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        
        num_attention_heads = num_heads
        attention_probs_dropout_prob = dropout_prob
        hidden_dropout_prob = 0.5

        model = SelfAttention(num_attention_heads, x_train.shape[1], hidden_dim, output_dim,
                                hidden_dropout_prob=hidden_dropout_prob, kernel_size=kernel_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.from_numpy(x_train).float().to(DEVICE)
        y_train_tensor = torch.from_numpy(y_train).float().to(DEVICE)
        x_test_tensor = torch.from_numpy(x_test).float().to(DEVICE)
        y_test_tensor = torch.from_numpy(y_test).float().to(DEVICE)

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=patience)
        best_corr_coef = -float('inf')
        best_mae = float('inf')
        best_mse = float('inf')
        best_r2 = -float('inf')
        for epoch in range(100):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            model.eval()
            y_test_preds, y_test_trues = [], []
            
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    y_test_pred = model(x_batch)
                    y_test_preds.extend(y_test_pred.cpu().numpy().reshape(-1).tolist())
                    y_test_trues.extend(y_batch.cpu().numpy().reshape(-1).tolist())

            corr_coef = np.corrcoef(y_test_preds, y_test_trues)[0, 1]
            mae = mean_absolute_error(np.array(y_test_trues), np.array(y_test_preds))
            mse = mean_squared_error(np.array(y_test_trues), np.array(y_test_preds))
            r2 = r2_score(np.array(y_test_trues), np.array(y_test_preds))
            scheduler.step(corr_coef)

            if corr_coef > best_corr_coef:
                best_mae = mae
                best_corr_coef = corr_coef
                best_mse = mse
                best_r2 = r2

            early_stopping(corr_coef)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        best_corr_coefs.append(best_corr_coef)
        best_maes.append(best_mae)
        best_mses.append(best_mse)
        best_r2s.append(best_r2)
        print(f'Fold {fold + 1}: MAE={best_mae:.4f}, MSE={best_mse:.4f}, R2={best_r2:.4f}, Corr={best_corr_coef:.4f}')
    
    print("==== Final Results ====")
    print(f"MAE: {np.mean(best_maes):.4f} ± {np.std(best_maes):.4f}")
    print(f"MSE: {np.mean(best_mses):.4f} ± {np.std(best_mses):.4f}")
    print(f"R2 : {np.mean(best_r2s):.4f} ± {np.std(best_r2s):.4f}")
    print(f"Corr: {np.mean(best_corr_coefs):.4f} ± {np.std(best_corr_coefs):.4f}")

    print(f"Time: {time.time() - time_star:.2f}s")
    return best_corr_coefs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(X, label):
    set_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    
    def objective(trial):
        lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
        heads = trial.suggest_int("heads", 1, 8, step=1)
        dropout = trial.suggest_float("dropout", 0.1, 0.9, step=0.1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42) 

        corr_scores = run_nested_cv_with_early_stopping(
            data=X,
            label=label,
            outer_cv=outer_cv,
            learning_rate=lr,
            num_heads=heads,
            dropout_prob=dropout,
            batch_size=batch_size,
            hidden_dim=64,
            output_dim=1,
            kernel_size=3,
            patience=5,
            DEVICE=device,
        )
        return np.mean(corr_scores) 

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) 

    print("best params:", study.best_params)
    print("successfully")
    return study.best_params

if __name__ == '__main__':
    main()
