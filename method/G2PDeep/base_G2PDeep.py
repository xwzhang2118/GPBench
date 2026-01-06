import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class ModelHyperparams:
    def __init__(self,
                 left_tower_filters_list: Optional[List[int]] = None,
                 left_tower_kernel_size_list: Optional[List[int]] = None,
                 right_tower_filters_list: Optional[List[int]] = None,
                 right_tower_kernel_size_list: Optional[List[int]] = None,
                 central_tower_filters_list: Optional[List[int]] = None,
                 central_tower_kernel_size_list: Optional[List[int]] = None,
                 dnn_size_list: Optional[List[int]] = None,
                 activation: str = "linear",
                 dropout_rate: float = 0.75):
        self.left_tower_filters_list = left_tower_filters_list or [10, 10]
        self.left_tower_kernel_size_list = left_tower_kernel_size_list or [3, 15]
        self.right_tower_filters_list = right_tower_filters_list or [10]
        self.right_tower_kernel_size_list = right_tower_kernel_size_list or [3]
        self.central_tower_filters_list = central_tower_filters_list or [10]
        self.central_tower_kernel_size_list = central_tower_kernel_size_list or [3]
        self.dnn_size_list = dnn_size_list or [1]
        self.activation = activation
        self.dropout_rate = dropout_rate

def get_activation(name: str):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")
    

class G2PDeep(nn.Module):
    def __init__(self, nsnp: int, hyperparams: ModelHyperparams):
        super().__init__()
        self.nsnp = nsnp
        hp = hyperparams

        # --- Left Tower ---
        self.left_convs = nn.ModuleList()
        in_ch = 4
        for filt, k in zip(hp.left_tower_filters_list, hp.left_tower_kernel_size_list):
            self.left_convs.append(nn.Conv1d(in_ch, filt, k, padding="same"))
            in_ch = filt

        # --- Right Tower ---
        self.right_convs = nn.ModuleList()
        in_ch = 4
        for filt, k in zip(hp.right_tower_filters_list, hp.right_tower_kernel_size_list):
            self.right_convs.append(nn.Conv1d(in_ch, filt, k, padding="same"))
            in_ch = filt

        # --- Channel alignment ---
        left_out_ch = hp.left_tower_filters_list[-1]
        right_out_ch = hp.right_tower_filters_list[-1]
        self.merged_ch = max(left_out_ch, right_out_ch)

        self.left_proj = nn.Conv1d(left_out_ch, self.merged_ch, 1) \
            if left_out_ch != self.merged_ch else nn.Identity()
        self.right_proj = nn.Conv1d(right_out_ch, self.merged_ch, 1) \
            if right_out_ch != self.merged_ch else nn.Identity()

        # --- Central Tower ---
        self.central_convs = nn.ModuleList()
        in_ch = self.merged_ch
        for filt, k in zip(hp.central_tower_filters_list, hp.central_tower_kernel_size_list):
            self.central_convs.append(nn.Conv1d(in_ch, filt, k, padding="same"))
            in_ch = filt

        # --DNN ---
        self.dropout = nn.Dropout(p=hp.dropout_rate)
        final_conv_ch = hp.central_tower_filters_list[-1]
        flattened_dim = final_conv_ch * nsnp

        dnn_layers = []
        prev = flattened_dim
        for out_sz in hp.dnn_size_list[:-1]:
                dnn_layers.append(nn.Linear(prev, out_sz))
                dnn_layers.append(get_activation(hp.activation))
                dnn_layers.append(nn.Dropout(hp.dropout_rate))
                prev = out_sz
        dnn_layers.append(nn.Linear(prev, hp.dnn_size_list[-1]))
        self.dnn = nn.Sequential(*dnn_layers)

        self.activation = get_activation(hp.activation)

    def forward(self, x):
        # (B, Seq, 4) -> (B, 4, Seq)
        if x.shape[-1] != 4:
            raise ValueError(f"Expected input with 4 channels, got {x.shape}")

        x = x.transpose(1, 2)

        # Left tower
        left = x
        for conv in self.left_convs:
            left = self.activation(conv(left))

        # Right tower
        right = x
        for conv in self.right_convs:
            right = self.activation(conv(right))

        merged = self.left_proj(left) + self.right_proj(right)

        # Central tower
        for conv in self.central_convs:
            merged = self.activation(conv(merged))

        x_flat = torch.flatten(merged, 1)
        x_flat = self.dropout(x_flat)
        return self.dnn(x_flat)

    def train_model(self, train_loader, valid_loader, num_epochs, learning_rate, patience, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        self.to(device)

        best_loss = float('inf')
        best_state = None
        trigger_times = 0

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)
            valid_loss /= len(valid_loader.dataset)

            # Early stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            cur_device = next(self.parameters()).device
            best_state = {k: v.to(cur_device) for k, v in best_state.items()}
            self.load_state_dict(best_state)
        return best_loss

    def predict(self, test_loader, device):
        self.eval()
        self.to(device)
        y_pred = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.squeeze(y_pred)
        return y_pred