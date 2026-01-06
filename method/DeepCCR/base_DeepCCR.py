import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepCCR(nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_seq_len=162,
                 lstm_hidden_size=64,
                 fc1_hidden_dim=128):
        super(DeepCCR, self).__init__()

        # ==================== Conv1 ====================
        self.conv1_kernel = 500
        self.conv1_stride = 100
        self.conv1_out_ch = 150

        if input_seq_len < self.conv1_kernel:
            pad_left = (self.conv1_kernel - input_seq_len) // 2
            pad_right = self.conv1_kernel - input_seq_len - pad_left
            self.conv1_padding = (pad_left, pad_right)
            conv1_input_len = self.conv1_kernel
        else:
            self.conv1_padding = (0, 0)
            conv1_input_len = input_seq_len

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=self.conv1_out_ch,
            kernel_size=self.conv1_kernel,
            stride=self.conv1_stride,
            padding=0
        )
        self.relu1 = nn.ReLU()

        conv1_seq_len = (conv1_input_len - self.conv1_kernel) // self.conv1_stride + 1

        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        pool1_seq_len = (conv1_seq_len + 1) // 2

        # ==================== BiLSTM ====================
        self.bilstm = nn.LSTM(input_size=150, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)

        # ==================== Conv2  ====================
        self.conv2_kernel = 30
        self.conv2_stride = 5
        self.conv2_out_ch = 150

        if pool1_seq_len < self.conv2_kernel:
            pad_left2 = (self.conv2_kernel - pool1_seq_len) // 2
            pad_right2 = self.conv2_kernel - pool1_seq_len - pad_left2
            self.conv2_padding = (pad_left2, pad_right2)
            conv2_input_len = self.conv2_kernel
        else:
            self.conv2_padding = (0, 0)
            conv2_input_len = pool1_seq_len

        self.conv2 = nn.Conv1d(
            in_channels=lstm_hidden_size*2,
            out_channels=self.conv2_out_ch,
            kernel_size=self.conv2_kernel,
            stride=self.conv2_stride,
            padding=0
        )
        self.relu2 = nn.ReLU()

        conv2_seq_len = (conv2_input_len - self.conv2_kernel) // self.conv2_stride + 1
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        pool2_seq_len = (conv2_seq_len + 1) // 2

        # ==================== FC ====================
        flatten_dim = self.conv2_out_ch * pool2_seq_len

        self.fc1 = nn.Linear(flatten_dim, fc1_hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_hidden_dim, 1)

    def forward(self, x):
        """
        x: [batch, channels, seq_len]
        """

        # -------- Conv1 --------
        if self.conv1_padding != (0, 0):
            x = F.pad(x, self.conv1_padding)

        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)

        # -------- BiLSTM --------
        x = x.permute(0, 2, 1)     # [B, T, C]
        x, _ = self.bilstm(x)
        x = x.permute(0, 2, 1)     # [B, C, T]

        # -------- Conv2 --------
        if self.conv2_padding != (0, 0):
            x = F.pad(x, self.conv2_padding)

        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)

        # -------- FC --------
        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

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
