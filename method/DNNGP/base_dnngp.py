import torch
import torch.nn as nn
import numpy as np


class DNNGP(nn.Module):
    def __init__(self, input_size, dropout1, dropout2):
        super().__init__()
        self.CNN1 = nn.Conv1d(in_channels = 1, out_channels=64, kernel_size=4)
        self.Relu1 = nn.ReLU()
        self.Drop1 = nn.Dropout(dropout1)

        self.Batchnorm = nn.BatchNorm1d(num_features=64)

        self.CNN2 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=4)
        self.Relu2 = nn.ReLU()
        self.Drop2 = nn.Dropout(dropout2)

        self.CNN3 = nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=4)
        self.Relu3 = nn.ReLU()

        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(in_features=64*(input_size-9), out_features=3)
        self.Output = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = self.CNN1(x)
        x = self.Relu1(x)
        x = self.Drop1(x)
        x = self.Batchnorm(x)
        x = self.CNN2(x)
        x = self.Relu2(x)
        x = self.Drop2(x)
        x = self.CNN3(x)
        x = self.Relu3(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Output(x)
        return x


    def train_model(self, train_loader, valid_loader, num_epochs, learning_rate, weight_decay, patience, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        self.to(device)

        best_loss = float('inf')
        best_state = None
        trigger_times = 0

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    labels = labels.unsqueeze(1)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)

            # ---------- Early stopping ----------
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = self.state_dict()
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return best_loss
        
    def predict(self, test_loader):
        self.eval()
        y_pred = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = self(inputs)  
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.squeeze(y_pred)
        return y_pred