import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepGS(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Identity()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout1d(p = 0.2)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            dummy_out = self.pool(self.conv1(dummy))
            conv_out_dim = dummy_out.view(1, -1).size(1)
        self.fc1 = nn.Linear(in_features=conv_out_dim, out_features=32)
        self.drop2 = nn.Dropout1d(p = 0.1)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.act2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1).unsqueeze(1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x) 
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        return x

    def train_model(self, train_loader, valid_loader, num_epochs, learning_rate, momentum, weight_decay, patience, device):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.L1Loss()
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