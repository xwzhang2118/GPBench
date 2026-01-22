import torch
import torch.nn as nn
import numpy as np


class MVP(nn.Module):
    def __init__(self, input_size, nb_filters=32):
        super().__init__()
        self.input_size = input_size
        self.nb_filters = nb_filters
        self.kernel_size = (3, 1)

        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=nb_filters, 
                                 kernel_size=self.kernel_size, padding='same')
        self.conv2d2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, 
                                 kernel_size=self.kernel_size, padding='same')
        self.relu = nn.ReLU()
        
        flattened_dim = nb_filters * input_size * 1
        self.fc1 = nn.Linear(flattened_dim, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        if x.dim() == 4 and x.size(1) != 1:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        elif x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(3)
        elif x.dim() == 5:
            x = x.squeeze(-1)
        
        for i in range(2):
            x_res = x
            if i == 0:
                x = self.conv2d1(x) 
            else:
                x = self.conv2d2(x)
            x = self.relu(x)
            x = self.conv2d2(x)
            x = x + x_res
            x = self.relu(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    def train_model(self, train_loader, valid_loader, num_epochs, learning_rate, patience, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        self.to(device)

        best_loss = float('inf')
        best_state = None
        trigger_times = 0

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)

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
        device = next(self.parameters()).device
        y_pred = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = self(inputs)
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.squeeze(y_pred)
        return y_pred
