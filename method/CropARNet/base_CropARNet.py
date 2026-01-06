import torch
import torch.nn as nn
import numpy as np

config = {
    "batch_size": 64,
    "weights_units": [64, 32],
    "regressor_units": [64, 32],
    "dropout": 0.3,
}


class SimpleSNPModel(nn.Module):
    def __init__(self, num_snps):
        super().__init__()
        try:
            self.config = config
            if not isinstance(num_snps, int) or num_snps <= 0:
                raise ValueError(f"num_snps must be positive integer, got {num_snps}")
                
            self.attention = self._build_attention_module(num_snps)
            self.regressor = self._build_regressor_module(num_snps)
        except Exception as e:
            raise ValueError(f"Model initialization failed: {str(e)}")

    def _build_attention_module(self, num_snps):
        """Build attention module with error checking"""
        try:
            layers = []
            prev_size = num_snps
            for i, h_size in enumerate(self.config['weights_units']):
                if not isinstance(h_size, int) or h_size <= 0:
                    raise ValueError(f"Invalid hidden size {h_size} in attention layer {i}")
                layers.append(nn.Linear(prev_size, h_size))
                if i < len(self.config['weights_units']) - 1:
                    layers.append(nn.GELU())
                prev_size = h_size
            layers.append(nn.Linear(prev_size, num_snps))
            layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)
        except Exception as e:
            raise ValueError(f"Attention module construction failed: {str(e)}")

    def _build_regressor_module(self, num_snps):
        """Build regressor module with error checking"""
        try:
            layers = []
            prev_size = num_snps
            for i, h_size in enumerate(self.config['regressor_units']):
                if not isinstance(h_size, int) or h_size <= 0:
                    raise ValueError(f"Invalid hidden size {h_size} in regressor layer {i}")
                layers.append(nn.Linear(prev_size, h_size))
                if i < len(self.config['regressor_units']) - 1:
                    layers.append(nn.LayerNorm(h_size))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(self.config['dropout']))
                prev_size = h_size
            layers.append(nn.Linear(prev_size, 1))
            return nn.Sequential(*layers)
        except Exception as e:
            raise ValueError(f"Regressor module construction failed: {str(e)}")

    def forward(self, x):
        """Forward pass with dimension checking"""
        try:
            if x.dim() != 2:
                raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")
                
            pre_sigmoid_weights = self.attention[:-1](x) 
            att_weights = self.attention(x) 
            weighted = x * att_weights + x  # Residual connection
            return self.regressor(weighted).squeeze(), pre_sigmoid_weights
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}")
        

    def train_model(self, train_loader, valid_loader, num_epochs, learning_rate, weight_decay, patience, device):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
                outputs, _ = self(inputs)
                labels = labels
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs,_ = self(inputs)
                    labels = labels
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
                outputs,_ = self(inputs)  
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.squeeze(y_pred)
        return y_pred

