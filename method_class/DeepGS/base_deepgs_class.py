import torch
import torch.nn as nn
import numpy as np


class DeepGS(nn.Module):
    """
    DeepGS for multi-class classification
    Fully compatible with:
    - Optuna hyperparameter optimization
    - 10-fold cross-validation
    """

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        # ========= Feature extractor =========
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=18,
            stride=1
        )
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=0.2)

        # ========= Dynamically infer FC input =========
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            dummy = self.pool1(self.act1(self.conv1(dummy)))
            conv_out_dim = dummy.view(1, -1).size(1)

        # ========= Classifier =========
        self.fc1 = nn.Linear(conv_out_dim, 32)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(32, num_classes)

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc2(x)   # logits
        return x

    # ==================================================
    # Training (classification)
    # ==================================================
    def train_model(
        self,
        train_loader,
        valid_loader,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        patience: int,
        device: torch.device
    ):
        self.to(device)

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None
        trigger_times = 0

        for epoch in range(num_epochs):
            # -------- Train --------
            self.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)

            # -------- Validation --------
            self.eval()
            valid_loss = 0.0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)

            valid_loss /= len(valid_loader.dataset)

            # -------- Early stopping --------
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = self.state_dict()
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return best_loss

    # ==================================================
    # Prediction (classification)
    # ==================================================
    def predict(self, test_loader):
        self.eval()
        device = next(self.parameters()).device

        y_pred = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)        # (N, C)
                preds = torch.argmax(outputs, dim=1)
                y_pred.append(preds.cpu().numpy())

        return np.concatenate(y_pred, axis=0)
