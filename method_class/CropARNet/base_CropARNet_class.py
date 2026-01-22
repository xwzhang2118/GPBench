import torch
import torch.nn as nn
import numpy as np

config = {
    "batch_size": 64,
    "weights_units": [64, 32],
    "classifier_units": [64, 32],
    "dropout": 0.3,
}


class SimpleSNPModel(nn.Module):
    """
    Classification version of SimpleSNPModel
    (Attention + Residual + MLP)
    """

    def __init__(self, num_snps: int, num_classes: int):
        super().__init__()

        if not isinstance(num_snps, int) or num_snps <= 0:
            raise ValueError(f"num_snps must be positive integer, got {num_snps}")
        if not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError(f"num_classes must be >=2, got {num_classes}")

        self.config = config
        self.num_classes = num_classes

        self.attention = self._build_attention_module(num_snps)
        self.classifier = self._build_classifier_module(num_snps, num_classes)

    # ==================================================
    # Attention module
    # ==================================================
    def _build_attention_module(self, num_snps):
        layers = []
        prev_size = num_snps

        for i, h_size in enumerate(self.config["weights_units"]):
            if h_size <= 0:
                raise ValueError(f"Invalid hidden size {h_size}")
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.GELU())
            prev_size = h_size

        layers.append(nn.Linear(prev_size, num_snps))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    # ==================================================
    # Classifier module
    # ==================================================
    def _build_classifier_module(self, num_snps, num_classes):
        layers = []
        prev_size = num_snps

        for i, h_size in enumerate(self.config["classifier_units"]):
            if h_size <= 0:
                raise ValueError(f"Invalid hidden size {h_size}")
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.LayerNorm(h_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config["dropout"]))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, num_classes))
        return nn.Sequential(*layers)

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")

        # Attention
        pre_sigmoid_weights = self.attention[:-1](x)
        att_weights = self.attention(x)

        # Residual weighted SNPs
        weighted = x * att_weights + x

        logits = self.classifier(weighted)
        return logits, pre_sigmoid_weights

    # ==================================================
    # Training (classification)
    # ==================================================
    def train_model(
        self,
        train_loader,
        valid_loader,
        num_epochs,
        learning_rate,
        weight_decay,
        patience,
        device
    ):
        self.to(device)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
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
                labels = labels.to(device).long()

                optimizer.zero_grad()
                outputs, _ = self(inputs)
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
                    labels = labels.to(device).long()

                    outputs, _ = self(inputs)
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
                    print(f"Early stopping at epoch {epoch + 1}")
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
                outputs, _ = self(inputs)  # logits
                preds = torch.argmax(outputs, dim=1)
                y_pred.append(preds.cpu().numpy())

        return np.concatenate(y_pred, axis=0)
