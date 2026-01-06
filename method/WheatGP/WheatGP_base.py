# model_definition.py
import torch
import torch.nn as nn

class ConvPart(nn.Module):
    def __init__(self):
        super(ConvPart, self).__init__()
        # First convolutional layer: input channels = 1, output channels = 2, kernel size = 1, padding = 1
        self.conv0 = nn.Conv1d(1, 2, 1, padding=1)
        # ReLU activation function after the first convolutional layer
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv1d(2, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(4, 8, 9, padding=1)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.drop(x)

        return x
class ShapeModule(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class
        super(ShapeModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, adjust_dim=True, concat=True):
        if adjust_dim:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x3 = x3.unsqueeze(1)
            x4 = x4.unsqueeze(1)
            x5 = x5.unsqueeze(1)
        if concat:
            A_flat = x1.view(x1.size(0), -1)
            B_flat = x2.view(x2.size(0), -1)
            C_flat = x3.view(x3.size(0), -1)
            D_flat = x4.view(x4.size(0), -1)
            E_flat = x5.view(x5.size(0), -1)
            output = torch.cat((A_flat, B_flat, C_flat, D_flat, E_flat), dim=1)
            output = output.reshape(output.shape[0], 1, -1)

            return output
        else:

            return x1, x2, x3, x4, x5

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        # Call the constructor of the parent class
        super(LSTMModule, self).__init__()
        # Define an LSTM layer with the specified input size, hidden size, number of layers, and batch first flag
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.drop(lstm_out)
        return lstm_out

class wheatGP_base(nn.Module):
    def __init__(self,  nsnp):
        super(wheatGP_base, self).__init__()
        self.ConvPart = ConvPart()
        self.lstm = LSTMModule(5 * 8*(nsnp//5 - 4), 128)
        self.shape_module = ShapeModule()
        self.fc = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x1, x2, x3, x4, x5):
        x1, x2, x3, x4, x5 = self.shape_module(x1, x2, x3, x4, x5, adjust_dim=True, concat=False)
        A = self.ConvPart(x1)
        B = self.ConvPart(x2)
        C = self.ConvPart(x3)
        D = self.ConvPart(x4)
        E = self.ConvPart(x5)

        output = self.shape_module(A, B, C, D, E, adjust_dim=False)  # Assume the dimensions have been adjusted before
        output = self.lstm(output)

        output = output[:, -1, :]
        output = self.fc(output)

        return output

    def freeze_layers(self, freeze_conv=True, freeze_lstm=True, freeze_fc=True):
        # Freeze or unfreeze the parameters of the convolutional part
        for param in self.ConvPart.parameters():
            param.requires_grad = not freeze_conv
        # Freeze or unfreeze the parameters of the LSTM module
        for param in self.lstm.parameters():
            param.requires_grad = not freeze_lstm
        # Freeze or unfreeze the parameters of the fully connected layer
        for param in self.fc.parameters():
            param.requires_grad = not freeze_fc