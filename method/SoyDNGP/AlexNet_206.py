import torch
from torch import nn 
from torch.nn import Module
import numpy as np
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
 
        self.h = h
        self.w = w
 
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
 
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
 
        return out

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(3,32,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.ReLU(),

            CA_Block(32,206,206,reduction=16),

            nn.Conv2d(32,64,kernel_size=4,padding=1,padding_mode='reflect',stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size=3,padding=1,padding_mode='reflect',stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(64,128,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Conv2d(128,128,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(128,256,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(256,256,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(256,512,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(512,512,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Conv2d(512,1024,kernel_size=3,padding=1,padding_mode='reflect',stride=2,bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Conv2d(1024,1024,kernel_size=3,padding=1,padding_mode='reflect',stride=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            CA_Block(1024,7,7,reduction=16),           
            
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.ReLU(),

            # nn.Linear(50176,6400),
            # nn.Dropout(0.4),
            # nn.ReLU(),

            nn.Linear(50176,1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 转为NCHW
        return self.net(x)
    
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
