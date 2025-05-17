# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('nyc_taxi.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# 添加时间特征
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour.astype(np.float32)
    df['dayofweek'] = df.index.dayofweek.astype(np.float32)
    df['day'] = df.index.day.astype(np.float32)
    df['month'] = df.index.month.astype(np.float32)
    df['weekofyear'] = df.index.isocalendar().week.astype(np.float32)
    return df

data = create_features(data)

# 数据标准化
scaler = MinMaxScaler()
data['value_scaled'] = scaler.fit_transform(data[['value']]).astype(np.float32)

# 检查数据类型
print(data.dtypes)

# 2. 鏋勫缓鏁版嵁闆?
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=24, forecast_horizon=1):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        features = self.data.iloc[idx:idx+self.window_size][['value_scaled', 'hour', 'dayofweek', 'day', 'month', 'weekofyear']].values
        target = self.data.iloc[idx+self.window_size:idx+self.window_size+self.forecast_horizon]['value_scaled'].values
        
        features = torch.FloatTensor(features)
        target = torch.FloatTensor(target)
        
        return features, target

train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

window_size = 24 * 7  # 一周的数据作为窗口
forecast_horizon = 1  # 预测下一个时间点

train_dataset = TimeSeriesDataset(train_data, window_size, forecast_horizon)
test_dataset = TimeSeriesDataset(test_data, window_size, forecast_horizon)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 模型构建
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3):
        super(CNNLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size
        
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
                                kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = 6
hidden_size = 64
num_layers = 2
output_size = forecast_horizon
kernel_size = 3

model = CNNLSTMModel(input_size, hidden_size, num_layers, output_size, kernel_size)

# 4. 璁粌妯″瀷
learning_rate = 0.001
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 5. 异常检测
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for features, targets in test_loader:
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.cpu().numpy())

predictions = np.array(predictions).flatten()
actuals = np.array(actuals).flatten()

errors = np.abs(predictions - actuals)
threshold = np.percentile(errors, 95)
print(f"Anomaly threshold: {threshold:.4f}")

anomalies = errors > threshold

plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
anomaly_indices = np.where(anomalies)[0]
plt.scatter(anomaly_indices, actuals[anomaly_indices], color='red', label='Anomaly')
plt.legend()
plt.title('Anomaly Detection Results')
plt.show()
