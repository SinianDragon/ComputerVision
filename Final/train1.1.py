import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 打印 PyTorch 版本号
print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 Excel 文件
file_path = '../data/train data.xlsx'
skiprows = 0
df = pd.read_excel(file_path, skiprows=skiprows)

# 提取输入（X）和输出（y）
X = df.iloc[:, 1:].values  # 输入特征：第2列到最后一列
y = df.iloc[:, 0].values  # 输出目标：第1列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理（标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换数据为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建 DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 输出层（双向 LSTM）

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 初始隐藏状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 初始细胞状态
        out, _ = self.lstm(x, (h0, c0))  # LSTM 输出
        out = self.fc(out[:, -1, :])  # 全连接层
        return out

# 实例化模型
input_size = X_train.shape[1]
hidden_size = 128
num_layers = 3
model = LSTMModel(input_size, hidden_size, num_layers).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 训练模型
num_epochs = 100
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.unsqueeze(1))
        test_loss = criterion(y_pred, y_test_tensor.view(-1, 1))
        print(f'Test Loss: {test_loss.item():.4f}')

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "best-LSTMmodel.pth")

# 加载最佳模型并用于预测
best_model = LSTMModel(input_size, hidden_size, num_layers).to(device)
best_model.load_state_dict(torch.load("best-LSTMmodel.pt"))
best_model.eval()

Bmse=0.6
new_data = pd.read_excel('../data/test data.xlsx')
X_new = new_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_new_tensor = best_model(X_new_tensor).to(device)
    y_pred_new = y_pred_new_tensor.cpu().numpy()

# output mse and rmse

mse = mean_squared_error(y, y_pred_new)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')

# compare model

if Bmse > mse:
  torch.save(model.state_dict(), "LSTM.pth")
  print("save new B1.3.pth model")

#save pred

new_data['Y'] = y_pred_new
new_data.to_excel('result.xlsx', index=False)
