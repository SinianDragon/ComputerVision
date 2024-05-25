import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import numpy as np

# 打印 PyTorch 版本号
print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_excel('../data/clean train data.xlsx')

# 提取特征和目标变量
x = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
y = df['Y'].values

# 数据归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(x)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3)

# 转换数据为 PyTorch Tensor
X_train_tensor = torch.Tensor(X_train).unsqueeze(2).to(device)  # 添加一个维度作为序列长度
Y_train_tensor = torch.Tensor(Y_train).unsqueeze(1).to(device)
X_test_tensor = torch.Tensor(X_test).unsqueeze(2).to(device)
Y_test_tensor = torch.Tensor(Y_test).unsqueeze(1).to(device)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out


# 实例化模型
input_size = 1  # 每个时间步输入的特征数
hidden_size = 128
num_layers = 4
model = GRUModel(input_size, hidden_size, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

# 训练模型
num_epochs = 1000
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "LSTM.pth")

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, Y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# 使用模型进行预测新数据
new_data = pd.read_excel('../data/clean test data.xlsx')
X_new = new_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.Tensor(X_new_scaled).unsqueeze(2).to(device)

model.eval()
with torch.no_grad():
    y_pred_new_tensor = model(X_new_tensor)
    y_pred_new = y_pred_new_tensor.cpu().numpy().squeeze()

# output mse and rmse

mse = mean_squared_error(y, y_pred_new)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')

# 将预测结果保存到新数据中
new_data['Y'] = y_pred_new
new_data.to_excel('result.xlsx', index=False)
