import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 打印 PyTorch 版本号
print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 Excel 文件
file_path = '../data/training train data.xlsx'
df = pd.read_excel(file_path)

# 提取输入特征和输出目标
X = df.iloc[:, 1:].values  # 所有列除了最后一列为输入特征
y = df.iloc[:, 0].values   # 最后一列为输出目标

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换数据为 PyTorch Tensor
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 创建 DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 定义优化模型
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# 实例化模型
input_size = X_train.shape[1]
model = RegressionModel(input_size).to(device)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 60  # 增加训练时长
# best_loss = float('inf')
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     epoch_loss = running_loss / len(train_loader)
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
#
#     # 在每个 epoch 结束后，检查测试集上的损失并保存最佳模型
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_test)
#         test_loss = criterion(y_pred, y_test.unsqueeze(1))
#         print(f'Test Loss: {test_loss.item():.4f}')
#
#         # 保存损失最小的模型
#         if test_loss < best_loss:
#             best_loss = test_loss
#             torch.save(model.state_dict(), "best_regression_model.pth")
#
#加载保存的最佳模型并进行预测
best_model = RegressionModel(input_size).to(device)
best_model.load_state_dict(torch.load("best_regression_model.pth"))

# 对新数据进行预测
nrows = 100
datapre = [0]*nrows
df = pd.read_excel(file_path,nrows=nrows)
new_data = df.iloc[:, 1:].values
new_data = scaler.transform(new_data)  # 调整新数据的形状并进行标准化
new_data_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)
for i,data in enumerate(new_data_tensor):
    with torch.no_grad():
        prediction = best_model(data)
        datapre[i] = prediction.item()
        print(f'预测值: {prediction.item()}')
# 写入预测值
with pd.ExcelWriter('results.xlsx', engine='openpyxl') as writer:
  df_pre = pd.DataFrame(datapre, columns=['Prediction'])
  df_pre.to_excel(writer, sheet_name='Sheet1', index=False)
