import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
# 打印 PyTorch 版本号
print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_excel('../data/clean train data.xlsx')
dt = pd.read_excel('../data/clean test data.xlsx')
#将第一列为零的数据删除并令存到新的dataframe中
#第一列为零的数据是测试数据，不参与训练
df_train = df
X = df.iloc[:, 1:].values
Y = df.iloc[:, 0].values
df_test = dt
X_1 = df_test.iloc[:, 1:].values
scaler_X = MinMaxScaler((0, 1))
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)


criterion = nn.MSELoss()

X_train_tensor = torch.Tensor(X_train).float().to(device)
Y_train_tensor = torch.Tensor(Y_train).float().unsqueeze(1).to(device)
X_test_tensor = torch.Tensor(X_test).float().to(device)
Y_test_tensor = torch.Tensor(Y_test).float().unsqueeze(1).to(device)

num_epochs = 150
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_Y = Y_train_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X).to(device)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor).to(device)
    loss = criterion(outputs, Y_test_tensor)
    mse = loss.item()
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
torch.save(model.state_dict(), "1.pth")
#将测试数据放入模型中进行预测

#先将X_1进行归一化
X_1_scaled = scaler_X.transform(X_1)
#将X_1转换为tensor
X_test_tensor_1 = torch.Tensor(X_1_scaled ).float().to(device)

with torch.no_grad():
    y_pred_new_tensor = model(X_test_tensor_1).to(device)
    y_pred_new = y_pred_new_tensor.cpu().numpy()
    mse = mean_squared_error(Y, y_pred_new)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')

#将outputs_1和x_1合并
df_test.loc[:, 'Y'] = y_pred_new

#将df_test保存到新的excel文件中
df_test.to_excel('predict.xlsx', index=False)


