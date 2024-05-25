import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 打印 PyTorch 版本号
print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_excel('../data/train data.xlsx')
X = df.iloc[:, 1:].values
Y = df.iloc[:, 0].values

scaler_X = MinMaxScaler((0, 1))
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

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

optimizer = optim.Adam(model.parameters(), lr=0.012)

criterion = nn.MSELoss().to(device)

X_train_tensor = torch.Tensor(X_train).float().to(device)
Y_train_tensor = torch.Tensor(Y_train).float().unsqueeze(1).to(device)
X_test_tensor = torch.Tensor(X_test).float().to(device)
Y_test_tensor = torch.Tensor(Y_test).float().unsqueeze(1).to(device)

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_Y = Y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor).to(device)
    loss = criterion(outputs, Y_test_tensor).to(device)
    mse = loss.item()
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    torch.save(model.state_dict(), "others_model.pth")

# 7. 使用模型预测新数据
new_data = pd.read_excel('../data/test data.xlsx')
X_new = new_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    y_pred_new_tensor = model(X_new_tensor).to(device)
    y_pred_new = y_pred_new_tensor.cpu().numpy()

# 将预测结果保存到新数据中
new_data['Predicted'] = y_pred_new
new_data.to_excel('result.xlsx', index=False)