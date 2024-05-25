# 2021211322 13002109 林哲《数字媒体智能技术平台》 大作业
import torch
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# print torch_version

print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read excel file

data = pd.read_excel('../data/train data.xlsx')

# read data

X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
y = data['Y'].values

# standardize data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split train test and transform to tensor

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# create DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# complex model(multiple layers == 5)
class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(16, 1)


    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x

model = DeepModel()
model.to(device)

# criterion optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)


# train model

num_epochs = 1000
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        test_loss=running_loss/len(train_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "1.3.pth")
        # scheduler.step(test_loss)
    if (epoch+1)%5 == 0:
     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}')
     print('-------------------------------------------------')

# eval test-data

model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    y_pred_tensor = model(X_test_tensor).to(device)
    test_loss = criterion(y_pred_tensor.squeeze(), y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')

# eval eval-data

Bmse=0.3
eval_model = DeepModel().to(device)
eval_model.load_state_dict(torch.load("B1.3.pth"))
# new_data = pd.read_excel('../data/test data.xlsx')
new_data = pd.read_excel('../data/test data.xlsx')
X_new = new_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
eval_model.eval()
with torch.no_grad():
    y_pred_new_tensor = eval_model(X_new_tensor).to(device)
    y_pred_new = y_pred_new_tensor.cpu().numpy()

# output mse and rmse

mse = mean_squared_error(y, y_pred_new)
mae = mean_absolute_error(y,y_pred_new)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')

# compare model

if Bmse > mse:
  torch.save(model.state_dict(), "B1.3.pth")
  print("save new B1.3.pth model")

#save pred

new_data['Y'] = y_pred_new
new_data.to_excel('result-clean.xlsx', index=False)
