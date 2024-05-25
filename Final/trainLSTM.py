import torch
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

# print torch_version

print("PyTorch 版本: ", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read excel file

data = pd.read_excel('../data/clean train data.xlsx')

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
class LSTMlWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMlWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(2), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.linear(attn_applied)
        return out


model = LSTMlWithAttention(6,64,1,3)
model.to(device)

# criterion optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)


# # train model

num_epochs = 1500
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
            torch.save(model.state_dict(), "LSTM.pth")
            print("1.3Model save as 1.3.pth")
        # scheduler.step(test_loss)
    if (epoch-1)%5 == 0:
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

Bmse=0.6
eval_model = LSTMlWithAttention().to(device)
eval_model.load_state_dict(torch.load("LSTM.pth"))
new_data = pd.read_excel('../data/clean test data.xlsx')
X_new = new_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].values
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
eval_model.eval()
with torch.no_grad():
    y_pred_new_tensor = eval_model(X_new_tensor).to(device)
    y_pred_new = y_pred_new_tensor.cpu().numpy()

# output mse and rmse

mse = mean_squared_error(y, y_pred_new)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')

# compare model

if Bmse > mse:
  torch.save(model.state_dict(), "B1.3.pth")
  print("save new B1.3.pth model")

#save pred

new_data['Y'] = y_pred_new
new_data.to_excel('result.xlsx', index=False)
