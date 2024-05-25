#林哲 2021211322 13002109 《数字媒体智能技术平台》项目五   基于LSTM的股市最高与最低价预测
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def load_and_preprocess_data(filepath):
    # 计算需要跳过的行数（skiprows）
    skiprows = range(1)  # 如果文件有标题行，确保从文件的第a行开始读取数据

    # 计算需要读取的行数（nrows）
    nrows = 62000

    # 指定需要读取的列（usecols）
    usecols = range(1, 8)  # 调整列的索引以匹配Python的0开始的计数

    # 读取xlsx文件
    df = pd.read_excel(filepath, skiprows=skiprows, nrows=nrows, usecols=usecols)

    df = df.fillna(0)
    X = df.iloc[:, 1:6].values
    X[:, 0] = pd.to_datetime(df.iloc[:, 1].values).view(np.int64)

    y = df.iloc[:, 12:16].values
    return X, y


def custom_split(X, y, train_size=4 * 288, test_size=288):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    total_samples = len(X)
    i = 0

    while i < total_samples:
        # 选择训练数据
        end_train = i + train_size
        if end_train <= total_samples:
            X_train.append(X[i:end_train])
            y_train.append(y[i:end_train])
        else:
            break  # 如果剩余的样本不足以形成一个完整的训练集，就结束循环

        # 选择测试数据
        i = end_train  # 更新索引到训练数据的末尾
        end_test = i + test_size
        if end_test <= total_samples:
            X_test.append(X[i:end_test])
            y_test.append(y[i:end_test])
        else:
            break  # 如果剩余的样本不足以形成一个完整的测试集，就结束循环

        i = end_test  # 更新索引到测试数据的末尾

    # 将列表转换为numpy数组
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_train, X_test, y_train, y_test


def split_and_scale_data(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = custom_split(X, y, train_size=4 * 288, test_size=288)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 获取均值和标准差
    means = scaler.mean_
    std_devs = scaler.scale_

    return X_train, X_test, y_train, y_test, means, std_devs


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    epoch_losses = []  # 用于保存每个epoch的损失
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_losses.append(loss.item())  # 保存当前epoch的损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return epoch_losses


def evaluate_model(model, test_loader, criterion, means_zhj, std_devs_zhj, device):
    model.eval()
    y_predzhj = []
    y_truezhj = []
    collected_inputs = []  # 初始化一个列表来收集特定列的数据
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_predzhj.extend(outputs[:, 5].cpu().numpy())
            y_truezhj.extend(labels[:, 5].cpu().numpy())
            collected_inputs.extend(inputs[:, 0].cpu().numpy())  # 收集第7列的数据

            # 收集第7列的数据，并立即转换为numpy数组
            # collected_inputs.append(inputs[:, :, 6].cpu().numpy())

    if collected_inputs:
        # 使用np.vstack确保数组可以垂直堆叠
        collected_inputs_np = np.vstack(collected_inputs)
    else:
        # 如果列表为空，则创建一个空的numpy数组
        collected_inputs_np = np.array([])

    # 执行标准化逆转换
    if collected_inputs_np.size > 0:
        t_recod_true = collected_inputs_np * std_devs_zhj[0] + means_zhj[0]
    else:
        t_recod_true = np.array([])

    # 转换为NumPy数组以进行评分计算
    y_true = np.array(y_truezhj)
    y_pred = np.array(y_predzhj)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f'MSE: {mse:.4f}')

    return t_recod_true, y_truezhj, y_predzhj, [mse, rmse]


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    filepath = 'stock_data.xlsx'
    X, y = load_and_preprocess_data(filepath)
    X_train, X_test, y_train, y_test, means_zhj, std_devs_zhj = split_and_scale_data(X, y)
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=64)

    model = LSTMModel(input_size=5, hidden_size=50, num_layers=2, output_size=5).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    epoch_losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    t_recod_true, y_true, y_pred, metrics = evaluate_model(model, test_loader, criterion, means_zhj, std_devs_zhj,
                                                           device)

    # 将损失和评价指标写入Excel
    with pd.ExcelWriter('results.xlsx', engine='openpyxl') as writer:
        # 写入每个epoch的损失
        df_loss = pd.DataFrame(epoch_losses, columns=['Loss'])
        df_loss.to_excel(writer, sheet_name='Sheet1', index=False)

        # 写入评价指标
        df_metrics = pd.DataFrame([metrics], columns=['MSE', 'RMSE'])
        df_metrics.to_excel(writer, sheet_name='Sheet2', startrow=0, index=False)

        # 写入预测结果
        df_results = pd.DataFrame({
            'time': t_recod_true.tolist(),
            'true': y_true,
            'pred': y_pred
        })
        df_results.to_excel(writer, sheet_name='Sheet3', index=False)


if __name__ == '__main__':
    main()

