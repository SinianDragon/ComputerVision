#林哲 2021211322 13002109 《数字媒体智能技术平台》项目三
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

print("PyTorch Version: ", torch.__version__)

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 第一层卷积层：输入通道1，输出通道32，卷积核大小3x3，步长1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 第二层卷积层：输入通道32，输出通道64，卷积核大小3x3，步长1
        self.conv3 = nn.Conv2d(64, 128, 3, 1)  # 第三层卷积层：输入通道64，输出通道128，卷积核大小3x3，步长1
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 全连接层1：输入维度是4x4x128，输出维度是1024
        self.fc2 = nn.Linear(512, 10)  # 全连接层2：输入维度是1024，输出维度是10(1024x10次运算)


# 前向传播函数
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积层，使用ReLU激活函数
        x = F.max_pool2d(x, 2, 2)  # 最大池化，池化窗口大小2x2，步长2
        x = F.relu(self.conv2(x))  # 第二层卷积层，使用ReLU激活函数
        x = F.max_pool2d(x, 2, 2)  # 再次进行最大池化
        x = F.relu(self.conv3(x))  # 第三层卷积层，使用ReLU激活函数
        x = F.max_pool2d(x, 2, 2)  # 最后的最大池化
        x = F.adaptive_avg_pool2d(x, (3, 3))  # 将特征图大小自适应为 (3, 3),使用 adaptive pooling 来自适应输入的大小
        x = x.view(-1, 3 * 3 * 128)  # 将特征图展平为多组一维向量，其中向量的参数个数为3 * 3 * 128，向量的个数取决于池化后的窗口，-1指自适应
        x = F.relu(self.fc1(x))  # 全连接层1维度500，将多组一维向量全连接，在连接后使用ReLU激活函数
        x = self.fc2(x)  # 全连接层2维度10，输出未经激活的权重
        return F.log_softmax(x, dim=1)  # 使用softmax对列进行对数(log)归一化，计算最终的概率

# 定义数据标准化(数据转换)函数
transform = transforms.Compose([transforms.ToTensor(),  # 将数据转换为张量
transforms.Normalize((0.1307,), (0.3081,))])  # 对数据进行标准化处理

# 训练数据集
train_dataset = datasets.MNIST("./mnist_data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

# 测试数据集（下载测试数据，）
test_dataset = datasets.MNIST("./mnist_data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch}, iteration: {batch_idx}, Loss: {loss.item()}")

# 定义测试函数
def test(model, device, test_loader):
    # 开始模型评估
    model.eval()
    # 初始化损失和准确度(置信率)
    test_loss = 0
    correct = 0
    # 取消梯度计算，并进行loss计算
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), "mnist_cnn.pt")
