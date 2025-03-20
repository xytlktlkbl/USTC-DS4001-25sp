import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class BostonHousingDataset(data.Dataset):
    """
    自定义数据集类，加载 Boston 房价数据，并转换为 PyTorch 数据集格式
    """
    def __init__(self, train=True):
        # 下载 Boston Housing 数据集
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        df = boston.frame  # 获取 Pandas DataFrame

        # 特征与标签
        X = df[boston.feature_names].values  # 提取特征
        y = df["MEDV"].values  # 提取目标值（房价）

        # 归一化特征数据（保持均值0，标准差1）
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.expand_dims(y, axis=1)  # 变为列向量

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 选择数据集类型
        if train:
            self.X, self.y = X_train, y_train
        else:
            self.X, self.y = X_test, y_test

        # 转换为 PyTorch 张量
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.X)

    def __getitem__(self, index):
        """返回单个样本"""
        return self.X[index], self.y[index]


class FNN(nn.Module):
    """
    经典的全连接神经网络（Feedforward Neural Network, FNN）
    结构：输入层 -> 隐藏层(ReLU) -> 输出层
    """
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 输入层到隐藏层
        #线性层，利用矩阵乘法，将输入映射到64维
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(64, 1)  # 隐藏层到输出层

    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Trainer:
    """
    训练和评估神经网络模型的类
    """
    def __init__(self, model, train_loader, test_loader, device="cpu", lr=0.01, epochs=300):
        self.model = model.to(device)  # 将模型移动到指定设备（CPU/GPU）
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs

        # 使用均方误差（MSE）作为损失函数 loss function
        self.criterion = nn.MSELoss()

        # 使用 SGD 优化器 ,即不断利用梯度迭代
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # 记录误差
        self.train_losses = []
        self.test_losses = []

    def train(self):
        """训练模型"""
        #model是FNN的实例
        self.model.train()  # 设置为训练模式
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()  # 梯度清零
                outputs = self.model(batch_X)  # 前向传播
                loss = self.criterion(outputs, batch_y)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                total_loss += loss.item()
            
            if epoch == self.epochs - 1:
                print_computational_graph(loss.grad_fn)
            
            # 计算训练误差
            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # 计算测试误差
            avg_test_loss = self.evaluate(calc_loss=True)
            self.test_losses.append(avg_test_loss)

            # 每 1 轮打印一次损失
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    def evaluate(self, calc_loss=False):
        """在测试集上评估模型"""
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        with torch.no_grad():  # 不计算梯度
            for batch_X, batch_y in self.test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)

        if not calc_loss:
            print(f"Test MSE Loss: {avg_loss:.4f}")

        return avg_loss

    def plot_losses(self):
        """绘制训练误差和测试误差曲线"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label="Train Loss", color='blue')
        plt.plot(range(1, self.epochs + 1), self.test_losses, label="Test Loss", color='red')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Testing Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()


def print_computational_graph(grad_fn, indent=0):
    if grad_fn is None:
        return
    print(" " * indent + f"↳ {type(grad_fn).__name__}")  # 只打印类名
    for next_fn in grad_fn.next_functions:
        if next_fn[0] is not None:
            print_computational_graph(next_fn[0], indent + 4)


# ========== 主程序 ==========
if __name__ == "__main__":
    # 设备选择：如果有 GPU 就用 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集
    train_dataset = BostonHousingDataset(train=True)
    test_dataset = BostonHousingDataset(train=False)

    # 创建数据加载器
    #batch 分批
    #shuffle 打乱

    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 初始化模型
    model = FNN(input_dim=train_dataset.X.shape[1])

    # 训练器
    #lr learning rate，类似步长
    #epochs 将完整的数据跑完称为1个epoch
    trainer = Trainer(model, train_loader, test_loader, device=device, lr=0.001, epochs=50)

    # 训练模型
    trainer.train()

    # 评估模型
    trainer.evaluate()

    # 绘制误差曲线
    trainer.plot_losses()
