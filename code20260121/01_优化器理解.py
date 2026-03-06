# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/21 21:31
Create User : 19410
Desc : PyTorch二分类全流程示例代码（包含SwanLab和TensorBoard监控）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# https://docs.wandb.ai/
# import wandb
# https://swanlab.cn/
# pip install swanlab
# 导入SwanLab，这是一个类似W&B的国产AI实验跟踪工具
import swanlab
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ===================== 1. 自定义数据集 =====================
class CustomDataset(Dataset):
    """自定义二分类数据集：生成带噪声的二维高斯分布数据"""

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # 生成两类数据：类别0（中心(1,1)）、类别1（中心(-1,-1)）
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        # 类别0数据：以(1, 1)为中心，生成 num_samples/2 个样本，每个样本2个特征
        class0 = np.random.normal(loc=[1, 1], scale=0.5, size=(num_samples // 2, 2))
        # 为类别0生成全0的标签矩阵
        class0_label = np.zeros((num_samples // 2, 1))
        # 类别1数据：以(-1, -1)为中心，生成 num_samples/2 个样本
        class1 = np.random.normal(loc=[-1, -1], scale=0.5, size=(num_samples // 2, 2))
        # 为类别1生成全1的标签矩阵
        class1_label = np.ones((num_samples // 2, 1))
        # 将两类数据在垂直方向拼接（vstack），并转换为 PyTorch 需要的 float32 类型
        self.data = np.vstack([class0, class1]).astype(np.float32)
        self.labels = np.vstack([class0_label, class1_label]).astype(np.float32)
        # 生成一个打乱的索引数组，用于打乱数据集，防止模型在训练时死记硬背某种顺序
        shuffle_idx = np.random.permutation(num_samples)
        self.data = self.data[shuffle_idx]
        self.labels = self.labels[shuffle_idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 将 NumPy 数组转换为 PyTorch 的 Tensor
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])


# ===================== 2. 定义神经网络模型 =====================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 全连接层1：输入维度为2（两个特征），输出维度为16
        self.fc1 = nn.Linear(2, 16)  # 输入层：2个特征 → 隐藏层16个神经元
        # 全连接层2：输入维度16，输出维度8
        self.fc2 = nn.Linear(16, 8)  # 隐藏层
        # 全连接层3（输出层）：输入维度8，输出维度1（因为是二分类）
        self.fc3 = nn.Linear(8, 1)  # 输出层：二分类输出1个值（sigmoid后判断）
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 输出0-1之间的概率
        return x


# ===================== 3. 训练配置与初始化 =====================
lr = 0.01  # 初始学习率
num_epochs = 30  # 训练轮数（遍历整个数据集的次数）

# swanlab登录 todo: 更改成自己的api_key
swanlab.login(api_key="E0rsxPAXIbjBF8LDzntv3")

# 初始化一个SwanLab项目，记录超参数
swanlab.init(
    project="my-pytorch-demo",
    config={
        "lr": lr,
        "num_epochs": num_epochs
    }
)

# 设备配置：优先使用GPU，没有则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化数据集和数据加载器
train_dataset = CustomDataset(num_samples=2000)
# 使用 DataLoader 包装数据集，设置批大小为 32，并且在每个 epoch 开始时打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数、优化器
model = SimpleNN().to(device)
# 定义损失函数：二元交叉熵损失 (Binary Cross Entropy Loss)，适用于二分类
criterion = nn.BCELoss()  # 二分类交叉熵损失（BCELoss）

# optimizer = optim.Adam(model.parameters(), lr=lr)  # 初始学习率0.01
# 定义优化器：这里使用的是 AdamW（带权重衰减的 Adam），负责根据梯度更新模型参数
optimizer = optim.AdamW(model.parameters(), lr=lr)  # 初始学习率0.01
# 学习率调度器：StepLR - 每step_size个epoch，学习率乘以gamma
# 作用是：每经过 step_size (2) 个 epoch，就把当前学习率乘以 gamma (0.5)
# 这有助于模型在训练后期进行更精细的收敛
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# 初始化 TensorBoard 日志记录器，按当前时间生成文件夹名，避免覆盖历史记录
log_dir = f"runs/simple_nn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard日志保存路径: {log_dir}")

# ===================== 4. 训练主循环 =====================
# 获取一个 epoch 中包含的 batch 数量
total_steps = len(train_loader)

for epoch in range(num_epochs):
    # 将模型设置为训练模式（这会启用 Dropout 和 BatchNorm 的训练行为，虽然本代码没用到）
    model.train()  # 模型进入训练模式
    # 初始化当前 epoch 的累计损失和正确预测数量
    running_loss = 0.0
    correct = 0
    total = 0
    # 遍历数据加载器，每次取出一个 batch_size (32) 的数据和标签
    for i, (inputs, labels) in enumerate(train_loader):
        # 将数据移到指定设备（GPU/CPU）
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 1. 前向传播：将输入送入模型，得到预测输出
        outputs = model(inputs)
        # 2. 计算损失：比较预测输出与真实标签的差异
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 统计损失和准确率
        running_loss += loss.item()
        # 统计准确率：
        # 因为经过了 Sigmoid，输出是 0-1 的概率。概率 > 0.5 判定为类别 1，否则为类别 0
        predicted = (outputs > 0.5).float()
        # 累计当前 batch 的总样本数
        total += labels.size(0)
        # 累计预测正确的样本数
        correct += (predicted == labels).sum().item()

    # 当前 epoch 结束后，调度器更新一次学习率
    scheduler.step()

    # 计算当前epoch的平均损失和准确率
    epoch_loss = running_loss / total_steps
    epoch_acc = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

    # 打印训练信息
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {current_lr:.6f}')

    # 写入TensorBoard日志
    writer.add_scalar('Training/Loss', epoch_loss, epoch)  # 损失曲线
    writer.add_scalar('Training/Accuracy', epoch_acc, epoch)  # 准确率曲线
    writer.add_scalar('Training/Learning_Rate', current_lr, epoch)  # 学习率变化
    swanlab.log({
        "Training/epoch_Loss": epoch_loss,
        "Training/epoch_accuracy": epoch_acc,
        "Training/epoch_lr": current_lr,
    }, step=epoch)

# ===================== 5. 收尾工作 =====================
writer.close()  # 关闭 TensorBoard 写入流，确保数据刷入磁盘
swanlab.finish()  # 通知 SwanLab 训练任务结束
torch.save(model.state_dict(), 'simple_nn_model.pth')  # 保存模型
print("训练完成！模型已保存为 simple_nn_model.pth")

# 使用 Matplotlib 可视化刚刚生成的自定义数据集
# data[:, 0] 是特征1 (x轴)，data[:, 1] 是特征2 (y轴)
# c=labels[:, 0] 根据标签值为散点着色，cmap='bwr' 是蓝白红渐变色条（Blue-White-Red）
plt.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], c=train_dataset.labels[:, 0], cmap='bwr', alpha=0.5)
plt.title('Custom Binary Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('dataset_visualization.png')
plt.show()
