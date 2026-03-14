#!/usr/bin/env python3
"""
三层全连接 BP 神经网络 - 基于 NumPy 实现

**结构**: 输入层 → 隐藏层 → 输出层

**功能**:
- 前向传播
- 反向传播
- 参数更新
- 训练循环
- 可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


# ==================== 激活函数 ====================

class Activation:
    """激活函数类"""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid 导数"""
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU 激活函数"""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU 导数"""
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh 激活函数"""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Tanh 导数"""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 激活函数（用于输出层）"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ==================== 损失函数 ====================

class Loss:
    """损失函数类"""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方误差损失"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """MSE 导数"""
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """交叉熵损失"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """交叉熵导数"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]


# ==================== 三层 BP 神经网络 ====================

class BPNeuralNetwork:
    """
    三层全连接 BP 神经网络
    
    结构: 输入层 → 隐藏层 → 输出层
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.01, activation: str = 'sigmoid'):
        """
        初始化神经网络
        
        参数:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
            learning_rate: 学习率
            activation: 激活函数类型 ('sigmoid', 'relu', 'tanh')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 选择激活函数
        if activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        else:  # sigmoid
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative

        # 初始化权重和偏置（Xavier 初始化）
        self._initialize_weights()

        # 训练历史
        self.loss_history: List[float] = []

    def _initialize_weights(self):
        """Xavier 初始化权重"""
        # 输入层 → 隐藏层
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * \
                  np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))

        # 隐藏层 → 输出层
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * \
                  np.sqrt(2.0 / (self.hidden_size + self.output_size))
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            X: 输入数据 (batch_size, input_size)
        
        返回:
            输出结果 (batch_size, output_size)
        """
        # 输入层 → 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1  # 线性变换
        self.a1 = self.activation(self.z1)  # 激活

        # 隐藏层 → 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 线性变换
        self.a2 = Activation.sigmoid(self.z2)  # Sigmoid 输出

        # 保存输入用于反向传播
        self.X = X

        return self.a2

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        反向传播
        
        参数:
            y_true: 真实标签 (batch_size, output_size)
            y_pred: 预测输出 (batch_size, output_size)
        
        返回:
            (dW1, dW2) 权重梯度
        """
        batch_size = y_true.shape[0]

        # 输出层误差
        # 使用交叉熵损失 + Sigmoid 输出时的简化梯度
        delta2 = (y_pred - y_true) / batch_size  # (batch_size, output_size)

        # 隐藏层误差
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)

        # 计算梯度
        dW2 = np.dot(self.a1.T, delta2)  # (hidden_size, output_size)
        db2 = np.sum(delta2, axis=0, keepdims=True)  # (1, output_size)

        dW1 = np.dot(self.X.T, delta1)  # (input_size, hidden_size)
        db1 = np.sum(delta1, axis=0, keepdims=True)  # (1, hidden_size)

        # 更新权重
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return dW1, dW2

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
              batch_size: int = 32, verbose: bool = True) -> List[float]:
        """
        训练神经网络
        
        参数:
            X: 训练数据 (n_samples, input_size)
            y: 训练标签 (n_samples, output_size)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
        
        返回:
            损失历史记录
        """
        n_samples = X.shape[0]
        self.loss_history = []

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = max(1, n_samples // batch_size)

            for i in range(n_batches):
                # 获取批次数据
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                loss = Loss.cross_entropy(y_batch, y_pred)
                epoch_loss += loss

                # 反向传播
                self.backward(y_batch, y_pred)

            # 记录平均损失
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            # 打印训练信息
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return self.loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 输入数据
        
        返回:
            预测结果
        """
        return self.forward(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        参数:
            X: 输入数据
        
        返回:
            预测类别
        """
        y_pred = self.predict(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        参数:
            X: 输入数据
            y: 真实标签
        
        返回:
            准确率
        """
        y_pred_class = self.predict_class(X)
        y_true_class = np.argmax(y, axis=1)
        return np.mean(y_pred_class == y_true_class)

    def plot_loss(self, save_path: Optional[str] = None):
        """
        绘制损失曲线
        
        参数:
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"损失曲线已保存到：{save_path}")

        plt.show()

    def save(self, path: str):
        """保存模型"""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 input_size=self.input_size,
                 hidden_size=self.hidden_size,
                 output_size=self.output_size)
        print(f"模型已保存到：{path}")

    def load(self, path: str):
        """加载模型"""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.input_size = int(data['input_size'])
        self.hidden_size = int(data['hidden_size'])
        self.output_size = int(data['output_size'])
        print(f"模型已从 {path} 加载")


# ==================== 示例：XOR 问题 ====================

def create_xor_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """创建 XOR 数据集"""
    X = np.random.randn(n_samples, 2)
    y = np.zeros((n_samples, 2))

    for i in range(n_samples):
        # XOR: 同号为 0，异号为 1
        if (X[i, 0] > 0 and X[i, 1] > 0) or (X[i, 0] < 0 and X[i, 1] < 0):
            y[i] = [1, 0]  # 类别 0
        else:
            y[i] = [0, 1]  # 类别 1

    return X, y


def create_moons_data(n_samples: int = 1000, noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """创建 moons 数据集"""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # One-hot 编码
    y_onehot = np.zeros((n_samples, 2))
    y_onehot[np.arange(n_samples), y] = 1

    return X, y_onehot


def main():
    """主函数 - 演示 XOR 问题"""
    print("=" * 60)
    print("三层全连接 BP 神经网络 - XOR 问题演示")
    print("=" * 60)

    # 创建数据集
    print("\n1. 创建 XOR 数据集...")
    X, y = create_xor_data(n_samples=1000)
    print(f"   数据形状：X={X.shape}, y={y.shape}")

    # 划分训练集和测试集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   训练集：{X_train.shape[0]} 样本")
    print(f"   测试集：{X_test.shape[0]} 样本")

    # 创建神经网络
    print("\n2. 创建神经网络...")
    nn = BPNeuralNetwork(
        input_size=2,
        hidden_size=16,
        output_size=2,
        learning_rate=0.1,
        activation='sigmoid'
    )
    print(f"   结构：{nn.input_size} → {nn.hidden_size} → {nn.output_size}")
    print(f"   学习率：{nn.learning_rate}")

    # 训练
    print("\n3. 开始训练...")
    nn.train(X_train, y_train, epochs=1000, batch_size=32, verbose=True)

    # 评估
    print("\n4. 模型评估...")
    train_acc = nn.accuracy(X_train, y_train)
    test_acc = nn.accuracy(X_test, y_test)
    print(f"   训练集准确率：{train_acc * 100:.2f}%")
    print(f"   测试集准确率：{test_acc * 100:.2f}%")

    # 预测示例
    print("\n5. 预测示例...")
    test_samples = 5
    for i in np.random.permutation(len(X_test))[:test_samples]:
        pred = nn.predict(X_test[i:i + 1])[0]
        pred_class = np.argmax(pred)
        true_class = np.argmax(y_test[i])
        print(f"   样本 {i + 1}: 预测={pred_class}, 真实={true_class}")

    # 绘制损失曲线
    print("\n6. 绘制损失曲线...")
    nn.plot_loss(save_path='./loss_curve.png')

    # 保存模型
    nn.save('./bp_model.npz')

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
