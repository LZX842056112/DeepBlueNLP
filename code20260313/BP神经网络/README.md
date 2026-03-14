# BP 神经网络实现文档

---

## 📋 

基于 NumPy 实现的三层全连接 BP（反向传播）神经网络。

### 网络结构

```
输入层 (input_size) → 隐藏层 (hidden_size) → 输出层 (output_size)
```

---

## 🎯 核心功能

| 功能 | 说明 |
|-----|------|
| **前向传播** | 输入 → 隐藏层 → 输出层 |
| **反向传播** | 计算梯度并更新权重 |
| **激活函数** | Sigmoid / ReLU / Tanh |
| **损失函数** | MSE / 交叉熵 |
| **批量训练** | 支持 mini-batch 训练 |
| **模型保存** | 支持保存和加载模型 |

---

## 📁 文件结构

```
BP 神经网络/
├── bp_neural_network.py    # 核心代码
├── README.md               # 本文档
```

---

## 🔧 使用方法

### 1. 基本使用

```python
from bp_neural_network import BPNeuralNetwork

# 创建网络
nn = BPNeuralNetwork(
    input_size=2,      # 输入特征数
    hidden_size=16,    # 隐藏层神经元数
    output_size=2,     # 输出类别数
    learning_rate=0.1, # 学习率
    activation='sigmoid'  # 激活函数
)

# 训练
nn.train(X_train, y_train, epochs=1000, batch_size=32)

# 预测
y_pred = nn.predict(X_test)

# 评估
acc = nn.accuracy(X_test, y_test)
```

### 2. 运行示例

```bash
# 运行 XOR 问题演示
python bp_neural_network.py
```

---

## 📊 网络结构详解

### 前向传播

```
z1 = X · W1 + b1     # 隐藏层线性变换
a1 = activation(z1)  # 隐藏层激活

z2 = a1 · W2 + b2    # 输出层线性变换
a2 = sigmoid(z2)     # 输出层激活（分类）
```

### 反向传播

```
# 输出层误差
delta2 = (a2 - y) / batch_size

# 隐藏层误差
delta1 = (delta2 · W2.T) * activation'(z1)

# 梯度计算
dW2 = a1.T · delta2
dW1 = X.T · delta1

# 参数更新
W2 -= lr * dW2
W1 -= lr * dW1
```

---

## 🎯 激活函数对比

| 激活函数 | 公式 | 优点 | 缺点 |
|---------|------|------|------|
| **Sigmoid** | 1/(1+e^-x) | 输出平滑 | 梯度消失 |
| **ReLU** | max(0,x) | 计算快 | Dead ReLU |
| **Tanh** | (e^x-e^-x)/(e^x+e^-x) | 零中心化 | 梯度消失 |

---

## 📈 训练参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| **learning_rate** | 0.01 | 学习率 |
| **epochs** | 1000 | 训练轮数 |
| **batch_size** | 32 | 批次大小 |
| **activation** | sigmoid | 激活函数 |

---
