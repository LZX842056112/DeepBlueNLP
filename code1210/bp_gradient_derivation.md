# BP算法梯度计算详细推导

## 网络结构
- 输入层：x1=5.0, x2=10.0
- 隐藏层：h1, h2, h3（3个神经元）
- 输出层：o1, o2（2个神经元）
- 激活函数：sigmoid函数 σ(z) = 1/(1+e^(-z))

## 前向传播公式

### 隐藏层计算：
```
net_h1 = w1×x1 + w2×x2 + b1
out_h1 = σ(net_h1)

net_h2 = w3×x1 + w4×x2 + b1  
out_h2 = σ(net_h2)

net_h3 = w5×x1 + w6×x2 + b1
out_h3 = σ(net_h3)
```

### 输出层计算：
```
net_o1 = out_h1×w7 + out_h2×w9 + out_h3×w11 + b2
out_o1 = σ(net_o1)

net_o2 = out_h1×w8 + out_h2×w10 + out_h3×w12 + b2
out_o2 = σ(net_o2)
```

## 损失函数
```
L = 0.5×(out_o1 - y1)² + 0.5×(out_o2 - y2)²
```

## 反向传播梯度计算

### 1. 输出层梯度
```
∂L/∂net_o1 = ∂L/∂out_o1 × ∂out_o1/∂net_o1
            = (out_o1 - y1) × out_o1×(1 - out_o1)
            
∂L/∂net_o2 = ∂L/∂out_o2 × ∂out_o2/∂net_o2  
            = (out_o2 - y2) × out_o2×(1 - out_o2)
```

### 2. 隐藏层到输出层权重梯度
```
∂L/∂w7 = ∂L/∂net_o1 × ∂net_o1/∂w7 = ∂L/∂net_o1 × out_h1
∂L/∂w8 = ∂L/∂net_o2 × ∂net_o2/∂w8 = ∂L/∂net_o2 × out_h1

∂L/∂w9 = ∂L/∂net_o1 × ∂net_o1/∂w9 = ∂L/∂net_o1 × out_h2  
∂L/∂w10 = ∂L/∂net_o2 × ∂net_o2/∂w10 = ∂L/∂net_o2 × out_h2

∂L/∂w11 = ∂L/∂net_o1 × ∂net_o1/∂w11 = ∂L/∂net_o1 × out_h3
∂L/∂w12 = ∂L/∂net_o2 × ∂net_o2/∂w12 = ∂L/∂net_o2 × out_h3
```

### 3. 输入层到隐藏层权重梯度

对于w1（x1→h1）：
```
∂L/∂w1 = ∂L/∂net_h1 × ∂net_h1/∂w1

其中：
∂net_h1/∂w1 = x1

∂L/∂net_h1 = [∂L/∂net_o1 × ∂net_o1/∂out_h1 + ∂L/∂net_o2 × ∂net_o2/∂out_h1] × ∂out_h1/∂net_h1

而：
∂net_o1/∂out_h1 = w7
∂net_o2/∂out_h1 = w8
∂out_h1/∂net_h1 = out_h1 × (1 - out_h1)

因此：
∂L/∂w1 = [∂L/∂net_o1 × w7 + ∂L/∂net_o2 × w8] × out_h1 × (1 - out_h1) × x1
```

同理可得其他权重的梯度计算公式。

## 参数更新
```
w_i = w_i - α × ∂L/∂w_i
```
其中α为学习率。

## 关键数学概念

1. **链式法则**：复合函数求导的基本工具
2. **偏导数**：多元函数对单个变量的导数
3. **梯度**：函数在某点处变化最快的方向
4. **sigmoid函数导数**：σ'(z) = σ(z)×(1-σ(z))