# LLM 位置编码

---

## 📋 目录

1. [位置编码概述](#1-位置编码概述)
2. [绝对位置编码](#2-绝对位置编码)
3. [相对位置编码](#3-相对位置编码)
4. [RoPE 旋转位置编码](#4-RoPE旋转位置编码)
5. [ALiBi 位置编码](#5-ALiBi位置编码)
6. [位置编码对比与选择](#6-位置编码对比与选择)
7. [实战练习](#7-实战练习)

---

## 1. 位置编码概述

### 1.1 为什么需要位置编码？

Transformer 模型本身是**排列不变**的，即：

```
输入："I love AI" 和 "AI love I"
在 Transformer 眼中是相同的（如果没有位置编码）
```

**位置编码的作用**：
- 为模型提供词序信息
- 区分相同词不同位置的语义
- 支持序列长度外推

### 1.2 位置编码的要求

| 要求 | 说明 |
|-----|------|
| **唯一性** | 每个位置有唯一编码 |
| **相对性** | 能表达位置间相对关系 |
| **外推性** | 支持比训练时更长的序列 |
| **可学习性** | 可以是固定的或可学习的 |

### 1.3 位置编码分类

```
位置编码
├── 绝对位置编码
│   ├── 可学习位置编码 (BERT, GPT-2)
│   └── 正弦位置编码 (Transformer)
├── 相对位置编码
│   ├── T5 相对位置编码
│   └── Transformer-XL 相对位置
├── 旋转位置编码 (RoPE)
└── 其他变体
    ├── ALiBi
    ├── Kerple
    └── NoPE
```

---

## 2. 绝对位置编码

### 2.1 可学习位置编码

**原理**：为每个位置学习一个嵌入向量

**公式**：

$$E_{pos} \in \mathbb{R}^{d_{model}}, pos \in [0, max\_len)$$

$$h_{pos} = TokenEmbedding + E_{pos}$$

```python
# 位置嵌入矩阵
PositionEmbedding = [p_0, p_1, p_2, ..., p_{max_len}]
# 每个 p_i 是可学习的向量
```

**优点**：
- 简单直接
- 模型可以学习最适合的位置表示

**缺点**：
- 无法外推到更长的序列
- 需要预先定义最大长度

### 2.2 正弦位置编码

**原理**：使用不同频率的正弦和余弦函数

**公式**：

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**其中**：
- $pos$: 位置索引 $(0, 1, 2, ...)$
- $i$: 维度索引 $(0, 1, 2, ..., d_{model}/2-1)$
- $d_{model}$: 嵌入维度

**向量形式**：

$$PE_{pos} = \begin{bmatrix}
\sin(pos \cdot \omega_0) \\
\cos(pos \cdot \omega_0) \\
\sin(pos \cdot \omega_1) \\
\cos(pos \cdot \omega_1) \\
\vdots
\end{bmatrix}, \quad \omega_k = 10000^{-2k/d_{model}}$$

**特点**：
- 固定不变，无需学习
- 可以外推到更长序列
- 不同维度捕捉不同尺度

---

## 3. 相对位置编码

### 3.1 为什么需要相对位置？

**绝对位置的问题**：
- "I love AI" 和 "You love AI" 中 "love" 位置不同但语义相似
- 相对位置更能捕捉语言中的相对关系

### 3.2 T5 相对位置编码

**核心思想**：在 Attention 中添加相对位置偏置

```python
# Attention 分数 + 相对位置偏置
Attention(Q, K, V) = softmax(QK^T / sqrt(d) + bias) V

# bias 根据相对位置 (i-j) 确定
```

**特点**：
- 桶化相对位置（减少参数量）
- 支持外推
- 计算效率高

---

## 4. RoPE旋转位置编码

### 4.1 核心原理

**Rotary Position Embedding (RoPE)** 通过旋转矩阵编码位置信息。

**核心公式**：

$$f_q(x_m, m) = R_{\Theta, m}^d \cdot W_q \cdot x_m$$

$$f_k(x_n, n) = R_{\Theta, n}^d \cdot W_k \cdot x_n$$

**其中**：
- $x_m$: 位置 $m$ 的输入向量
- $W_q, W_k$: Query 和 Key 的投影矩阵
- $R_{\Theta, m}^d$: 旋转矩阵

### 4.2 旋转矩阵

$$R_{\Theta, m}^d = \begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

**频率定义**：

$$\Theta = \{\theta_i = base^{-2i/d} \mid i = 0, 1, ..., d/2-1\}$$

**通常**：$base = 10000$

### 4.3 二维旋转

**对于每对维度** $[x_{2i}, x_{2i+1}]$：

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

**展开**：

$$x'_{2i} = x_{2i} \cdot \cos(m\theta_i) - x_{2i+1} \cdot \sin(m\theta_i)$$

$$x'_{2i+1} = x_{2i} \cdot \sin(m\theta_i) + x_{2i+1} \cdot \cos(m\theta_i)$$

### 4.4 相对位置编码性质

**Query 和 Key 的点积**：

$$f_q(x_m, m)^T \cdot f_k(x_n, n) = x_m^T W_q^T R_{\Theta, n-m}^d W_k x_n$$

**关键**：点积只依赖于相对位置 $(m-n)$

### 4.5 实现步骤

1. 将向量分成 d/2 对
2. 每对应用不同频率的旋转
3. 旋转角度与位置成正比

---

## 5. ALiBi位置编码

### 5.1 核心思想

**Attention with Linear Biases (ALiBi)** 不使用位置嵌入，而是在 Attention 分数中添加与距离成比例的偏置。

**标准 Attention**：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**ALiBi Attention**：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + b\right) V$$

**其中偏置矩阵** $b$：

$$b_{i,j} = \begin{cases}
m \cdot (j - i) & \text{if } j \leq i \text{ (causal)} \\
-\infty & \text{if } j > i \text{ (masked)}
\end{cases}$$

**其中**：
- $i$: Query 位置
- $j$: Key 位置
- $m$: 斜率（负数，使远处位置注意力减小）

### 5.2 斜率设置

**每个注意力头有不同的斜率**：

$$m_j = \frac{1}{2^{\lceil j \cdot \log_2(n_{heads}) / n_{heads} \rceil}}, \quad j \in [0, n_{heads})$$

**示例**（8 个注意力头）：

| Head | 斜率 | $2^k$ |
|-----|------|-------|
| 0-1 | 1/2 | 2 |
| 2-3 | 1/4 | 4 |
| 4-7 | 1/8 | 8 |

### 5.3 偏置效果

**对于位置 $i$ 的 Query 和位置 $j$ 的 Key**：

$$bias(i, j) = m \cdot (j - i)$$

**特点**：
- $j = i$: 偏置为 0
- $j < i$: 偏置为负（过去的信息）
- $j > i$: 掩码（未来的信息不可见）

### 5.4 特点

- 无需位置嵌入
- 天然支持外推
- 计算简单
- 不同头关注不同距离

---

## 6. 位置编码对比与选择

### 6.1 对比表格

| 编码方式 | 可学习 | 外推性 | 计算成本 | 适用场景 |
|---------|--------|--------|---------|---------|
| **可学习** | ✅ | ❌ | 低 | 短序列 |
| **正弦** | ❌ | ✅ | 低 | 通用 |
| **T5 相对** | ✅ | ✅ | 中 | 长文本 |
| **RoPE** | ❌ | ✅ | 中 | LLaMA 等 |
| **ALiBi** | ❌ | ✅ | 低 | 长文本外推 |

### 6.2 选择建议

| 场景 | 推荐编码 |
|-----|---------|
| **短文本分类** | 可学习位置编码 |
| **通用 Transformer** | 正弦位置编码 |
| **长文本理解** | RoPE / ALiBi |
| **需要外推** | RoPE / ALiBi / T5 |
| **多语言模型** | RoPE |

---

## 7. 实战练习

### 7.1 实现正弦位置编码

见代码文件：`code/01_position_embedding_basic.py`

### 7.2 实现 RoPE

见代码文件：`code/02_rope.py`

### 7.3 实现 ALiBi

见代码文件：`code/03_alibi.py`

---

## 📁 代码文件

```
code/
├── 01_position_embedding_basic.py  # 基础位置编码
├── 02_rope.py                      # RoPE
├── 03_alibi.py                     # ALiBi
```

---

---

---


