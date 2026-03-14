# 文本编码转换工具集

---

## 📋 

基于纯 Python 原生代码实现的文本编码转换工具集，不依赖任何第三方库。

### 实现功能

| 功能 | 类名 | 说明 |
|-----|------|------|
| **序号化转换** | `LabelEncoder` | 类别标签 → 数字序号 |
| **哑编码转换** | `OneHotEncoder` | 类别 → 独热向量 |
| **Token 哑编码** | `TokenOneHotEncoder` | 文本 token → 独热编码 |
| **词袋法** | `CountVectorizer` | 统计词频 |
| **TF-IDF** | `TfidfVectorizer` | 词频 -逆文档频率 |

---

## 📁 

```
文本编码转换/
├── text_encoding.py    # 核心代码
├── README.md           # 本文档
```

---

## 🔧 使用方法

### 1. 序号化转换 (Label Encoding)

```python
from text_encoding import LabelEncoder

labels = ['cat', 'dog', 'bird', 'cat', 'dog']

encoder = LabelEncoder()
encoded = encoder.fit_transform(labels)

print(encoded)  # [0, 1, 2, 0, 1]
print(encoder.class_to_idx_)  # {'cat': 0, 'dog': 1, 'bird': 2}

# 逆转换
decoded = encoder.inverse_transform(encoded)
print(decoded)  # ['cat', 'dog', 'bird', 'cat', 'dog']
```

---

### 2. 哑编码转换 (One-Hot Encoding)

```python
from text_encoding import OneHotEncoder

labels = ['cat', 'dog', 'bird', 'cat']

encoder = OneHotEncoder()
encoded = encoder.fit_transform(labels)

print(encoded)
# [
#   [1, 0, 0],  # cat
#   [0, 1, 0],  # dog
#   [0, 0, 1],  # bird
#   [1, 0, 0]   # cat
# ]
```

---

### 3. Token 哑编码

```python
from text_encoding import TokenOneHotEncoder

texts = [
    "The cat sat on the mat",
    "The dog ran in the park"
]

encoder = TokenOneHotEncoder()
encoded = encoder.fit_transform(texts)

print(f"词汇表大小：{encoder.vocab_size_}")
print(f"编码维度：{len(encoded[0])}")
```

---

### 4. 词袋法 (Bag of Words)

```python
from text_encoding import CountVectorizer

texts = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "The cat and dog are friends"
]

vectorizer = CountVectorizer(remove_stopwords=True)
bow = vectorizer.fit_transform(texts)

print(f"词汇表：{vectorizer.get_feature_names()}")
print(f"词袋矩阵：{bow}")
```

**输出示例**:
```
词汇表：['and', 'are', 'cat', 'dog', 'friends', 'mat', 'sat']
词袋矩阵：
  Doc 0: [0, 0, 1, 0, 0, 1, 1]
  Doc 1: [0, 0, 0, 1, 0, 0, 0]
  Doc 2: [1, 1, 1, 1, 1, 0, 0]
```

---

### 5. TF-IDF 转换

```python
from text_encoding import TfidfVectorizer

texts = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "The cat and dog are friends"
]

vectorizer = TfidfVectorizer(remove_stopwords=True)
tfidf = vectorizer.fit_transform(texts)

print(f"词汇表：{vectorizer.get_feature_names()}")
print(f"TF-IDF 矩阵：{tfidf}")
```

---

## 📊 算法原理

### 1. 序号化转换

```
类别 → 索引映射
cat → 0
dog → 1
bird → 2
```

### 2. 哑编码

```
类别 → 独热向量
cat → [1, 0, 0]
dog → [0, 1, 0]
bird → [0, 0, 1]
```

### 3. 词袋法

```
统计每个词在文档中的出现次数

Doc: "The cat sat on the mat"
BoW: {cat: 1, sat: 1, mat: 1}
```

### 4. TF-IDF

```
TF (词频) = 词在文档中出现的次数 / 文档总词数

IDF (逆文档频率) = log(文档总数 / 包含该词的文档数)

TF-IDF = TF × IDF
```

---

## 🎯 核心类说明

### LabelEncoder

| 方法 | 说明 |
|-----|------|
| `fit(labels)` | 拟合编码器 |
| `transform(labels)` | 转换为序号 |
| `fit_transform(labels)` | 拟合并转换 |
| `inverse_transform(indices)` | 逆转换 |

### OneHotEncoder

| 方法 | 说明 |
|-----|------|
| `fit(labels)` | 拟合编码器 |
| `transform(labels)` | 转换为独热编码 |
| `fit_transform(labels)` | 拟合并转换 |

### CountVectorizer

| 方法 | 说明 |
|-----|------|
| `fit(texts)` | 构建词汇表 |
| `transform(texts)` | 转换为词袋向量 |
| `fit_transform(texts)` | 拟合并转换 |
| `get_feature_names()` | 获取词汇表 |

### TfidfVectorizer

| 方法 | 说明 |
|-----|------|
| `fit(texts)` | 拟合 |
| `transform(texts)` | 转换为 TF-IDF |
| `fit_transform(texts)` | 拟合并转换 |
| `get_feature_names()` | 获取词汇表 |

---

## ⚙️ 参数说明

### CountVectorizer

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `min_df` | int | 1 | 最小文档频率 |
| `max_df` | float | 1.0 | 最大文档频率 |
| `remove_stopwords` | bool | False | 是否去除停用词 |

### TfidfVectorizer

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `min_df` | int | 1 | 最小文档频率 |
| `max_df` | float | 1.0 | 最大文档频率 |
| `remove_stopwords` | bool | False | 是否去除停用词 |
| `use_idf` | bool | True | 是否使用 IDF |
| `smooth_idf` | bool | True | 是否平滑 IDF |
| `norm` | str | 'l2' | 归一化方式 |

---

## 🧪 运行示例

```bash
# 运行演示
python text_encoding.py
```

**输出**:
```
============================================================
文本编码转换工具集 - 纯 Python 实现
============================================================

============================================================
1. 序号化转换 (Label Encoding)
============================================================

原始标签：['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat']
编码后：[0, 1, 2, 0, 1, 2, 0]
类别映射：{'cat': 0, 'dog': 1, 'bird': 2}
逆转换：['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat']

============================================================
2. 哑编码转换 (One-Hot Encoding)
============================================================
...
```

---
