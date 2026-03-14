#!/usr/bin/env python3
"""
文本编码转换工具集 - 纯 Python 实现

**功能**:
1. 序号化转换 (Label Encoding)
2. 文本/Token 哑编码转换 (One-Hot Encoding)
3. 文本词袋法转换 (Bag of Words)
4. 文本 TF-IDF 转换
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set


# ==================== 文本预处理 ====================

class TextPreprocessor:
    """文本预处理器"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        分词
        
        参数:
            text: 输入文本
        
        返回:
            分词后的列表
        """
        # 转为小写
        text = text.lower()
        # 提取单词（字母和数字）
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    @staticmethod
    def remove_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
        """
        去除停用词
        
        参数:
            tokens: 分词列表
            stopwords: 停用词集合
        
        返回:
            过滤后的分词列表
        """
        if stopwords is None:
            # 默认英文停用词
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
                'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'
            }
        
        return [token for token in tokens if token not in stopwords]


# ==================== 1. 序号化转换 (Label Encoding) ====================

class LabelEncoder:
    """
    序号化转换器 (Label Encoding)
    
    将类别标签转换为数字序号
    """
    
    def __init__(self):
        self.classes_: List[str] = []
        self.class_to_idx_: Dict[str, int] = {}
    
    def fit(self, labels: List[str]) -> 'LabelEncoder':
        """
        拟合编码器
        
        参数:
            labels: 标签列表
        
        返回:
            self
        """
        # 获取唯一类别（保持顺序）
        seen = set()
        self.classes_ = []
        for label in labels:
            if label not in seen:
                self.classes_.append(label)
                seen.add(label)
        
        # 创建类别到索引的映射
        self.class_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        
        return self
    
    def transform(self, labels: List[str]) -> List[int]:
        """
        转换标签为序号
        
        参数:
            labels: 标签列表
        
        返回:
            序号列表
        """
        return [self.class_to_idx_[label] for label in labels]
    
    def fit_transform(self, labels: List[str]) -> List[int]:
        """
        拟合并转换
        
        参数:
            labels: 标签列表
        
        返回:
            序号列表
        """
        self.fit(labels)
        return self.transform(labels)
    
    def inverse_transform(self, indices: List[int]) -> List[str]:
        """
        逆转换：序号转标签
        
        参数:
            indices: 序号列表
        
        返回:
            标签列表
        """
        idx_to_class = {idx: label for label, idx in self.class_to_idx_.items()}
        return [idx_to_class[idx] for idx in indices]


# ==================== 2. 文本/Token 哑编码转换 (One-Hot Encoding) ====================

class OneHotEncoder:
    """
    哑编码器 (One-Hot Encoding)
    
    将类别转换为独热向量
    """
    
    def __init__(self):
        self.classes_: List[str] = []
        self.class_to_idx_: Dict[str, int] = {}
    
    def fit(self, labels: List[str]) -> 'OneHotEncoder':
        """
        拟合编码器
        
        参数:
            labels: 标签列表
        
        返回:
            self
        """
        # 获取唯一类别
        seen = set()
        self.classes_ = []
        for label in labels:
            if label not in seen:
                self.classes_.append(label)
                seen.add(label)
        
        self.class_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        
        return self
    
    def transform(self, labels: List[str]) -> List[List[int]]:
        """
        转换为独热编码
        
        参数:
            labels: 标签列表
        
        返回:
            独热编码列表
        """
        n_classes = len(self.classes_)
        one_hot = []
        
        for label in labels:
            vector = [0] * n_classes
            idx = self.class_to_idx_[label]
            vector[idx] = 1
            one_hot.append(vector)
        
        return one_hot
    
    def fit_transform(self, labels: List[str]) -> List[List[int]]:
        """
        拟合并转换
        
        参数:
            labels: 标签列表
        
        返回:
            独热编码列表
        """
        self.fit(labels)
        return self.transform(labels)


class TokenOneHotEncoder:
    """
    Token 哑编码器
    
    将文本 token 转换为独热编码
    """
    
    def __init__(self):
        self.vocabulary_: Dict[str, int] = {}
        self.vocab_size_: int = 0
    
    def fit(self, texts: List[str]) -> 'TokenOneHotEncoder':
        """
        构建词汇表
        
        参数:
            texts: 文本列表
        
        返回:
            self
        """
        # 分词并构建词汇表
        vocab = set()
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            vocab.update(tokens)
        
        # 排序并创建索引
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.vocab_size_ = len(self.vocabulary_)
        
        return self
    
    def transform(self, texts: List[str]) -> List[List[int]]:
        """
        转换为独热编码（每个文本一个向量）
        
        参数:
            texts: 文本列表
        
        返回:
            独热编码列表
        """
        one_hot_vectors = []
        
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            vector = [0] * self.vocab_size_
            
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    vector[idx] = 1
            
            one_hot_vectors.append(vector)
        
        return one_hot_vectors
    
    def transform_tokens(self, tokens: List[str]) -> List[List[int]]:
        """
        将 token 列表转换为独热编码
        
        参数:
            tokens: token 列表
        
        返回:
            每个 token 的独热编码
        """
        one_hot_vectors = []
        
        for token in tokens:
            vector = [0] * self.vocab_size_
            if token in self.vocabulary_:
                idx = self.vocabulary_[token]
                vector[idx] = 1
            one_hot_vectors.append(vector)
        
        return one_hot_vectors
    
    def fit_transform(self, texts: List[str]) -> List[List[int]]:
        """
        拟合并转换
        
        参数:
            texts: 文本列表
        
        返回:
            独热编码列表
        """
        self.fit(texts)
        return self.transform(texts)


# ==================== 3. 文本词袋法转换 (Bag of Words) ====================

class CountVectorizer:
    """
    词袋法转换器 (Bag of Words)
    
    统计每个词在文档中出现的次数
    """
    
    def __init__(self, min_df: int = 1, max_df: float = 1.0, 
                 remove_stopwords: bool = False):
        """
        初始化
        
        参数:
            min_df: 最小文档频率（词至少出现在多少个文档中）
            max_df: 最大文档频率（词最多出现在多少比例的文档中）
            remove_stopwords: 是否去除停用词
        """
        self.min_df = min_df
        self.max_df = max_df
        self.remove_stopwords = remove_stopwords
        self.vocabulary_: Dict[str, int] = {}
        self.vocab_size_: int = 0
    
    def fit(self, texts: List[str]) -> 'CountVectorizer':
        """
        构建词汇表
        
        参数:
            texts: 文本列表
        
        返回:
            self
        """
        n_docs = len(texts)
        
        # 统计每个词的文档频率
        doc_freq = Counter()
        all_tokens = set()
        
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            
            if self.remove_stopwords:
                tokens = TextPreprocessor.remove_stopwords(tokens)
            
            # 去重后统计文档频率
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
            all_tokens.update(tokens)
        
        # 过滤词汇
        min_doc_count = self.min_df
        max_doc_count = self.max_df * n_docs if self.max_df <= 1.0 else self.max_df
        
        filtered_vocab = []
        for token in sorted(all_tokens):
            df = doc_freq[token]
            if min_doc_count <= df <= max_doc_count:
                filtered_vocab.append(token)
        
        # 创建词汇表
        self.vocabulary_ = {token: idx for idx, token in enumerate(filtered_vocab)}
        self.vocab_size_ = len(self.vocabulary_)
        
        return self
    
    def transform(self, texts: List[str]) -> List[List[int]]:
        """
        转换为词袋向量
        
        参数:
            texts: 文本列表
        
        返回:
            词袋向量列表
        """
        bow_vectors = []
        
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            
            if self.remove_stopwords:
                tokens = TextPreprocessor.remove_stopwords(tokens)
            
            # 统计词频
            token_counts = Counter(tokens)
            
            # 创建向量
            vector = [0] * self.vocab_size_
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    vector[idx] = count
            
            bow_vectors.append(vector)
        
        return bow_vectors
    
    def fit_transform(self, texts: List[str]) -> List[List[int]]:
        """
        拟合并转换
        
        参数:
            texts: 文本列表
        
        返回:
            词袋向量列表
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名（词汇表）
        
        返回:
            词汇表列表
        """
        idx_to_token = {idx: token for token, idx in self.vocabulary_.items()}
        return [idx_to_token[idx] for idx in range(self.vocab_size_)]


# ==================== 4. 文本 TF-IDF 转换 ====================

class TfidfTransformer:
    """
    TF-IDF 转换器
    
    将词频矩阵转换为 TF-IDF 矩阵
    """
    
    def __init__(self, use_idf: bool = True, smooth_idf: bool = True,
                 norm: str = 'l2'):
        """
        初始化
        
        参数:
            use_idf: 是否使用 IDF
            smooth_idf: 是否平滑 IDF（加 1 防止除零）
            norm: 归一化方式 ('l1', 'l2', None)
        """
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.idf_: List[float] = []
    
    def fit(self, X: List[List[int]]) -> 'TfidfTransformer':
        """
        计算 IDF
        
        参数:
            X: 词频矩阵 (n_docs, n_features)
        
        返回:
            self
        """
        n_docs = len(X)
        n_features = len(X[0]) if X else 0
        
        # 统计每个词出现在多少个文档中
        df = [0] * n_features
        for doc in X:
            for j, count in enumerate(doc):
                if count > 0:
                    df[j] += 1
        
        # 计算 IDF
        self.idf_ = []
        for j in range(n_features):
            if self.smooth_idf:
                idf = math.log((n_docs + 1) / (df[j] + 1)) + 1
            else:
                idf = math.log(n_docs / df[j]) if df[j] > 0 else 0
            self.idf_.append(idf)
        
        return self
    
    def transform(self, X: List[List[int]]) -> List[List[float]]:
        """
        转换为 TF-IDF
        
        参数:
            X: 词频矩阵
        
        返回:
            TF-IDF 矩阵
        """
        tfidf_vectors = []
        
        for doc in X:
            # 计算 TF-IDF
            tfidf = []
            for j, tf in enumerate(doc):
                if self.use_idf:
                    tfidf_val = tf * self.idf_[j]
                else:
                    tfidf_val = float(tf)
                tfidf.append(tfidf_val)
            
            # 归一化
            if self.norm == 'l1':
                norm_val = sum(abs(x) for x in tfidf)
                if norm_val > 0:
                    tfidf = [x / norm_val for x in tfidf]
            elif self.norm == 'l2':
                norm_val = math.sqrt(sum(x ** 2 for x in tfidf))
                if norm_val > 0:
                    tfidf = [x / norm_val for x in tfidf]
            
            tfidf_vectors.append(tfidf)
        
        return tfidf_vectors
    
    def fit_transform(self, X: List[List[int]]) -> List[List[float]]:
        """
        拟合并转换
        
        参数:
            X: 词频矩阵
        
        返回:
            TF-IDF 矩阵
        """
        self.fit(X)
        return self.transform(X)


class TfidfVectorizer:
    """
    TF-IDF 向量化器
    
    整合 CountVectorizer 和 TfidfTransformer
    """
    
    def __init__(self, min_df: int = 1, max_df: float = 1.0,
                 remove_stopwords: bool = False, use_idf: bool = True,
                 smooth_idf: bool = True, norm: str = 'l2'):
        """
        初始化
        
        参数:
            min_df: 最小文档频率
            max_df: 最大文档频率
            remove_stopwords: 是否去除停用词
            use_idf: 是否使用 IDF
            smooth_idf: 是否平滑 IDF
            norm: 归一化方式
        """
        self.min_df = min_df
        self.max_df = max_df
        self.remove_stopwords = remove_stopwords
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.norm = norm
        
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            remove_stopwords=remove_stopwords
        )
        self.transformer = TfidfTransformer(
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            norm=norm
        )
        self.vocabulary_: Dict[str, int] = {}
    
    def fit(self, texts: List[str]) -> 'TfidfVectorizer':
        """
        拟合
        
        参数:
            texts: 文本列表
        
        返回:
            self
        """
        self.vectorizer.fit(texts)
        self.vocabulary_ = self.vectorizer.vocabulary_.copy()
        return self
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """
        转换
        
        参数:
            texts: 文本列表
        
        返回:
            TF-IDF 矩阵
        """
        X = self.vectorizer.transform(texts)
        return self.transformer.transform(X)
    
    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        """
        拟合并转换
        
        参数:
            texts: 文本列表
        
        返回:
            TF-IDF 矩阵
        """
        self.fit(texts)
        X = self.vectorizer.transform(texts)
        return self.transformer.fit_transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名
        
        返回:
            词汇表列表
        """
        return self.vectorizer.get_feature_names()


# ==================== 工具函数 ====================

def print_matrix(matrix: List[List], header: Optional[List[str]] = None,
                 title: str = "Matrix"):
    """
    打印矩阵
    
    参数:
        matrix: 矩阵数据
        header: 列名
        title: 标题
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 打印表头
    if header:
        print(" " * 15, end="")
        for h in header[:10]:  # 最多显示 10 列
            print(f"{h[:8]:>10}", end="")
        if len(header) > 10:
            print(f" ...(+{len(header)-10})", end="")
        print()
    
    # 打印数据
    for i, row in enumerate(matrix):
        print(f"Doc {i:<10}", end="")
        for val in row[:10]:
            if isinstance(val, float):
                print(f"{val:>10.4f}", end="")
            else:
                print(f"{val:>10}", end="")
        if len(row) > 10:
            print(f" ...(+{len(row)-10})", end="")
        print()
    
    print(f"{'='*60}\n")


# ==================== 示例演示 ====================

def demo_label_encoder():
    """演示序号化转换"""
    print("\n" + "="*60)
    print("1. 序号化转换 (Label Encoding)")
    print("="*60)
    
    labels = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat']
    
    print(f"\n原始标签：{labels}")
    
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    
    print(f"编码后：{encoded}")
    print(f"类别映射：{encoder.class_to_idx_}")
    
    # 逆转换
    decoded = encoder.inverse_transform(encoded)
    print(f"逆转换：{decoded}")


def demo_one_hot_encoder():
    """演示哑编码转换"""
    print("\n" + "="*60)
    print("2. 哑编码转换 (One-Hot Encoding)")
    print("="*60)
    
    labels = ['cat', 'dog', 'bird', 'cat', 'dog']
    
    print(f"\n原始标签：{labels}")
    
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(labels)
    
    print(f"\n编码结果:")
    for i, (label, vec) in enumerate(zip(labels, encoded)):
        print(f"  {label} -> {vec}")


def demo_token_one_hot():
    """演示 Token 哑编码"""
    print("\n" + "="*60)
    print("3. Token 哑编码转换")
    print("="*60)
    
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "The cat and dog are friends"
    ]
    
    print(f"\n原始文本:")
    for i, text in enumerate(texts):
        print(f"  Doc {i}: {text}")
    
    encoder = TokenOneHotEncoder()
    encoded = encoder.fit_transform(texts)
    
    print(f"\n词汇表大小：{encoder.vocab_size_}")
    print(f"\n编码结果 (前 10 个词):")
    feature_names = list(encoder.vocabulary_.keys())[:10]
    print_matrix(encoded, header=feature_names, title="Token One-Hot Encoding")


def demo_bow():
    """演示词袋法"""
    print("\n" + "="*60)
    print("4. 词袋法转换 (Bag of Words)")
    print("="*60)
    
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "The cat and dog are friends",
        "The mat is in the park"
    ]
    
    print(f"\n原始文本:")
    for i, text in enumerate(texts):
        print(f"  Doc {i}: {text}")
    
    vectorizer = CountVectorizer(remove_stopwords=True)
    bow = vectorizer.fit_transform(texts)
    
    feature_names = vectorizer.get_feature_names()
    
    print(f"\n词汇表：{feature_names}")
    print_matrix(bow, header=feature_names, title="Bag of Words")


def demo_tfidf():
    """演示 TF-IDF"""
    print("\n" + "="*60)
    print("5. TF-IDF 转换")
    print("="*60)
    
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "The cat and dog are friends",
        "The mat is in the park"
    ]
    
    print(f"\n原始文本:")
    for i, text in enumerate(texts):
        print(f"  Doc {i}: {text}")
    
    vectorizer = TfidfVectorizer(remove_stopwords=True)
    tfidf = vectorizer.fit_transform(texts)
    
    feature_names = vectorizer.get_feature_names()
    
    print(f"\n词汇表：{feature_names}")
    print_matrix(tfidf, header=feature_names, title="TF-IDF")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("文本编码转换工具集 - 纯 Python 实现")
    print("="*60)
    
    # 运行所有演示
    demo_label_encoder()
    demo_one_hot_encoder()
    demo_token_one_hot()
    demo_bow()
    demo_tfidf()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
