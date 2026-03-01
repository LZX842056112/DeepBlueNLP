# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:21
Create User : 19410
Desc : 定义分词器相关操作
"""
# 引入 dataclass，用于快速创建只有属性、没有复杂逻辑的数据类
from dataclasses import dataclass
from typing import List, Optional, Dict

# 从 Hugging Face 导入 BERT 的专用分词器
from transformers import BertTokenizer

# 假设这是你自己在 utils.py 里写的一个简单的分词函数（比如按字切分，或者用 jieba 切词）
from .utils import split_text_to_tokens


# 1. 定义标准输出结构
@dataclass
class TokenizerOutput:
    """
    数据类：规定了所有分词器处理完毕后，必须返回统一的数据格式。
    这就像是一份“契约”，后面的 Dataset 和 DataLoader 只要看到这个对象，就知道去哪里拿数据。
    """
    text: str  # 原始文本，例如 "我爱中国"
    tokens: List[str]  # 切分后的字符列表，例如 ["我", "爱", "中", "国"]
    token_ids: List[int]  # 映射后的数字 ID 列表，例如 [10, 25, 48, 99]
    label: Optional[str] = None  # 原始文本标签（可选，预测时没有标签），例如 "positive"
    label_id: Optional[int] = 0  # 映射后的标签数字 ID，例如 1


# 2. 定义分词器基类 (接口)
class TokenizerBase:
    """
    分词器抽象基类：它不干具体的活，只是定义了“一个合格的分词器必须具备哪些功能”。
    任何继承它的子类，都必须实现下面这些方法。
    """

    # __call__ 魔法方法，使得你可以像调用函数一样调用对象实例，例如: tokenizer("文本")
    def __call__(self, text: str, label: Optional[str] = None):
        raise NotImplementedError("子类实现")

    # @property 装饰器把方法变成属性调用，例如用 tokenizer.vocab_size 而不是 tokenizer.vocab_size()
    @property
    def vocab_size(self):
        raise NotImplementedError("子类实现")

    @property
    def num_classes(self):
        raise NotImplementedError("子类实现")

    @property
    def pad_token_id(self):
        raise NotImplementedError("子类实现")

    @property
    def unk_token_id(self):
        raise NotImplementedError("子类实现")

    @property
    def token2ids(self):
        raise NotImplementedError("子类实现")

    @property
    def label2ids(self):
        raise NotImplementedError("子类实现")


# 3. 自定义分词器实现 (适合从零训练模型)
class Tokenizer(TokenizerBase):
    """
    基础分词器：依赖你自己传入的词表字典 (token2ids) 进行文本到 ID 的转换。
    """

    def __init__(self,
                 token2ids: Dict[str, int],  # token到id的映射字典
                 label2ids: Dict[str, int],  # 标签名称到id的映射字典
                 unk_token='<UNK>',  # Unknown token (遇到词表里没有的字时的占位符)
                 pad_token='<PAD>'  # Padding token (为了将不同长度的句子补齐到相同长度)
                 ):
        super().__init__()
        self._token2ids = token2ids
        # 初始化时就把 UNK 和 PAD 的对应数字 ID 找出来存好，提高后续运行速度
        self._unk_token_id = self._token2ids[unk_token]
        self._pad_token_id = self._token2ids[pad_token]
        self._label2ids = label2ids

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 1. 分词 (调用外部定义的函数)
        tokens = split_text_to_tokens(text)

        # 2. 将每个token转换为token id
        # 核心逻辑：遍历 tokens，如果在词表字典里找不到对应的字，就用 unk_token_id 兜底
        token_ids = [self._token2ids.get(token, self.unk_token_id) for token in tokens]

        # 3. 标签转换
        label_id = None
        if label is not None:
            label = str(label)
            label_id = self._label2ids[label]

        # 把所有结果打包成前面定义好的标准类返回
        return TokenizerOutput(text, tokens, token_ids, label, label_id)

    # 下面是对基类中 @property 的具体实现，直接返回实例变量
    @property
    def vocab_size(self):
        return len(self._token2ids)

    @property
    def num_classes(self):
        return len(self._label2ids)

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def unk_token_id(self):
        return self._unk_token_id

    @property
    def token2ids(self):
        return self._token2ids

    @property
    def label2ids(self):
        return self._label2ids


# 4. BERT 代理分词器 (适合微调预训练模型)
class ProxyBertTokenizer(TokenizerBase):
    """
    代理/适配器模式：把 Hugging Face 的 BertTokenizer 包装一层，
    让它伪装成我们自己定义的 TokenizerBase 接口标准。
    """

    def __init__(self,
                 bert_tokenizer_file: str,  # 比如 "bert-base-chinese" 的路径
                 label2ids: Dict[str, int],  # 标签名称到id的映射字典
                 ):
        super().__init__()

        # 实例化真正的、底层的 HF BertTokenizer
        self.proxy: BertTokenizer = BertTokenizer.from_pretrained(bert_tokenizer_file)
        self._label2ids = label2ids

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 1. 分词 (使用 BERT 自带的 tokenize 方法)
        tokens = self.proxy.tokenize(text)

        # 2. 将每个token转换为token id
        # ⚠️ 这里有个隐患提示：self.proxy(text)['input_ids'] 默认会在句首和句尾加上特殊字符 [CLS] 和 [SEP]
        # 但是上面的 tokens = self.proxy.tokenize(text) 默认不加。
        # 这会导致 tokens 和 token_ids 的长度不一致！在后续训练对齐时可能会报错。
        token_ids = self.proxy(text)['input_ids']

        # 3. 标签转换
        label_id = None
        if label is not None:
            label = str(label)
            # 注意：如果传入的 label 不在字典里会报 KeyError
            label_id = self.label2ids[label]

        return TokenizerOutput(text, tokens, token_ids, label, label_id)

    # 代理转发：自己没有这些数据，直接去问里面的 self.proxy 要
    @property
    def vocab_size(self):
        return self.proxy.vocab_size

    @property
    def num_classes(self):
        return len(self._label2ids)

    @property
    def pad_token_id(self):
        return self.proxy.pad_token_id

    @property
    def unk_token_id(self):
        return self.proxy.unk_token_id

    @property
    def token2ids(self):
        # 复制一份真正的词表返回
        return dict(self.proxy.vocab.copy())

    @property
    def label2ids(self):
        return self._label2ids
