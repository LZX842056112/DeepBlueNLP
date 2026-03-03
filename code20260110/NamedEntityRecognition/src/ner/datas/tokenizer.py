# -*- coding: utf-8 -*-
import torch
from typing import Union, Dict, List


# 全角转半角的工具函数，统一字符格式，减少词表冗余
def fullwidth_to_halfwidth(text):
    """
    将全角字符（如：１２３、ＡＢＣ）转换为半角字符（如：123、ABC）
    """
    # 定义全角字符表和对应的半角字符表
    fullwidth = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂“”"
    halfwidth = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".""'

    # 创建映射表并执行转换
    translation_table = str.maketrans(fullwidth, halfwidth)
    return text.translate(translation_table)


class Tokenizer(object):
    def __init__(self,
                 vocabs: Union[str, Dict[str, int]],  # 可以是词表文件路径，也可以是现成的字典
                 unk_token: str = '[UNK]',  # 未知词标记
                 pad_token: str = '[PAD]',  # 填充标记
                 cls_token: str = '[CLS]',  # 序列开始标记
                 sep_token: str = '[SEP]'  # 序列结束/分隔标记
                 ):
        super().__init__()
        # 如果传入的是路径，则调用 load_vocabs 加载文件
        if isinstance(vocabs, str):
            vocabs = self.load_vocabs(vocabs)
        self.vocabs: Dict[str, int] = vocabs
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        # 预存常用特殊标记的 ID，提高查找效率
        self.unk_token_id = self.vocabs[self.unk_token]
        self.pad_token_id = self.vocabs[self.pad_token]

    @classmethod
    def load_vocabs(cls, vocab_file: str):
        """从文件中逐行读取字符，构建 {字符: 索引} 字典"""
        vocabs = {}
        with open(vocab_file, "r", encoding="utf-8") as reader:
            for vocab in reader:
                vocabs[vocab.strip()] = len(vocabs)  # 利用当前字典长度作为 ID，实现自动增量
        return vocabs

    @classmethod
    def split_text_to_tokens(cls, text: str) -> List[str]:
        """核心切分逻辑：转小写 -> 全角转半角 -> 字符串转列表（按字符切分）"""
        text = text.lower()
        text = fullwidth_to_halfwidth(text)
        return list(text)  # 例如 "你好" -> ["你", "好"]

    def __call__(self, text: str, append_cls=False, append_sep=False):
        """使实例可以像函数一样被调用，执行完整的分词流程"""
        # 1. 划分 tokens
        token_offset = 0  # 记录实际文本相对于输出序列的起始偏移
        tokens: List[str] = self.split_text_to_tokens(text)

        # 根据需求添加 BERT 风格的特殊标记
        if append_cls:
            token_offset = 1  # 因为开头加了 [CLS]，原始文本的第一个字现在索引是 1
            tokens = [self.cls_token] + tokens
        if append_sep:
            tokens = tokens + [self.sep_token]

        # 2. 将字符转换为词表中的 ID。如果词表中没有，则指向 [UNK]
        token_ids = [self.vocabs.get(token, self.unk_token_id) for token in tokens]

        # 3. 结果返回，包含原始文本、token列表、ID列表和偏移量
        return {
            'text': text,
            "tokens": tokens,
            "token_ids": token_ids,
            "token_offset": token_offset
        }
