# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 15:45
Create User : 19410
Desc : 分词器构造 --> 直接使用字符作为token
"""
from typing import Union, Dict, List


def fullwidth_to_halfwidth(text):
    """
    将全角字符转换为半角字符
    :param text:
    :return:
    """
    fullwidth = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂“”"
    halfwidth = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".""'

    translation_table = str.maketrans(fullwidth, halfwidth)
    return text.translate(translation_table)


class Tokenizer(object):
    def __init__(self,
                 vocabs: Union[str, Dict[str, int]],
                 unk_token: str = '[UNK]',
                 pad_token: str = '[PAD]',
                 cls_token: str = '[CLS]',
                 sep_token: str = '[SEP]'
                 ):
        super().__init__()
        if isinstance(vocabs, str):
            vocabs = self.load_vocabs(vocabs)
        self.vocabs: Dict[str, int] = vocabs
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.unk_token_id = self.vocabs[self.unk_token]
        self.pad_token_id = self.vocabs[self.pad_token]

    @classmethod
    def load_vocabs(cls, vocab_file: str):
        vocabs = {}
        with open(vocab_file, "r", encoding="utf-8") as reader:
            for vocab in reader:
                vocabs[vocab.strip()] = len(vocabs)
        return vocabs

    @property
    def vocab_size(self):
        return len(self.vocabs)

    @classmethod
    def split_text_to_tokens(cls, text: str) -> List[str]:
        text = text.lower()
        text = fullwidth_to_halfwidth(text)
        return list(text)

    @classmethod
    def _split_max_length(
            cls,
            tokens: List[str],
            label_names: List[str] = None,
            no_entity_label_name: str = 'Other',
            max_length=None
    ):
        has_labels = label_names is not None and len(label_names) > 0
        if has_labels:
            assert len(tokens) == len(label_names), f"token和label的长度分别为:{len(tokens)} - {len(label_names)}"

        if max_length is None:
            yield {
                "tokens": tokens,
                "label_names": label_names
            }
            return

        """
        定义好长文本划分的规则
        1. 截取的位置最好选择标点符号位置: --> ，。
        2. 截断的时候不能够将实体截断: --> 当label_names非空/非None的时候
        3. 相邻的两个截取片段中最好一段重叠的区域
        """
        total_len = len(tokens)
        start_pos = 0  # 包含
        while start_pos < total_len:
            end_pos = start_pos + max_length  # 不包含
            # 1. 截取的位置最好选择标点符号位置: --> ，。
            # 2. 截断的时候不能够将实体截断: --> 当label_names非空/非None的时候
            k = start_pos + 1
            while True:
                if k - start_pos >= max_length:
                    break
                if k >= total_len:
                    end_pos = k + 1
                    break
                if tokens[k] in (',', '.', '，', '。'):
                    if has_labels:
                        if label_names[k] == no_entity_label_name:
                            end_pos = k + 1
                    else:
                        end_pos = k + 1
                k += 1

            # 2. 截断的时候不能够将实体截断: --> 当label_names非空/非None的时候
            if has_labels and end_pos < total_len - 1:
                k = min(total_len - 1, end_pos - 1)
                while k > start_pos:
                    if label_names[k] == no_entity_label_name:
                        break
                    k -= 1
                end_pos = k + 1

            # 3. 返回当前的结果
            yield {
                "tokens": tokens[start_pos:end_pos],
                "label_names": label_names[start_pos:end_pos] if has_labels else None
            }

            # 4. 更新下一次的开始位置
            if end_pos >= total_len - 1:
                break
            k = end_pos - 2
            while k > start_pos:
                if tokens[k] in (',', '.', '，', '。'):
                    end_pos = k + 1
                    break
                k -= 1
            start_pos = end_pos

    def __call__(
            self,
            text: str, append_cls=False, append_sep=False,
            token_label_names: List[str] = None,
            no_entity_label_name: str = 'Other',
            max_length=None,
            return_pt=True
    ):
        if max_length is not None:
            if append_cls:
                max_length = max_length - 1
            if append_sep:
                max_length = max_length - 1
        has_labels = token_label_names is not None and len(token_label_names) > 0
        # 1. 划分tokens
        tokens: List[str] = self.split_text_to_tokens(text)

        # 分割代码
        for item in self._split_max_length(tokens, token_label_names, no_entity_label_name, max_length):
            sub_tokens = item['tokens']
            sub_label_names = item['label_names']
            if append_cls:
                sub_tokens = [self.cls_token] + sub_tokens
                if has_labels:
                    sub_label_names = [no_entity_label_name] + sub_label_names
            if append_sep:
                sub_tokens = sub_tokens + [self.sep_token]
                if has_labels:
                    sub_label_names = sub_label_names + [no_entity_label_name]

            # 2. 将token转换为id
            sub_tokens_ids = [self.vocabs.get(token, self.unk_token_id) for token in sub_tokens]
            token_masks = [1.0 for _ in sub_tokens_ids]
            if return_pt:
                import torch
                sub_tokens_ids = torch.tensor(sub_tokens_ids)
                token_masks = torch.tensor(token_masks)

            # 3. 结果返回
            yield {
                'text': text,
                "tokens": sub_tokens,
                "token_ids": sub_tokens_ids,
                "token_masks": token_masks,
                "label_names": sub_label_names
            }
