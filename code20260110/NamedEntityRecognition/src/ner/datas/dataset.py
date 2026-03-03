# -*- coding: utf-8 -*-
import copy
import json
from typing import Dict
from torch.utils.data import Dataset
from .tokenizer import Tokenizer


class NerTokenClassifyDataset(Dataset):
    def __init__(self, in_file: str, tokenizer: Tokenizer, label2id: Dict[str, int], append_special_token=False):
        """
        :param in_file: 原始数据路径 (JSONL格式)
        :param tokenizer: 分词器实例
        :param label2id: 标签到ID的映射字典 (如 {"B-Disease": 1, ...})
        :param append_special_token: 是否添加 [CLS] 和 [SEP]
        """
        super().__init__()

        self.tokenizer: Tokenizer = tokenizer
        self.label2id: Dict[str, int] = label2id

        self.datas = []
        # 1. 读取原始数据文件
        with open(in_file, "r", encoding="utf-8") as reader:
            for line in reader:
                # 将每一行 JSON 字符串转换为 Python 字典
                obj = json.loads(line.strip())
                # 2. 调用分词器对原始文本进行切分和编号
                token_result = self.tokenizer(
                    obj['originalText'],
                    append_sep=append_special_token,
                    append_cls=append_special_token
                )

                # 3. 标签对齐处理
                token_offset = token_result['token_offset']  # 获取偏移量（考虑[CLS]的影响）
                # 初始化标签列表，默认全部填为 'Other' (即 'O')
                token_label_names = ['Other'] * len(token_result['token_ids'])

                # 遍历原始标注中的每一个实体
                for entity in obj['entities']:
                    label_type = entity['label_type']
                    # 计算实体在 Token 序列中的实际起始和结束位置（加上偏移量）
                    start_pos = entity['start_pos'] + token_offset  # 包含起点的索引
                    end_pos = entity['end_pos'] + token_offset  # 不包含终点的索引

                    entity_len = end_pos - start_pos  # 计算实体长度

                    # 4. 根据 BIESO 体系分配具体标签
                    if entity_len == 1:
                        # 单字实体标记为 S (Single)
                        token_label_names[start_pos] = f'S-{label_type}'
                    elif entity_len > 1:
                        # 长度大于1：开头标记为 B (Begin)
                        token_label_names[start_pos] = f'B-{label_type}'
                        # 结尾标记为 E (End)
                        token_label_names[end_pos - 1] = f'E-{label_type}'
                        # 中间部分标记为 M (Middle)
                        for i in range(start_pos + 1, end_pos - 1):
                            token_label_names[i] = f'M-{label_type}'

                # 5. 将标签名称映射为数字 ID
                token_label_ids = [self.label2id[token_label_name] for token_label_name in token_label_names]

                # 6. 构造最终的数据字典
                data = {
                    'text': token_result['text'],  # 原始文本
                    'tokens': token_result['tokens'],  # 分词后的列表
                    'token_ids': token_result['token_ids'],  # 对应的词表 ID
                    'token_masks': [1.0 for _ in token_result['token_ids']],  # 注意力掩码
                    'label_names': token_label_names,  # 转换后的标签名列表
                    'label_ids': token_label_ids  # 转换后的标签 ID 列表
                }

                # 严格校验：确保 Token 数量和标签数量完全一致
                assert len(data['token_ids']) == len(
                    data['label_ids']), f"长度不匹配: {len(data['token_ids'])} vs {len(data['label_ids'])}"

                self.datas.append(data)

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.datas)

    def __getitem__(self, item):
        """获取单个样本，使用 deepcopy 避免多线程数据污染"""
        return copy.deepcopy(self.datas[item])
