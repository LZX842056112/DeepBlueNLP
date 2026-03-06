# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/17 16:32
Create User : 19410
Desc : 推理预测
"""
import numpy as np
import torch
from transformers import BertTokenizer, BertForTokenClassification, DataCollatorForTokenClassification

from ner.datas.tokenizer import Tokenizer
from ner.datas.utils import parse_record
from ner.utils import trans_entity2tuple, extract_entities


def process_example(text, tokenizer, max_length, label2id):
    _iter = parse_record(
        text, None, tokenizer,
        True, max_length, label2id,
        return_pt=False
    )
    return list(_iter)


@torch.no_grad()
def t0():
    bert_path = "./output/travel_query/modules/final-best-model"

    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    tokenizer: Tokenizer = Tokenizer(
        vocabs=bert_tokenizer.vocab,
        unk_token=bert_tokenizer.unk_token,
        pad_token=bert_tokenizer.pad_token,
        cls_token=bert_tokenizer.cls_token,
        sep_token=bert_tokenizer.sep_token
    )
    bert_model = BertForTokenClassification.from_pretrained(
        bert_path,
        weights_only=False  # 当前参数在某些版本中需要加入，某些版本不需要 -- 具体根据异常来决定
    )
    bert_model.eval().cpu()
    label2id = bert_model.config.label2id
    max_length = bert_model.config.max_position_embeddings

    # 批次数据合并的方法
    data_collator = DataCollatorForTokenClassification(tokenizer=bert_tokenizer)

    text = "帮我查询一下明天去北京的飞机最便宜的是什么时候出发的"
    text = "帮我查询一下明天从北京来的飞机最便宜的是什么时候出发的"
    features = process_example(text, tokenizer, max_length, label2id)
    new_features = []
    for feature in features:
        new_features.append({
            "input_ids": feature['token_ids'],
            "attention_mask": feature['token_masks']
        })
    batch = data_collator.torch_call(new_features)
    print(batch)

    # 调用模型
    result = bert_model(**batch)
    logits = result.logits  # [bs,?,c]

    token_masks = batch['attention_mask']
    pred_class_ids = torch.argmax(logits, dim=-1)  # [bs, ?]
    pred_class_ids = (pred_class_ids * token_masks.numpy().astype(np.int32)
                      + (1 - token_masks.numpy().astype(np.int32)) * (- 100))  # 所有填充位置的预测类别重置为-100
    print(pred_class_ids.shape)
    print(pred_class_ids)

    pred_entities = trans_entity2tuple(
        label_ids=pred_class_ids.numpy(),
        label_id2names=bert_model.config.id2label
    )
    print(pred_entities)

    token_lengths = torch.sum(token_masks, dim=1).numpy()
    final_entities = extract_entities(
        text, pred_entities, sub_text_lengths=token_lengths, append_special_token=True
    )
    print(final_entities)


if __name__ == '__main__':
    t0()
