# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/6 10:18
Create User : 19410
Desc : xxx
"""

# 由于transformers框架对应的网站 https://huggingface.co 需要外网访问，所以这里弄一个国内使用的网站
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'


def tt01():
    from transformers import DebertaTokenizer, DebertaModel

    model_id = 'microsoft/deberta-base'
    tokenizer = DebertaTokenizer.from_pretrained(model_id)
    model = DebertaModel.from_pretrained(model_id)
    print(tokenizer)
    print(model)

    text = "active unaffable my name is 小小，上海有什么龘鱻驫"
    # 针对给定文本进行分词、token id转换的操作
    token_encoder = tokenizer([text, "我是小明"], return_tensors="pt", padding=True)
    print(token_encoder)

    # 调用模型
    bert_output = model(
        input_ids=token_encoder['input_ids'],
        attention_mask=token_encoder['attention_mask'],
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True
    )
    print(type(bert_output))


if __name__ == '__main__':
    tt01()
