# -*- coding: utf-8 -*-
"""
Create User : 19410
Desc : xxx
"""

# 由于transformers框架对应的网站 https://huggingface.co 需要外网访问，所以这里弄一个国内使用的网站
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'


def tt01():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
    from transformers import T5Tokenizer, T5Model

    model_id = 'google-t5/t5-base'
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModel.from_pretrained(model_id)
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5Model.from_pretrained(model_id)
    print(tokenizer)
    print(model)

    article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    summary = "Weiter Verhandlung in Syrien."
    inputs = tokenizer(article, return_tensors="pt")
    labels = tokenizer(text_target=summary, return_tensors="pt")
    print(labels)

    outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    hidden_states = outputs.last_hidden_state
    print(hidden_states.shape)


if __name__ == '__main__':
    tt01()
