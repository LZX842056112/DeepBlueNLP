# -*- coding: utf-8 -*-
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if 'nt' in os.name:
    os.environ['XDG_CACHE_HOME'] = r'D:/cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

model_id = "Tongjilibo/chinese_t5_pegasus_small"

# # 方式1：直接使用model自动加载 --> 仅支持Transformer框架的模型结构的下载
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModel.from_pretrained(model_id)
# print(tokenizer)
# print(model)

# 方式2：直接使用transformers内部的下载API进行文件下载
local_path = snapshot_download(repo_id=model_id)
print(local_path)
# cp -aL /xxxxx/* /mnt/workspace/models/chinese_t5_pegasus_small