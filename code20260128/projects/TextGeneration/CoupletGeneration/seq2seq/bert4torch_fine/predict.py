# -*- coding: utf-8 -*-
import os

import torch

import jieba
from bert4torch.models import build_transformer_model
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.tokenizers import Tokenizer

max_c_len = 256
max_t_len = 32
batch_size = 128
# epochs = 100
epochs = 2
steps_per_epoch = None

pretrain_model = r"/mnt/workspace/models/chinese_t5_pegasus_small"
#pretrain_model = r"D:\cache\huggingface\hub\models--Tongjilibo--chinese_t5_pegasus_small\snapshots\3e5558b23dbf6ace9c1cc024b9ddc6b0e5a3f8da"
config_path = os.path.join(pretrain_model, 'bert4torch_config.json')
checkpoint_path = os.path.join(pretrain_model, 'pytorch_model.bin')
dict_path = os.path.join(pretrain_model, 'vocab.txt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)
model = build_transformer_model(config_path, checkpoint_path, add_trainer=True).to(device)


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        res = model.decoder.predict([output_ids] + inputs)
        return res[-1][:, -1, :] if isinstance(res, list) else res[:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1):
        token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])  # 得到编码器的输出
        output_ids = self.beam_search(encoder_output, top_k=topk)[0]  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids.cpu().numpy()])


autotitle = AutoTitle(
    bos_token_id=tokenizer._token_start_id,
    eos_token_id=tokenizer._token_end_id,
    max_new_tokens=max_t_len,
    device=device,
    top_k=4
)

print("模型恢复....")
#model.load_weights("best_model_v2.pt")
model.load_weights("best_model_v3.pt")  # 在v2的基础上支持给定上联两个字预测下联两个字

# %%
for upper in [
    '下联生成:日晃百花色',
    '下联生成:碧林青旧竹',
    '上下联生成:日月',
    '上下联生成:红绿',
    '下联生成:红绿',
    '下联生成:五湖春色满',
    '下联生成:五云开锦绣',
    '上下联生成:五千',
    '上下联生成:五四'
]:
    print(u'生成下联:  ', upper, '  ', autotitle.generate(upper, topk=2))

# %%

# print(model)
