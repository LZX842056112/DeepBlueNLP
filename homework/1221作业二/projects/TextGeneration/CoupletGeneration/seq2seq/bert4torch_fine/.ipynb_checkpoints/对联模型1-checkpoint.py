# %%
import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, seed_everything, ListDataset
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.callbacks import Callback
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba

jieba.initialize()


# %%
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, "r", encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                try:
                    text1, text2 = line.split("，")  # 分割得到上联和下联
                    text2 = text2[:-1]
                    D.append((text1, text2))
                except:
                    pass
        return D


# %%
def collate_fn(batch):
    """
        单条样本格式：content：[CLS]文章[SEP]  tgt: [CLS]标题[SEP]
        单条样本格式: 编码器 -> [CLS]上联[SEP]  解码器 --> [CLS]下联[SEP]
    """
    batch_content_ids, batch_titile_ids = [], []
    for upper, lower in batch:
        token_ids, _ = tokenizer.encode(upper, maxlen=max_c_len)
        batch_content_ids.append(token_ids)

        token_ids, _ = tokenizer.encode(lower, maxlen=max_t_len)
        batch_titile_ids.append(token_ids)

    batch_content_ids = torch.tensor(sequence_padding(batch_content_ids), dtype=torch.long, device=device)
    batch_titile_ids = torch.tensor(sequence_padding(batch_titile_ids), dtype=torch.long, device=device)
    return [[batch_content_ids], [batch_titile_ids[:, :-1]]], batch_titile_ids[:, 1:].flatten()


# %%

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, y_true):
        y_pred = outputs[-1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)


# %%
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
        encoder_output = model.encoder.predict([token_ids])
        output_ids = self.beam_search(encoder_output, topk=topk)[0]  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids.cpu().numpy()])


# %%
def just_show():
    texts = [
        '日晃百花色'
    ]
    for text in texts:
        print(text, "  ", autotitle.generate(text))


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        super(Evaluator, self).__init__()
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        just_show()
        metrics = self.evaluate(valid_dataset.data)  # 评测模型
        metrics_test = self.evaluate(test_dataset.data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('./best_model_v0.pt')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
        print('test_data:', metrics_test)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for upper, lower in tqdm(data):
            total += 1
            title = ' '.join(lower).lower()
            pred_title = ' '.join(autotitle.generate(upper, topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[title.split(' ')], hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1, rouge_2, rouge_l, bleu = rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total
        return {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu}


# %% [markdown]
# # 开始运行

# %%
# %% 属性定义

max_c_len = 256  # 编码器最大长度
max_t_len = 32  # 解码器的最大长度
batch_size = 64  # 批次大小
epochs = 50
steps_per_epoch = None

# pretrain_model = r"/mnt/workspace/models/chinese_t5_pegasus_small"
pretrain_model = r"D:\cache\huggingface\hub\models--Tongjilibo--chinese_t5_pegasus_small\snapshots\3e5558b23dbf6ace9c1cc024b9ddc6b0e5a3f8da"
config_path = os.path.join(pretrain_model, 'bert4torch_config.json')
checkpoint_path = os.path.join(pretrain_model, 'pytorch_model.bin')
dict_path = os.path.join(pretrain_model, 'vocab.txt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

# %%
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

train_dataset = MyDataset("./datas/poetry.txt")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
valid_dataset = MyDataset("./datas/poetry_min.txt")
test_dataset = MyDataset("./datas/poetry_min.txt")

# %%
model = build_transformer_model(config_path, checkpoint_path, add_trainer=True).to(device)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-4))

# %%
autotitle = AutoTitle(
    bos_token_id=tokenizer._token_start_id,
    eos_token_id=tokenizer._token_end_id,
    max_new_tokens=max_t_len,
    device=device
)

# %% [markdown]
# ## 训练代码

# %%
evaluator = Evaluator()
print(u'生成下联:', autotitle.generate(u'日晃百花色'))
# model.fit(
#     train_dataloader,
#     steps_per_epoch=steps_per_epoch,
#     epochs=epochs,
#     callbacks=[evaluator]
# )

# %%
just_show()

# %% [markdown]
# ## 模型恢复 + 测试

# %%
model.load_weights("best_model_v0.pt")

# %%
for upper in [
    '日晃百花色',
    '碧林青旧竹',
    '碧绿',
    '五湖春色满'
]:
    print(u'生成下联:  ', upper, '  ', autotitle.generate(upper, topk=2))

# %%
