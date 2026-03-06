#! -*- coding:utf-8 -*-
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

choice = 'train'  # train表示训练，infer表示推理
# choice = 'infer'  # train表示训练，infer表示推理

maxlen = 256  # 句子截断的最大长度，超过256的字会被砍掉或者分割
batch_size = 16  # 每次输入模型进行训练的样本数量
# 本地 BERT 预训练模型的路径（包含 config, pytorch_model.bin, vocab.txt）
model_dir = r"D:\cache\huggingface\hub\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
config_path = f'{model_dir}/bert4torch_config.json'  # 模型结构配置
checkpoint_path = f'{model_dir}/pytorch_model.bin'  # 模型权重文件
dict_path = f'{model_dir}/vocab.txt'  # 字典表，用于将字映射为ID

# 自动检测设备：如果有N卡GPU就用CUDA加速，否则用CPU慢跑
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === 数据集路径配置 ===
data_dir = '../datas/intention'
train_files = [f'{data_dir}/min_train.csv']
valid_files = [f'{data_dir}/min_val.csv']
test_files = [f'{data_dir}/min_test.csv']

# 意图识别的 12 个类别标签
label_names = [
    "Travel-Query",
    "Music-Play",
    "FilmTele-Play",
    "Video-Play",
    "Radio-Listen",
    "HomeAppliance-Control",
    "Weather-Query",
    "Alarm-Update",
    "Calendar-Query",
    "TVProgram-Play",
    "Audio-Play",
    "Other"
]
# 构建字典：标签名 -> 标签ID (例如: "Weather-Query" -> 6)
label_name2id = {label_name: label_id for label_id, label_name in enumerate(label_names)}
# 构建字典：标签ID -> 标签名 (例如: 6 -> "Weather-Query")
label_id2name = {label_id: label_name for label_id, label_name in enumerate(label_names)}

# 固定随机种子，保证每次运行代码划分的数据、初始化的参数一致，方便复现实验结果
seed_everything(42)

# 建立分词器：负责将中文字符切分成词元，do_lower_case=True 会将英文字母转为小写
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    # 假设 CSV 格式为：句子文本 \t 意图标签
                    text, label = l.strip().split('\t')
                    label = label_name2id[label]
                    # text_segmentate 会根据标点符号对过长的长文本进行切分，防止超过 maxlen
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D


def collate_fn(batch):
    """
    数据打包函数：将 MyDataset 吐出来的单条数据，打包成一个批次 (batch)。
    主要工作是进行 token化 和 sequence_padding (补齐)
    """
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        # encode 会自动在句子开头加 [CLS]，结尾加 [SEP]
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        # 句子标识，这里单句分类全是0
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    # sequence_padding 会把 batch 内所有句子补齐到该 batch 的最大长度 (用 0 填充)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    # 返回模型需要的输入格式: [token_ids, segment_ids], labels
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()


# 实例化 DataLoader，负责在训练时源源不断地提供 Batch 数据
# noinspection PyTypeChecker
train_dataloader = DataLoader(
    MyDataset(train_files),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
valid_dataloader = DataLoader(
    MyDataset(valid_files),
    batch_size=batch_size,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    MyDataset(test_files),
    batch_size=batch_size,
    collate_fn=collate_fn
)

from bert4torch.trainer import SequenceClassificationTrainer

# 加载 BERT 特征提取层
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,  # 提取 [CLS] 词元的特征向量（通常代表整个句子的语义）
    gradient_checkpoint=True  # 开启梯度检查点，用时间换空间，大幅节省显存
)
# 套上分类头，构建序列分类训练器，输出维度等于标签数量 (12)
model = SequenceClassificationTrainer(bert, num_labels=len(label_names)).to(device)


# 定义使用的loss和optimizer，这里支持自定义
def test_metric_func(*args, **kwargs):
    return 1.0


# 编译模型：指定多分类交叉熵损失、Adam优化器以及评估指标
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 微调BERT通常用极小的学习率 (如 2e-5)
    metrics=['accuracy', {'test_metric': test_metric_func}, test_metric_func]
)


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        # 记录历史最佳准确率
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        # 每个 Epoch 结束时，在验证集和测试集上跑一遍评估
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        # 如果当前模型的验证准确率创了新高，就保存它的权重
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in tqdm(data):
            # 预测类别：通过 argmax 获取概率最大的对应索引
            y_pred = model.predict(x_true).argmax(axis=1)
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total


def inference(texts):
    '''单条/多条样本推理（投入生产环境时用）'''
    for text in texts:
        # 对输入文本进行分词
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        # 转为 Tensor 并且增加一个 Batch 维度 ([None, :] 等价于 unsqueeze(0))
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)[None, :]

        # 获取模型裸输出 (Logits)
        logit = model.predict([token_ids, segment_ids])
        # softmax 转概率，argmax 找最大概率的索引，最后转为 numpy 数组
        y_pred = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
        print(text, ' ----> ', y_pred)


if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        # 开始训练，跑 10 个 Epoch
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
    else:
        # 如果 choice = 'infer'，则加载已经训练好的权重进行测试
        model.load_weights('best_model.pt')
        inference(['去龙门最近的路怎样走', '明天天气怎么样', '现在将空调开启制热模式吧'])
