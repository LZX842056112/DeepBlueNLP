# -*- coding: utf-8 -*-

import copy
import os
from typing import Union, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class MyDataset(Dataset):
    def __init__(self, word_file, tokenizer: BertTokenizer):
        super(MyDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.datas = self.load_data(word_file)

    def load_data(self, word_file):
        datas = []
        with open(word_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                text1, text2 = line.split("，")  # 分割成上下联
                text2 = text2[:-1]

                encoder_input_ids = self.tokenizer.encode_plus(list(text1))['input_ids']
                decoder_input_ids = self.tokenizer.encode_plus(list(text2))['input_ids']
                datas.append((
                    encoder_input_ids,
                    f"{self.cls_token} {text1} {self.sep_token}",
                    len(encoder_input_ids),
                    decoder_input_ids,
                    f"{self.cls_token} {text2} {self.sep_token}",
                    len(decoder_input_ids),
                ))
        return datas

    def __getitem__(self, index):
        encoder_input_ids, encoder_text, encoder_input_ids_num, \
            decoder_input_ids, decoder_text, decoder_input_ids_num, = self.datas[index]
        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.int64),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.int64),
            "encoder_attention_mask": torch.ones(encoder_input_ids_num, dtype=torch.int64),
            "decoder_attention_mask": torch.ones(decoder_input_ids_num, dtype=torch.int64),
            "encoder_text": encoder_text,
            "decoder_text": decoder_text,
        }

    def __len__(self):
        return len(self.datas)


def padding_tensors(values):
    return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)


def collate_fn(batch):
    result = {}
    keys = batch[0].keys()
    bs = len(batch)
    for key in keys:
        values = [batch[i][key] for i in range(bs)]
        if isinstance(values[0], torch.Tensor):
            values = padding_tensors(values)  # 数据填充，均填充0
        result[key] = values

    if 'labels' not in result:
        # noinspection PyUnresolvedReferences
        labels = copy.deepcopy(result['decoder_input_ids'])
        labels[labels == 0] = -100  # 填充位置不计算损失
        result['labels'] = labels

    return result


class Seq2SeqWithBertModel(BertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: BertConfig):
        super(Seq2SeqWithBertModel, self).__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = BertModel(encoder_config, add_pooling_layer=False)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.decoder = BertModel(decoder_config, add_pooling_layer=False)

        self.cls = BertOnlyMLMHead(config)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.Tensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        # 获取编码器的输出结果
        if encoder_hidden_states is None:
            encoder_output = self.encoder(
                input_ids,  # 编码器的输入id
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False
            )
            encoder_hidden_states = encoder_output[0]  # 最后一层的输出
            encoder_attention_mask = attention_mask

        # 解码器
        outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


def train(vocab_file, word_file, output_dir, hidden_size=128, batch_size=8, total_epoch=100):
    model_save_dir = os.path.join(output_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 数据加载相关代码逻辑
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tokenizer.save_pretrained(model_save_dir)  # 先保存一下
    ds = MyDataset(word_file=word_file, tokenizer=tokenizer)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 模型创建
    cfg = BertConfig(
        is_decoder=True,
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=hidden_size * 4,
        tie_word_embeddings=False
    )
    net = Seq2SeqWithBertModel(cfg)
    net.to(device=device)
    opt = optim.AdamW(params=net.parameters(), lr=0.001)

    # 遍历数据进行训练
    opt.zero_grad()
    for epoch in range(total_epoch):
        net.train()
        for batch in dataloader:
            for key in batch.keys():
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device)

            # 前向过程
            net_output = net(
                input_ids=batch['encoder_input_ids'],
                attention_mask=batch['encoder_attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                labels=batch['labels'],
                use_cache=False,
                return_dict=True
            )
            loss = net_output.loss

            # 反向传播
            loss.backward()
            opt.step()
            opt.zero_grad()

            print(f"{epoch} Loss: {loss.item():.3f}")

    # 模型持久化保存
    net.save_pretrained(model_save_dir)


@torch.no_grad()
def predict():
    output_dir = "output/bert01"
    model_save_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)  # 恢复解析器
    bert = Seq2SeqWithBertModel.from_pretrained(model_save_dir)  # 恢复模型
    print(bert)
    bert.eval()

    def _predict(_encoder_input_ids, _decoder_input_ids):
        output_token_ids = copy.deepcopy(_decoder_input_ids)
        new_tokens = 0
        past_key_values = None

        # 获取编码器的输出
        encoder_output = bert.encoder(
            _encoder_input_ids,  # 编码器的输入id
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )
        encoder_hidden_states = encoder_output[0]  # 最后一层的输出

        while new_tokens < 7:
            bert_output = bert(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=_decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = bert_output.past_key_values

            logits = bert_output.logits
            logits = logits[:, -1, :]  # 最后一个时刻的预测置信度
            pred_ids = torch.argmax(logits, dim=-1, keepdim=True)  # 选择预测概率最大的token作为当前时刻的预测结果 [N,1]

            if pred_ids[0, 0].item() == tokenizer.sep_token_id:
                break
            output_token_ids = torch.concat([output_token_ids, pred_ids], dim=1)
            _decoder_input_ids = pred_ids

            new_tokens += 1
        # 将id转换为token
        t = tokenizer.convert_ids_to_tokens(list(output_token_ids.cpu().numpy()[0]))
        return "".join(t[1:])

    while True:
        text = input("请输入5字上联:")
        encoder_input_ids = tokenizer.encode_plus(list(text))['input_ids']
        encoder_input_ids = torch.tensor([encoder_input_ids], dtype=torch.int64)
        decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.int64)
        word = _predict(encoder_input_ids, decoder_input_ids)
        print(word)


if __name__ == '__main__':
    # train(
    #     vocab_file="../datas/vocab.txt",
    #     word_file="../datas/poetry_min.txt",
    #     output_dir="./output/bert01",
    #     hidden_size=128,
    #     batch_size=4,
    #     total_epoch=100
    # )
    predict()
