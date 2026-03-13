# -*- coding: utf-8 -*-

# 由于transformers框架对应的网站 https://huggingface.co 需要外网访问，所以这里弄一个国内使用的网站
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'


def tt01():
    from transformers import BertTokenizer, BertModel

    model_id = "bert-base-chinese"
    model_id = r"D:\huggingface\huggingface\hub\models--bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertModel.from_pretrained(model_id)
    print(tokenizer)
    print(model)

    text = "active unaffable my name is 小小，上海有什么龘鱻驫"
    # 针对给定文本进行分词、token id转换的操作
    """
    分词、token id转换对应的参数信息
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:

    采用WordPiece的分词原理： --- 其实针对中文来讲就是分字(也就是以字作为词)
        针对在词汇映射表中的token，直接获取该单次对应的token id；
        针对未知词(也就是不在词汇映射表中)，那么进行前缀提取匹配的逻辑产生多个token id；比如：unaffable 这个token会被拆分为多个子token ["u", "##na", "##ff", "##able"]; PS：若某个单词不管如何拆分在词汇表中都不存在，那么直接设置为[UNK]
    """
    token_encoder = tokenizer([text, "我是小明"], return_tensors="pt", padding=True)
    print(token_encoder)

    # 调用模型
    """
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    """
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
