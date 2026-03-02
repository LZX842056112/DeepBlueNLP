# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:52
Create User : 19410
Desc : 意图识别数据的相关模型训练
"""

import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")))

print(sys.path)


def run_preprocess():
    """
    数据预处理
    :return:
    """
    # 下列代码能够正常运行的前提是: text_classify 它所在的文件夹必须在sys.path环境变量中
    from text_classify.datas import preprocess

    preprocess.intention_process(
        intention_data_file=r"./datas/intention/train.csv",
        token2id_file=r"./output/intention/token2id.json",
        label2id_file=r"./output/intention/label2id.json",
    )


def run_train_lstm():
    # 下列代码能够正常运行的前提是: text_classify 它所在的文件夹必须在sys.path环境变量中
    from text_classify.trainer.trainer import Trainer
    from text_classify.config import Config
    from text_classify.datas.tokenizer import Tokenizer
    from text_classify.utils import load_json

    tokenizer = Tokenizer(
        token2ids=load_json(r"./output/intention/token2id.json"),
        label2ids=load_json(r"./output/intention/label2id.json")
    )
    cfg = Config(
        model_output_dir="./output/intention/lstm2/models",
        summary_dir="./output/intention/lstm2/logs",
        tokenizer=tokenizer,
        train_file="./datas/intention/train0.csv",
        eval_file="./datas/intention/val0.csv",
        # total_epoch=5,
        total_epoch=10,
        batch_size=8,
        hidden_size=128,
        lr=0.01
    )

    trainer = Trainer(cfg)
    trainer.training()

    # 针对转换后的静态结构，我们可以通过 https://netron.app/ 查看结构


def run_train_bert():
    # 下列代码能够正常运行的前提是: text_classify 它所在的文件夹必须在sys.path环境变量中
    from text_classify.trainer.trainer import Trainer
    from text_classify.config import Config
    from text_classify.datas.tokenizer import ProxyBertTokenizer
    from text_classify.utils import load_json

    bert_path = r"D:\cache\huggingface\hub\models--bert-base-chinese"

    tokenizer = ProxyBertTokenizer(
        bert_tokenizer_file=bert_path,
        label2ids=load_json(r"./output/intention/label2id.json")
    )
    cfg = Config(
        model_output_dir="./output/intention/bert1/models",
        summary_dir="./output/intention/bert1/logs",
        tokenizer=tokenizer,
        train_file="./datas/intention/train0.csv",
        eval_file="./datas/intention/val0.csv",
        # total_epoch=5,
        total_epoch=20,
        batch_size=8,
        hidden_size=128,
        lr=0.01,
        bert_path=bert_path,
        network_type="bert",
        freeze=11,
        device="cuda",
        max_no_improved_epoch=2
    )

    trainer = Trainer(cfg)
    trainer.training()

    # 针对转换后的静态结构，我们可以通过 https://netron.app/ 查看结构


def run_export():
    # 下列代码能够正常运行的前提是: text_classify 它所在的文件夹必须在sys.path环境变量中
    from text_classify.config import Config
    from text_classify.datas.tokenizer import ProxyBertTokenizer
    from text_classify.utils import load_json
    from text_classify.export import export_jit, export_onnx

    bert_path = r"D:\cache\huggingface\hub\models--bert-base-chinese"

    tokenizer = ProxyBertTokenizer(
        bert_tokenizer_file=bert_path,
        label2ids=load_json(r"./output/intention/label2id.json")
    )
    cfg = Config(
        model_output_dir="./output/intention/bert1/models",
        summary_dir="./output/intention/bert1/logs",
        tokenizer=tokenizer,
        train_file="./datas/intention/train0.csv",
        eval_file="./datas/intention/val0.csv",
        # total_epoch=5,
        total_epoch=10,
        batch_size=8,
        hidden_size=128,
        lr=0.01,
        bert_path=bert_path,
        network_type="bert",
        freeze=11
    )

    # export_jit(cfg, model_name="best.pkl")
    export_onnx(cfg, model_name="best.pkl")


def run_predictor():
    from text_classify.deploy.jit_predictor import Predictor

    p = Predictor(
        # jit_model_path="./output/intention/lstm2/models/best.pt"
        jit_model_path="./output/intention/bert1/models/best.pt"
    )

    while True:
        v = input("请输入文本:")
        if "q" == v:
            break

        r = p.predict(v, k=3)
        print(r)


def run_predictor_val():
    import pandas as pd
    from text_classify.deploy.jit_predictor import Predictor

    p = Predictor(
        jit_model_path="./output/intention/bert1/models/best.pt"
    )

    df = pd.read_csv(r"./datas/intention/val0.csv", sep="\t", header=None, names=['text', 'label'])
    datas = []
    for _, row in df.iterrows():
        r = p.predict(row['text'], k=3)
        if r[0]['cls_name'] == row['label']:
            continue
        data = {
            'text': row['text'],
            'label': row['label'],
            'rl_1': r[0]['cls_name'],
            'rp_1': round(r[0]['prob'], 2),
            'rl_2': r[1]['cls_name'],
            'rp_2': round(r[1]['prob'], 2),
            'rl_3': r[2]['cls_name'],
            'rp_3': round(r[2]['prob'], 2)
        }
        datas.append(data)

    df = pd.DataFrame(datas)
    df.sort_values(by=["label", "rl_1"], inplace=True)
    df.to_csv(r"./datas/intention/val0_result.csv", index=False)


def run_predictor_with_onnx():
    from text_classify.deploy.onnx_predictor import Predictor

    p = Predictor(
        onnx_model_path="./output/intention/bert1/deploy/best.onnx"
    )

    print(p.predict("帮我打开一下空调"))

    while True:
        v = input("请输入文本:")
        if "q" == v:
            break

        r = p.predict(v, k=3)
        print(r)


if __name__ == '__main__':
    # run_preprocess()
    # run_train_lstm()
    # run_predictor()
    # run_train_bert()
    # run_export()
    # run_predictor_val()
    run_predictor_with_onnx()
