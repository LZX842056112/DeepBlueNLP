# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 15:14
Create User : 19410
Desc : xxx
"""
import numpy as np
import torch

from ner.utils import extract_entities


@torch.no_grad()
def t0():
    from ner.datas.tokenizer import Tokenizer
    from ner.datas.dataset import build_collect_fn
    from ner.utils import trans_entity2tuple

    # ckpt = torch.load(r"./dsw_output/medical/bert/models/best.pkl", map_location='cpu')
    ckpt = torch.load(r"./output/travel_query/bert/models/best.pkl", map_location='cpu')
    net = ckpt['net']  # 网络结构恢复
    net.cpu().eval()
    append_special_token = ckpt.get('append_special_token', False)
    max_length = ckpt.get('max_length', None)
    label2ids = ckpt['label2ids']
    id2labels = {_id: _label for _label, _id in label2ids.items()}

    # PS: 临时修改一下
    # append_special_token = True
    # max_length = 128

    tokenizer = Tokenizer(
        vocabs=r'./datas/vocab.txt'
    )
    collect_fn = build_collect_fn(tokenizer.pad_token_id)

    text = "2天前无明显诱因患者感中下腹隐痛，呈持续性，伴腹胀，阵发性加重。有肛门排气排便。不伴发热、畏寒、寒战、反酸、烧心、嗳气、呃逆、头昏、头痛、乏力、胸闷、胸痛、心悸、气促、腹泻、黑便。腹痛与进食、活动、体位无明显关系，无肩背部放射痛。5小时前腹痛加重，呈剧痛，立即入我院急诊，查血常规：白细胞数 11.6×10^9/L、中性粒细胞百分比 77.64 %、淋巴细胞百分比 18.32 %、嗜酸性粒细胞百分比 0.24 %、中性粒细胞绝对值 9.02×10^9/L，肾功：尿素 2.32 mmol/L、尿酸 522.7 μmol/l，血淀粉酶 58.6 U/L，电解质1：钾 2.26 mmol/l、钠 127.6 mmol/、氯 84.1 mmol/l。腹部平片：上腹部肠曲积气、扩张，提示肠梗阻？结合临床或必要时进一步CT检查；腰椎骨质增生；考虑两上肺炎症，结合临床。彩超：中度脂肪肝；胆囊轻度增大 请结合临床；左肾囊肿；右肾切除术后。急诊以“腹痛待查”收入我科。患者此次患病以来，精神状态欠佳，大小便正常。体重无明显减轻"
    text = "查一下周六去上海的汽车票"

    batch = tokenizer(
        text, append_cls=append_special_token, append_sep=append_special_token,
        max_length=max_length
    )
    batch = list(batch)
    batch = collect_fn(batch)
    # print(batch)
    token_ids = batch['token_ids']
    token_masks = batch['token_masks']

    _, pred_probas = net(token_ids, token_masks)
    print(pred_probas.shape)

    pred_class_ids = torch.argmax(pred_probas, dim=-1)  # [bs, ?]
    pred_class_ids = (pred_class_ids * token_masks.numpy().astype(np.int32)
                      + (1 - token_masks.numpy().astype(np.int32)) * (- 100))  # 所有填充位置的预测类别重置为-100
    print(pred_class_ids.shape)
    print(pred_class_ids)

    pred_entities = trans_entity2tuple(
        label_ids=pred_class_ids.numpy(),
        label_id2names=id2labels
    )
    print(pred_entities)

    token_lengths = torch.sum(token_masks, dim=1).numpy()
    final_entities = extract_entities(
        text, pred_entities, sub_text_lengths=token_lengths, append_special_token=append_special_token
    )
    print(final_entities)

    # [{'label_type': '解剖部位', 'start_pos': 11, 'end_pos': 14, 'entity': '中下腹'}, {'label_type': '解剖部位', 'start_pos': 23, 'end_pos': 24, 'entity': '腹'}, {'label_type': '解剖部位', 'start_pos': 33, 'end_pos': 35, 'entity': '肛门'}, {'label_type': '解剖部位', 'start_pos': 42, 'end_pos': 43, 'entity': '发'}, {'label_type': '解剖部位', 'start_pos': 45, 'end_pos': 46, 'entity': '畏'}, {'label_type': '解剖部位', 'start_pos': 48, 'end_pos': 49, 'entity': '寒'}, {'label_type': '解剖部位', 'start_pos': 51, 'end_pos': 52, 'entity': '反'}, {'label_type': '解剖部位', 'start_pos': 55, 'end_pos': 56, 'entity': '心'}, {'label_type': '解剖部位', 'start_pos': 57, 'end_pos': 58, 'entity': '嗳'}, {'label_type': '解剖部位', 'start_pos': 60, 'end_pos': 61, 'entity': '呃'}, {'label_type': '解剖部位', 'start_pos': 63, 'end_pos': 64, 'entity': '头'}, {'label_type': '解剖部位', 'start_pos': 66, 'end_pos': 67, 'entity': '头'}, {'label_type': '解剖部位', 'start_pos': 69, 'end_pos': 70, 'entity': '乏'}, {'label_type': '解剖部位', 'start_pos': 72, 'end_pos': 73, 'entity': '胸'}, {'label_type': '解剖部位', 'start_pos': 75, 'end_pos': 76, 'entity': '胸'}, {'label_type': '解剖部位', 'start_pos': 78, 'end_pos': 79, 'entity': '心'}, {'label_type': '解剖部位', 'start_pos': 81, 'end_pos': 82, 'entity': '气'}, {'label_type': '解剖部位', 'start_pos': 84, 'end_pos': 85, 'entity': '腹'}, {'label_type': '解剖部位', 'start_pos': 90, 'end_pos': 91, 'entity': '腹'}, {'label_type': '解剖部位', 'start_pos': 108, 'end_pos': 111, 'entity': '肩背部'}, {'label_type': '解剖部位', 'start_pos': 119, 'end_pos': 120, 'entity': '腹'}, {'label_type': '解剖部位', 'start_pos': 128, 'end_pos': 129, 'entity': '立'}, {'label_type': '实验室检验', 'start_pos': 150, 'end_pos': 153, 'entity': '×10'}, {'label_type': '影像检查', 'start_pos': 238, 'end_pos': 240, 'entity': '2 '}, {'label_type': '实验室检验', 'start_pos': 241, 'end_pos': 243, 'entity': 'mo'}, {'label_type': '实验室检验', 'start_pos': 249, 'end_pos': 251, 'entity': ' 5'}, {'label_type': '实验室检验', 'start_pos': 264, 'end_pos': 266, 'entity': '淀粉'}, {'label_type': '实验室检验', 'start_pos': 280, 'end_pos': 284, 'entity': '1：钾 '}, {'label_type': '实验室检验', 'start_pos': 294, 'end_pos': 297, 'entity': 'l、钠'}, {'label_type': '药物', 'start_pos': 299, 'end_pos': 301, 'entity': '27'}, {'label_type': '药物', 'start_pos': 327, 'end_pos': 329, 'entity': '片：'}, {'label_type': '影像检查', 'start_pos': 341, 'end_pos': 345, 'entity': '示肠梗阻'}, {'label_type': '解剖部位', 'start_pos': 346, 'end_pos': 349, 'entity': '结合临'}, {'label_type': '解剖部位', 'start_pos': 349, 'end_pos': 350, 'entity': '床'}, {'label_type': '影像检查', 'start_pos': 357, 'end_pos': 361, 'entity': 'CT检查'}, {'label_type': '解剖部位', 'start_pos': 362, 'end_pos': 365, 'entity': '腰椎骨'}, {'label_type': '解剖部位', 'start_pos': 375, 'end_pos': 376, 'entity': '症'}, {'label_type': '影像检查', 'start_pos': 390, 'end_pos': 392, 'entity': '；胆'}, {'label_type': '疾病和诊断', 'start_pos': 395, 'end_pos': 401, 'entity': '增大 请结合'}, {'label_type': '疾病和诊断', 'start_pos': 404, 'end_pos': 409, 'entity': '左肾囊肿；'}, {'label_type': '疾病和诊断', 'start_pos': 418, 'end_pos': 423, 'entity': '以“腹痛待'}, {'label_type': '疾病和诊断', 'start_pos': 424, 'end_pos': 430, 'entity': '”收入我科。'}, {'label_type': '疾病和诊断', 'start_pos': 437, 'end_pos': 441, 'entity': '来，精神'}, {'label_type': '解剖部位', 'start_pos': 453, 'end_pos': 454, 'entity': '重'}]


def t1():
    from ner.deploy.onnx_predictor import Predictor

    predictor = Predictor(
        # onnx_model_path="./dsw_output/medical/bert/models/best.onnx"
        onnx_model_path="./output/travel_query/bert/models/best.onnx"
    )
    text = "2天前无明显诱因患者感中下腹隐痛，呈持续性，伴腹胀，阵发性加重。有肛门排气排便。不伴发热、畏寒、寒战、反酸、烧心、嗳气、呃逆、头昏、头痛、乏力、胸闷、胸痛、心悸、气促、腹泻、黑便。腹痛与进食、活动、体位无明显关系，无肩背部放射痛。5小时前腹痛加重，呈剧痛，立即入我院急诊，查血常规：白细胞数 11.6×10^9/L、中性粒细胞百分比 77.64 %、淋巴细胞百分比 18.32 %、嗜酸性粒细胞百分比 0.24 %、中性粒细胞绝对值 9.02×10^9/L，肾功：尿素 2.32 mmol/L、尿酸 522.7 μmol/l，血淀粉酶 58.6 U/L，电解质1：钾 2.26 mmol/l、钠 127.6 mmol/、氯 84.1 mmol/l。腹部平片：上腹部肠曲积气、扩张，提示肠梗阻？结合临床或必要时进一步CT检查；腰椎骨质增生；考虑两上肺炎症，结合临床。彩超：中度脂肪肝；胆囊轻度增大 请结合临床；左肾囊肿；右肾切除术后。急诊以“腹痛待查”收入我科。患者此次患病以来，精神状态欠佳，大小便正常。体重无明显减轻"
    text = "查一下周六去上海的汽车票"
    result = predictor.predict(text)
    print(result)


def t2():
    import requests

    url = "http://127.0.0.1:9001/predict"

    def tt_get():
        text = "，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
        text = "患者2年余前因确诊胃癌，于2010年12月28日在外院行胃大部分切除及食道上段切除术。术后病理示“胃中分化腺癌，浸润胃壁全层至浆膜上，向下侵犯食管鳞状下皮粘膜上层，淋巴结见癌转移3/18，PT3N2 IIIA期”。术后予XELOX方案化疗3个疗程，化疗后间有呕血。定期复查，未见异常。10天前复查CEA 49NG/ML；PET示“胃底及食管后见淋巴结高代谢，考虑转移瘤可能性大”，穿刺病理提示腺癌。于2013-02-05给予替吉奥（S1）+白蛋白紫杉醇化疗第1程。现为进一步治疗，收入我科。近期以来，患者精神、胃纳可，无腹痛、腹胀，无发热、寒战、恶心、呕吐，大小便无异常，体重无明显上降。"
        response = requests.get(
            url, json={'text': text}
        )
        if response.status_code == 200:
            print("访问服务器成功GET")
            # 明确的知道服务器返回的是json格式，那么可以直接调用json()这个方法转换为字典对象
            result = response.json()
            print(result)
            print(type(result))
            if result['code'] == 0:
                print(f"调用模型成功GET，结果为:{result['data']}")
            else:
                print(f"调用模型服务器处理异常，异常信息为:{result['msg']}")
        else:
            print("访问服务器失败：网络异常等等")

    def tt_post():
        # NOTE: 在fastapi框架中，post请求的参数必须通过json参数给定
        text = "，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
        text = "患者2年余前因确诊胃癌，于2010年12月28日在外院行胃大部分切除及食道上段切除术。术后病理示“胃中分化腺癌，浸润胃壁全层至浆膜上，向下侵犯食管鳞状下皮粘膜上层，淋巴结见癌转移3/18，PT3N2 IIIA期”。术后予XELOX方案化疗3个疗程，化疗后间有呕血。定期复查，未见异常。10天前复查CEA 49NG/ML；PET示“胃底及食管后见淋巴结高代谢，考虑转移瘤可能性大”，穿刺病理提示腺癌。于2013-02-05给予替吉奥（S1）+白蛋白紫杉醇化疗第1程。现为进一步治疗，收入我科。近期以来，患者精神、胃纳可，无腹痛、腹胀，无发热、寒战、恶心、呕吐，大小便无异常，体重无明显上降。"

        response = requests.post(
            url,
            json={'text': text},
        )
        if response.status_code == 200:
            print("访问服务器成功POST")
            # 明确的知道服务器返回的是json格式，那么可以直接调用json()这个方法转换为字典对象
            result = response.json()
            print(result)
            print(type(result))
            if result['code'] == 0:
                print(f"调用模型成功POST，结果为:{result['data']}")
            else:
                print(f"调用模型服务器处理异常，异常信息为:{result['msg']}")
        else:
            print("访问服务器失败：网络异常等等")

    tt_get()
    tt_post()


if __name__ == '__main__':
    t2()
