import random

import torch

from spert import util


def create_train_sample(
        doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int,
        build_rel_data: bool = True
):
    encodings = doc.encoding  # 当前文本对应的token id列表
    token_count = len(doc.tokens)  # 当前文本的实际token数量
    context_size = len(encodings)  # 当前文本的token id(输入模型的input ids)长度

    # region ==== 该部分为实体的训练数据处理逻辑  ====

    # positive entities 正样本(实体)的处理
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)  # 在token中的起始位置、终止位置
        pos_entity_types.append(e.entity_type.index)  # 实体类别id
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))  # 给定当前span的范围
        pos_entity_sizes.append(len(e.tokens))  # span的长度 实体的长度

    # negative entities 负样本(实体)的处理 ---> 遍历产生
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span  # token片段对应的input ids的范围
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)
    # sample negative entities 上部分代码随机产生了大量的候选负样本(实体)列表，这里随机选择部分作为进入模型训练的负样本(实体)
    neg_entity_samples = random.sample(
        list(zip(neg_entity_spans, neg_entity_sizes)),
        min(len(neg_entity_spans), neg_entity_count)
    )
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])
    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)  # 0表示不属于实体

    # endregion

    # region ==== 该部分为关系的训练数据处理逻辑  ====

    # positive relations 正样本(关系)的处理

    # collect relations between entity pairs
    entity_pair_relations = dict()
    for rel in doc.relations:
        pair = (rel.head_entity, rel.tail_entity)
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # build positive relation samples
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair  # 主体、客体
        s1, s2 = head_entity.span, tail_entity.span  # 主体的范围、客体的范围
        # 主体、客体在当前正样本(实体)中的是第几个
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_rel_spans.append((s1, s2))

        pair_rel_types = [r.relation_type.index for r in rels]  # 关系类别id列表
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]  # 关系类别id进行MultiOneHot转换
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))  # 将两个实体中间位置设为true，其它位置为false

    # negative relations 负样本(关系)构建 -- 遍历所有的正样本实体，来构建负关系对象
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    # sample negative relations 抽样
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [(0,) * (rel_type_count - 1)] * len(neg_rel_spans)

    # endregion

    # merge 正负样本合并
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens 对应输入token id的mask矩阵
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if build_rel_data and rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count - 1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    """
        T: 处理后的token序列长度
        N1: 表示实体正样本和负样本的总数量
        N2: 表示关系正样本和负样本的总数量
        REL_N: 表示关系的类别数量
        encodings: 原始的token进行wordsprice处理之后的对应id列表，也就是理解成模型的原始输入x，eg: [101, 2342, ...., 234, 102], shape: [T]
        context_masks: 恒定为1/True的一个列表, shape: [T]
        entity_masks: 表示实体正样本和负样本的mask列表，每个实体一个mask列表；
            如果某个token属于当前实体，那么该token对应位置为1/True，否则为False, shape: [N1,T]
        entity_sizes: 每个实体(正样本和负样本)的长度，shape: [N1]
        entity_types：每个实体(正样本和负样本)的类别id，shape: [N1]
        entity_sample_masks: 全为1, shape: [N1]
        rels: 给定实体对的索引下标,下标取值范围:[0,N1)，shape: [N2,2]
        rel_masks.rel_masks：表示任意两个实体之间的关系mask列表，
            每个实体对存在一个mask；其中两个实体之间的token对应位置为1/True，其它位置为False，shape:[N2,T]
        rel_types: 给定两个实体对之间的关系类别id进行multi onehot之后的值， eg: [0,1,0,0,1]表示这两个实体之间的关系存在两种:关系2和关系5，shape:[N2,REL_N]
        rel_sample_masks：全为1，shape: [N2]
    """
    return dict(
        encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
        entity_sizes=entity_sizes, entity_types=entity_types,
        rels=rels, rel_masks=rel_masks, rel_types=rel_types,
        entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks
    )


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding  # 当前文本对应的token id列表
    token_count = len(doc.tokens)  # 当前文本的token长度（原始实际token长度）
    context_size = len(encodings)  # 当前文本的token id列表长度

    # create entity candidates 产生候选实体片段集合 --> 待预测的实体组合
    entity_spans = []  # 候选实体span片段列表 保存的是片段的index范围
    entity_masks = []  # 保存片段对应mask向量，属于片段位置就是True，不属于就是False
    entity_sizes = []  # 片段的长度

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            if i == 6 and size == 4:
                print(i, size)
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(
        encodings=encodings, context_masks=context_masks,
        entity_masks=entity_masks,
        entity_sizes=entity_sizes,
        entity_spans=entity_spans,
        entity_sample_masks=entity_sample_masks
    )


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
