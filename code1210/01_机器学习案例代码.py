# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/10 20:10
Create User : 19410
Desc : 机器学习训练 + 推理
"""

import os
import warnings
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import json

warnings.filterwarnings('ignore')


class Predictor(object):
    def __init__(self):
        super().__init__()
        # 1. 模型恢复
        obj = joblib.load("./output/01/ml.pkl")
        poly = obj['poly']
        algo = obj['algo']
        print(f"模型恢复完成: {poly} -- {algo}")
        self.poly = poly
        self.algo = algo

    def predict(self, x):
        # 2. 数据转换
        x = self.poly.transform(x)

        # 3. 模型预测
        y_pred_proba = self.algo.predict_proba(x)

        # 4. 结果拼接
        y_pred_idx_per_sample = np.argmax(y_pred_proba, axis=1).tolist()
        y_pred_proba_per_sample = y_pred_proba[range(len(y_pred_idx_per_sample)), y_pred_idx_per_sample].round(
            3).tolist()
        result = list(map(lambda t: {'id': t[0], 'proba': t[1]}, zip(y_pred_idx_per_sample, y_pred_proba_per_sample)))
        return result


# noinspection PyTypeChecker
def training():
    # 1. 加载数据
    X, Y = make_circles(
        n_samples=1000,  # 样本数目
        noise=0.1,  # 噪声样本比例
        factor=0.2,  # 内圈直径是外圈直径的factor倍
        random_state=24  # 随机数种子
    )
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=24)
    print(f"训练数据shape形状为: {type(x_train)} - {x_train.shape} -- {type(y_train)} - {y_train.shape}")
    print(f"评估数据shape形状为: {type(x_test)} - {x_test.shape} -- {type(y_test)} - {y_test.shape}")
    print(f"类别取值: {np.unique(y_train)} - {np.bincount(y_train)} -- {np.unique(y_test)} - {np.bincount(y_test)}")

    # 2. 特征工程
    poly = PolynomialFeatures(degree=2)
    x_train = poly.fit_transform(x_train)
    x_test = poly.transform(x_test)
    print(f"多项式扩展转换规则: {poly.get_feature_names_out(['x1', 'x2'])}")
    print(f"转换后训练数据shape形状为: {type(x_train)} - {x_train.shape}")
    print(f"转换后评估数据shape形状为: {type(x_test)} - {x_test.shape}")

    # 3. 模型创建
    algo = LogisticRegression(max_iter=10)
    # algo = DecisionTreeClassifier()

    # 4. 模型训练
    algo.fit(x_train, y_train)

    # 5. 模型评估
    pred_train = algo.predict(x_train)
    pred_test = algo.predict(x_test)
    print(f"预测结果类型: {type(pred_train)} - {pred_train.shape}")
    print(f"训练数据上的准确率: {metrics.accuracy_score(y_train, pred_train)}")
    print(f"评估数据上的准确率: {metrics.accuracy_score(y_test, pred_test)}")
    print(f"训练数据上的分类报告: \n{metrics.classification_report(y_train, pred_train)}\n")
    print(f"评估数据上的分类报告: \n{metrics.classification_report(y_test, pred_test)}\n")

    # 6. 模型持久化
    joblib_dump_file = "./output/01/ml.pkl"
    os.makedirs(os.path.dirname(joblib_dump_file), exist_ok=True)
    joblib.dump({
        'poly': poly,
        'algo': algo
    }, joblib_dump_file)

    if isinstance(algo, LogisticRegression):
        json_dump_file = "./output/01/ml.json"
        with open(json_dump_file, "w", encoding="utf-8") as writer:
            json.dump(
                {
                    'poly': poly.get_feature_names_out(['x1', 'x2']).tolist(),  # 获取多项式的组合规则，并转换为list输出
                    'algo': {
                        'intercept': algo.intercept_.tolist(),  # 获取LR的截距项，并转换为list输出
                        'coef': algo.coef_.tolist()  # 提取LR的参数项，并转换为list输出(json默认仅支持普通python类型)
                    }
                },  # 持久化的对象
                writer,  # 输出文件对象
                indent=2,  # json格式化空格 -- 每个级别前面空2个空格
                ensure_ascii=False  # 中文不进行编码输出，直接输出中文
            )


def interface():
    # 1. 模型恢复
    obj = joblib.load("./output/01/ml.pkl")
    poly = obj['poly']
    algo = obj['algo']
    print(f"模型恢复完成: {poly} -- {algo}")

    # 2. 数据转换
    x = [
        [0.05, -0.01],
        [0.1, 0.3],
        [-0.4, 0.2],
        [1.0, 1.2],
        [0.0, 0.75],
        [0.0, -1.2]
    ]
    x = poly.transform(x)

    # 3. 模型预测
    y_pred_proba = algo.predict_proba(x)
    print(f"获取预测概率对象为: {type(y_pred_proba)} - {y_pred_proba.shape}")

    # 4. 结果拼接
    y_pred_idx_per_sample = np.argmax(y_pred_proba, axis=1).tolist()
    y_pred_proba_per_sample = y_pred_proba[range(len(y_pred_idx_per_sample)), y_pred_idx_per_sample].round(3).tolist()
    result = list(map(lambda t: {'id': t[0], 'proba': t[1]}, zip(y_pred_idx_per_sample, y_pred_proba_per_sample)))
    print(result)


def interface02():
    p = Predictor()

    r0 = p.predict([[0, 0.3], [1.2, 1.5]])
    print(r0)

    while True:
        v = input("请输入样本特征，使用空格隔开:")
        if "q" == v:
            break

        x = list(map(lambda s: float(s.strip()), v.split(" ")))
        r = p.predict([x])
        print(r)


if __name__ == '__main__':
    training()
    # interface()
    # interface02()
