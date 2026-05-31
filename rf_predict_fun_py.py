# 导包
import os

import jieba
import pandas as pd
import pickle
import sys

from config import Config


pd.set_option('display.max_columns', None)
# 初始化配置
conf = Config()

# 加载自定义词典（在模块加载时执行一次）
_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dict.txt")
if os.path.exists(_dict_path):
    jieba.load_userdict(_dict_path)

# todo 1 加载模型和向量化器
with open(conf.rf_model_save_path, 'rb') as f:
    model_idf = pickle.load(f)

model = model_idf['model']
tfidf = model_idf['tfidf']
print("模型加载完成")


# 定义预测函数
def predict_func(data):
    """

    :param data: 为待预测数据：{"text", "待预测的文本"}
    :return: {"text",为待预测文本. "pred_class",具体的分类}
    """
    text = data['text']

    # jieba分词
    words = ' '.join(jieba.lcut(text)[:30])

    # 获取数值特征
    feature = tfidf.transform([words])

    # 模型预测
    pred_id = model.predict(feature)
    print("模型预测结果: ", pred_id[0])

    # 将索引转换成具体的分类
    label2class = {i: lines.strip() for i, lines in enumerate(open(conf.class_path, encoding='utf-8'))}
    print(f"label2class: {label2class}")
    pre_class = label2class[int(pred_id[0])]
    print(f"pred_class: {pre_class}")

    # 查看预测概率
    proba = model.predict_proba(feature)[0]
    print("\n--- 各类别预测概率 ---")
    for idx, prob in enumerate(proba):
        print(f"{label2class[idx]}: {prob:.4f}")
    print("----------------------\n")

    # 打印特征中包含的词汇
    print("提取到的特征词:", tfidf.inverse_transform(feature))

    data['pred_class'] = pre_class
    return data


if __name__ == '__main__':
    data = {"text": "携带方便 佳能 A495最新报价仅为760元"}
    predict_func(data)



