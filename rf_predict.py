# 导包
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from config import Config
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# 初始化配置
conf = Config()

# todo 1. 加载数据
df = pd.read_csv(conf.process_dev_datapath, sep='\t', header=1, names=['text', 'label', 'words'])
words = df['words']
labels = df['label']
#
# print(words.head(10))
# print(labels.head(10))

# todo 2. 加载模型和向量化器
with open(conf.rf_model_save_path, 'rb') as f:
    # 加载模型和向量化器
    f_model = pickle.load(f)
model = f_model['model']
tfidf: TfidfVectorizer = f_model['tfidf']
print("模型加载完成")
print(f"词表打印tfidf{tfidf.vocabulary_}")

# todo 3. 数据转数值特征
y_test = tfidf.transform(words)

# todo 4. 模型预测
y_pred = model.predict(y_test)
print(f"预测结果是：{y_pred[0:10]}")

# todo 5. 评估
print(f'准确率:{accuracy_score(labels, y_pred)}')
print(f'精确率:{precision_score(labels, y_pred, average="macro")}')
print(f'召回率:{recall_score(labels, y_pred, average="macro")}')
print(f'f1_score:{f1_score(labels, y_pred, average="macro")}')