# 导包
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import Config
from tqdm import tqdm

pd.set_option('display.expand_frame_repr', False)   # 禁止DataFrame在输出时自动换行
pd.set_option('display.max_columns', None)          # 取消DaraFrame输出时的最大列数限制

conf = Config()

# todo 1 读取数据
df = pd.read_csv(conf.process_train_datapath, sep='\t', names=['text', 'label', 'words'])
words = df['words'][:20000]
labels = df['label'][:20000]

print(f"words:{words[:10]}")
print(f"labels:{labels[:10]}")

# todo 2 预处理 数值特征转换
# 读取停用词文件
stop_words = open(conf.stop_words_path, encoding="utf-8").read().split()
# 构建TF-IDF对象
tfidf = TfidfVectorizer(stop_words=stop_words)
feature = tfidf.fit_transform(words)
# print(f"特征维度：{feature.shape}")
# print(f"tfidf词表：{tfidf.get_feature_names_out()[:10]}")
# print(f"特征矩阵：{feature[:10]}")

# todo 3 划分数据集
# 划分数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=42)

# todo 4 构建模型对象
model = RandomForestClassifier()

# todo 5 模型训练
for _ in tqdm(range(1)):
    model.fit(x_train, y_train)
print("模型训练完成")

# todo 6 模型预测
y_pred = model.predict(x_test)
print(f'准确率:{accuracy_score(y_test, y_pred)}')
print(f'精确率:{precision_score(y_test, y_pred, average="macro")}')
print(f'召回率:{recall_score(y_test, y_pred, average="macro")}')
print(f'f1_score:{f1_score(y_test, y_pred, average="macro")}')
# todo 7 模型评估


# todo 8 保存模型
with open(conf.rf_model_save_path, 'wb') as f:
    pickle.dump({'model': model, 'tfidf': tfidf}, f)

print("模型保存完成")
