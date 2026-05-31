# 数据处理与持久化
import pandas as pd     # 表格数据处理
import pickle           # 对象序列化/反序列化
# 特征提取
from sklearn.feature_extraction.text import TfidfVectorizer     # 文本转向量（TF-IDF）
# 数据拆分
from sklearn.model_selection import train_test_split            # 划分训练/测试集
# 分类模型
from sklearn.ensemble import RandomForestClassifier             # 随机森林分类器
# 性能评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 自定义
from config import Config
# 进度可视化
from tqdm import tqdm

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)

# 初始化
conf = Config()

# todo 1 读取数据
df = pd.read_csv(conf.process_train_datapath, sep='\t', names=['text', 'label', 'words'])
words = df['words']
labels = df['label']

# todo 2 数据预处理  数值特征转换
# 读取停用词
stop_words = open(conf.stop_words_path, encoding='utf-8').read().split()
# 构建TF-IDF对象
tfidf = TfidfVectorizer(stop_words=stop_words)
feature = tfidf.fit_transform(words)

# todo 3 l
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=42)

# todo 4 构建模型对象
model = RandomForestClassifier()

# todo 5 模型训练
for _ in tqdm(range(1)):
    model.fit(x_train, y_train)
print("模型训练完成")

# todo 6 模型预测
y_pred = model.predict(x_test)

# todo 7 模型评估
print(f"准确率：{accuracy_score(y_test, y_pred)}")
print(f"精确率：{precision_score(y_test, y_pred, average="macro")}")
print(f"召回率：{recall_score(y_test, y_pred, average="macro")}")
print(f"f1值：{f1_score(y_test, y_pred, average="macro")}")

# todo 8 保存模型
with open(conf.rf_model_save_path, 'wb') as f:
    pickle.dump({'model': model, 'tfidf': tfidf}, f)

print("模型保存完成")



























