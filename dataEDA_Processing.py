import pandas as pd
from config import Config
import jieba

# 初始化配置文件
conf = Config()

# data_path = conf.train_datapath
# data_path = conf.dev_datapath
data_path = conf.test_datapath

# todo 1 获取数据路径
data = pd.read_csv(data_path, sep='\t', names=['text', 'label'])

# todo 2 读取数据路径
print(data.head())


# todo 3 分词
# 使用jieba分词
def split_sentence(s):
    return ' '.join(jieba.lcut(s)[:30])


data['words'] = data['text'].apply(split_sentence)
print(data[['text', 'words']].head())

# todo 4 保存处理后的数据
if "train" in data_path:
    data.to_csv(conf.process_train_datapath, sep='\t', index=False)
    print("保存训练数据成功！")
if "test" in data_path:
    data.to_csv(conf.process_test_datapath, sep='\t', index=False)
    print("保存测试数据成功！")
if "dev" in data_path:
    data.to_csv(conf.process_dev_datapath, sep='\t', index=False)
    print("保存预测数据成功！")



