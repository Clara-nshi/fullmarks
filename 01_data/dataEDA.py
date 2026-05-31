# """
# dataEDA: 探索性数据分析：文本长度、标签情况...
# """
# # 导包
# import pandas as pd
# from collections import Counter
#
# from config import Config  # 导入配置类
#
# # 创建配置类对象
# conf = Config()
#
# # todo 1 读取数据
# data = pd.read_csv(conf.train_datapath, sep='\t', names=['text', 'label'])
#
#
# # todo 2 打印前10行数据
# # print(data.head(10))
#
#
# # todo 3 新增一列数据，统计数据文本的长度
# data['text_len'] = data['text'].str.len()
# print("数据文本的长度: ", data.head())
#
# # todo 4. 统计文本长度的分布情况
# print("文本长度的分布情况: ", data['text_len'].describe())
#
# # 打印文本长度的平均值
# print("文本长度的平均值: ", data['text_len'].mean())
#
# # 打印文本长度的标准差
# print("文本长度的标准差: ", data['text_len'].std())
#
# # 打印文本长度的最大值
# print("文本长度的最大值: ", data['text_len'].max())
#
# # 打印文本长度的最小值
# print("文本长度的最小值: ", data['text_len'].min())
#
#
# # 第二步：统计标签分布
# label_counts = Counter(data['label'])
# print("\n标签分布：")
# for label, count in label_counts.items():
#     print(f"标签{label}： {count}次")
#
# # 第三步：计算标签比例
# total_rows = len(data)  # 总行数
# print("\n标签分布：")
# for label, count in label_counts.items():
#     percent = (count / total_rows) * 100   # 计算百分比
#     print(f"标签{label}： {percent:.2f}%")  # 输出百分比，保留2位小数
#

# 导包
import pandas as pd
from config import Config
from collections import Counter

conf = Config()
data = pd.read_csv(conf.train_datapath, sep='\t', names=['text', 'labels'])
#
# # 查看数据
# print(data.head())

# 统计文本长度
data['text_len'] = data['text'].str.len()
print(data.head())

print("文本长度的分布情况", data.describe())

print("文本的平均值：", data['text_len'].mean())
print("文本的标准差：", data['text_len'].std())
print("文本的最大值：", data['text_len'].max())
print("文本的最小值：", data['text_len'].min())

# 统计标签分布
labels_counter = Counter(data['labels'])
for labels, count in labels_counter.items():
    print(f"标签：{labels}, 次数{count}次")



# 统计标签比例
total_rows = len(data)
for labels, count in labels_counter.items():
    percent = (count/total_rows) * 100
    print(f"标签：{labels}, {percent:.2f}%")


