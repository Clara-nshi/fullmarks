import fasttext
from config import Config
import datetime

# 获取当前时间
current_time = datetime.time

# 创建配置文件对象
conf = Config()

# 模型训练
model = fasttext.train_supervised(
    input=conf.process_train_datapath_char,
    lr=0.01,
    dim=8,
    minn=1,
    maxn=10
)
print("模型训练完成")

# 模型保存
save_path = conf.ft_model_save_path + 'model_char_1_default.bin'
model.save_model(save_path)
print("模型训练完成")

# 模型预测
pred = model.predict("发 改 委 治 理 涉 企 收 费 每 年 为 企 业 减 负 超 百 亿")
print("预测结果", pred)

# 模型的评估(样本数量、精确率、召回率)
print(model.test(conf.process_test_datapath_char))

# 打印模型关键信息
print(f"模型的词表：{model.words}")
# 打印获取字词
print(f"获取字词：{model.get_subwords("金融公司董事长")}")
# 打印词表的维度
print(f"词表的维度：", model.get_dimension())

