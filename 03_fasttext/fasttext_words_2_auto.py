import fasttext
from config import Config
import datetime

# 创建配置文件对象
conf = Config()

# 模型训练
model = fasttext.train_supervised(
    input=conf.process_train_datapath_word,
    autotuneValidationFile=conf.process_dev_datapath_word,
    thread=1,
    verbose=3,
    autotuneDuration=120
)
print("模型训练完成")

# 模型保存
save_path = conf.ft_model_save_path + 'model_word_2_auto.bin'
model.save_model(save_path)
print("模型训练完成")

# 模型预测
pred = model.predict("去 新西兰 体验 舌尖 上 的 饕餮 之旅 ( 组图 )")
print("预测结果", pred)

# 模型的评估(样本数量、精确率、召回率)
print(model.test(conf.process_test_datapath_char))

# 打印模型关键信息
print(f"模型的词表：{model.words}")
# 打印获取字词
print(f"获取字词：{model.get_subwords("金融公司董事长")}")
# 打印词表的维度
print(f"词表的维度：", model.get_dimension())

