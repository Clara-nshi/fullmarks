import fasttext
from config import Config


conf = Config()

# 模型训练
model = fasttext.train_supervised(
    input=conf.process_train_datapath_char,
    autotuneValidationFile=conf.process_dev_datapath_char,
    thread=1,
    verbose=3,
    autotuneDuration=300
)

# 模型保存
save_path = conf.ft_model_save_path + "model_char_2_auto"
model.save_model(save_path)
print("模型训练完成")


# 模型预测
pred = model.predict("前 活 塞 传 奇 球 星 重 返 N B A   “ 坏 小 子 ” 加 入 狼 队 教 练 组")
print("预测结果", pred)

# 样本的评估（样本数量、精确率、召回率）
print(model.test(conf.process_test_datapath_char))

# 打印模型关键信息
print(f"模型的词表：{model.words}")
# 打印获取次子
print(f"获取字词：{model.get_subwords("金融公司董事长")}")
# 打印词表的维度
print(f"词表的维度：{model.get_dimension()}")