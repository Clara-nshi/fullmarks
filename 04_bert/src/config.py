import torch
from transformers import BertModel, BertTokenizer, BertConfig

# 自动检测并选择可用设备
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert"  # 模型名称
        # 路径
        self.root_path = 'D:/clara_nshi/python/fullmarks/'
        # 原始数据路径
        self.train_datapath = self.root_path + 'data/train.txt'
        self.test_datapath = self.root_path + 'data/test.txt'
        self.dev_datapath = self.root_path + 'data/dev.txt'
        # 类别文档
        self.class_path = self.root_path + "data/class.txt"

        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        # 模型训练保存路径
        self.model_save_path = self.root_path + "04-bert/save_models/test_bertclassifer_model.pt"  # 模型训练结果保存路径

        # 模型训练+预测的时候（自动选择GPU或CPU）
        self.device = DEVICE

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "C:/Users/acer/.cache/modelscope/hub/models/google-bert/bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path).to(self.device)  # 加载预训练BERT模型并移至设备
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # BERT模型的配置

        # 量化模型存放地址
        self.bert_model_quantization_model_path = self.root_path + "04-bert/save_models/test_bertclassifer_quantization_model.pt"  # 模型训练结果保存路径


if __name__ == '__main__':
    conf = Config()
    print(conf.bert_model_quantization_model_path)
    # 打印类别名单
    print(conf.train_datapath)
    # 打印类别名单
    print(conf.class_list)
    # 打印模型保存路径
    print(conf.model_save_path)
    # 打印设备
    print(conf.device)
    # 测试2
    # 打印模型参数
    print(conf.bert_model)
    print(conf.tokenizer)
    print(conf.bert_config)
