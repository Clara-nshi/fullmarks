import torch  # 深度学习框架
from torch.utils.data import Dataset, DataLoader  # 数据集、数据加载器
from transformers import BertTokenizer  # BERT分词器
from tqdm import tqdm  # 进度条
import time  # 时间模块
from datetime import timedelta  # 时间差
from config import Config  # 配置文件

conf = Config()


def load_raw_data(file_path):
    """
    加载原始数据
    :param file_path: 数据文件路径
    :return: 数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc="加载原始数据"):
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')
            data.append((text, int(label)))
    return data


class TextDataset(Dataset):
    def __init__(self, data):
        # data 数据：（text, label）
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][0]
        label = self.data[idx][1]
        return text, label


# 创建整理函数
def collate_fn(batch):
    """
    整理函数: 将每一批次数据整理成模型需要的格式：转向量
    :param batch:
    :return: input_ids, attention_mask, labels  数据类型是张量类型
    """
    # 获取文本和标签，通过zip函数解包获取数据
    print(f"collate_fn: {type(batch)}{type(batch[0])}{len(batch)}")
    texts, labels = zip(*batch)     # text, label = [item()]

    # 分词
    tokens = conf.tokenizer.batch_encode_plus(
        texts,                              # 文本数据
        max_length=conf.pad_size,           # 最大长度
        padding='max_length',               # 填充策略
        truncation=True,                    # 截断策略
        return_attention_mask=True,         # 返回注意力掩码
        return_token_type_ids=True,         # 返回token类型ID
        return_tensors='pt'
    )
    # 获取输入向量、注意力掩码、标签
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    token_type_ids = tokens['token_type_ids']
    labels = torch.tensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


# 定义数据加载器
def build_dataloader():
    """
    构建数据加载器

    :return: 训练集、验证集、测试集的数据加载器
    """
    print("开始加载数据")
    train_data = load_raw_data(conf.train_datapath)
    test_data = load_raw_data(conf.test_datapath)
    dev_data = load_raw_data(conf.dev_datapath)
    print("数据加载完成")
    train_data_list = TextDataset(train_data)
    test_data_list = TextDataset(test_data)
    dev_data_list = TextDataset(dev_data)
    print("数据集构建完成")
    train_dataloader = DataLoader(train_data_list, batch_size=conf.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data_list, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_data_list, batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn)
    print("数据加载器构建完成")
    return train_dataloader, test_dataloader, dev_dataloader


if __name__ == '__main__':
    # # 加载原始数据
    # data_list = load_raw_data(conf.train_datapath)
    # print(data_list[:10], len(data_list), type(data_list), type(data_list[0]))
    #
    # # 创建数据集
    # dataset = TextDataset(data_list)
    # print(dataset[0])
    # print(dataset[1])
    # print(type(dataset))
    # print(type(dataset[0]))

    # 测试1
    # 获取数据加载器
    train_loader, test_loader, dev_loader = build_dataloader()
    print(f"训练集数据加载器：{train_loader}")
    print(f"测试集数据加载器：{test_loader}")
    print(f"验证集数据加载器：{dev_loader}")

    print(f"train_loader: {type(train_loader)}")
    print(f"test_loader: {type(test_loader)}")
    print(f"dev_loader: {type(dev_loader)}")

    for i, batch in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, labels = batch
        print(f"第{i}批次数据：{input_ids.shape}{attention_mask.shape}{token_type_ids.shape}{labels.shape}")
        print(f"input_ids: {input_ids}")
        print(f"attention_mask: {attention_mask}")
        print(f"token_type_ids: {token_type_ids}")
        print(f"labels: {labels}")
        break
