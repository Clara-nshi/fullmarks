import torch
import torch.nn as nn
from transformers import BertModel
from config import Config
from utils import build_dataloader

conf = Config()


# Bert分类模型
class BertClfModel(nn.Module):
    def __init__(self):
        super(BertClfModel, self).__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # desc 增加线形层
        self.fc = nn.Linear(conf.bert_config.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        """

        :param input_ids:
        :param attention_mask:
        :return:
        """
        logistic = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        print(f'aaa', logistic)
        logistic = self.fc(logistic.pooler_output)
        print(f'bbb', logistic)
        return logistic


if __name__ == '__main__':
    text = ['王者荣耀', '今天天气很好']
    tokenizer = conf.tokenizer.batch_encode_plus(
        text,
        max_length=20,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids = torch.tensor(tokenizer['input_ids']).to(conf.device)
    attention_mask = torch.tensor(tokenizer['attention_mask']).to(conf.device)
    # 创建模型对象
    model = BertClfModel()
    model.to(conf.device)

    logis = model.forward(input_ids, attention_mask)
    logis = torch.softmax(logis, dim=1)
    pred = logis.argmax(dim=-1)
    print(conf.class_list[int(pred[0])])
    print(conf.class_list[int(pred[1])])
