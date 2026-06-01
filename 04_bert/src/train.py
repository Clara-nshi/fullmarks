import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from config import Config
from utils import build_dataloader
from bert_classifer_model import BertClfModel

# 加载配置对象，包含模型参数、路径等
conf = Config()


# 定义模型调练函数
def model2train():
    """
    Bert模型训练函数，模型保存
    :return:
    """
    # 4个准备
    # todo 1.1 准备数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # todo 1.2 准备模型并发送到设备
    model = BertClfModel().to(conf.device)
    # todo 1.3 准备损失函数， 使用：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # todo 1.4 准备优化器
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)

    # 定义最好的 f1 值，初始值为 0
    best_f1 = 0.0

    # 两个循环
    # todo 2.1 外循环：轮次
    for epoch in range(conf.num_epochs):
        # 设置模型为训练模式
        model.train()
        # 初始化累计损失， 初始化训练集和真实标签
        total_loss = 0.0
        train_preds, train_labels = [], []
        # todo 2.2 内循环：批次
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"训练集训练中...")):
            # 获取数据，批次数据 input_idx, attention_mask, labels
            input_idx, attention_mask, token_type_ids, labels = batch

            # 把数据传输到对应的设备
            input_idx = input_idx.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)
            # 五个关键核心
            # todo 3.1 前向传播 获取预测结果
            logits = model(input_idx, attention_mask)
            # todo 3.2 计算损失
            loss = criterion(logits, labels)
            # 计算累积损失
            total_loss += loss.item()
            # todo 3.3
            optimizer.zero_grad()
            # todo 3.4
            loss.backward()
            # todo 3.5
            optimizer.step()

            # 获取预测
            y_pred_list = torch.argmax(logits, dim=1)

            # 把数据传输到cpu设备上，转为列表，减少GPU内存的占用，防止显存溢出
            train_preds.extend(y_pred_list.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

            # 将10个批次打印一次损失值，同时最后一个批次也会打印数据
            if (i + 1) % 10 == 0 or i == len(train_dataloader) - 1:
                print(f"训练集第{i + 1}批次损失值：{loss.item()}")

                # 计算批次准确率
                train_acc = accuracy_score(train_labels, train_preds)
                # 计算批次f1值
                train_f1 = f1_score(train_labels, train_preds, average="macro")

                # 计算平均损失值
                batch_count = i % 10 + 1
                avg_loss = total_loss / batch_count

                # 清空累计损失和预测和真实标签
                print(f"\nEpoch: {epoch + 1}, Batch: {i + 1}, Loss: {avg_loss:.4f}, acc:{train_acc:.4f}, f1:{train_f1:.4f}")

                total_loss = 0.0
                train_preds, train_labels = [], []    # 存储训练集预测和真实标签


if __name__ == '__main__':
    model2train()






















