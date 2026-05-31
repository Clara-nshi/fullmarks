import os


class Config:
    # 初始化
    def __init__(self):
        # 动态获取项目根目录（config.py 所在目录的上一级）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.dirname(base_dir).replace("\\", "/") + "/"
        # 训练集文件路径
        self.train_datapath = '../data/train.txt'
        # 测试集文件路径
        self.test_datapath = '../data/test.txt'
        # 验证集文件路径
        self.dev_datapath = '../data/dev.txt'
        # 分类文件路径
        self.class_doc_path = '../data/class.txt'
        # 停用词文件路径
        self.stop_words_path = '../data/stopwords.txt'
        self.model_save_path = '../model/train.txt'

        # 处理后的数据路径
        self.process_train_datapath_char = self.root_path + "03_fasttext/final_data/train_process_char.txt"
        self.process_test_datapath_char = self.root_path + "03_fasttext/final_data/test_process_char.txt"
        self.process_dev_datapath_char = self.root_path + "03_fasttext/final_data/dev_process_char.txt"

        # 词级别 fasttext
        self.process_train_datapath_word = self.root_path + "03_fasttext/final_data/train_process_word.txt"
        self.process_test_datapath_word = self.root_path + "03_fasttext/final_data/test_process_word.txt"
        self.process_dev_datapath_word = self.root_path + "03_fasttext/final_data/dev_process_word.txt"

        # 保存模型路径
        self.ft_model_save_path = self.root_path + "03_fasttext/save_models/"
        # 处理完的数据（用于调练）
        self.final_data = self.root_path + "03_fasttext/final_data/"
        # 保存预测结果
        self.id2class_dict = {i: line.strip() for i, line in enumerate(open(self.class_doc_path))}


if __name__ == '__main__':
    import pandas as pd
    conf = Config()
    print(conf.train_datapath)
    print(conf.test_datapath)
    print(conf.dev_datapath)
    print(conf.class_doc_path)
    print(conf.stop_words_path)
    print(conf.model_save_path)
    print(conf.process_train_datapath_char)
    print(conf.process_test_datapath_char)
    print(conf.process_dev_datapath_char)
    print(conf.process_train_datapath_word)
    print(conf.process_test_datapath_word)
    print(conf.process_dev_datapath_word)
    print(conf.ft_model_save_path)
    print(conf.final_data)
    print(conf.id2class_dict)

