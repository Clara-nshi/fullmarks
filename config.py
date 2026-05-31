import os


class Config():
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
        self.class_path = '../data/class.txt'
        # 停用词文件路径
        self.stop_words_path = '../data/stopwords.txt'
        self.model_save_path = '../model/train.txt'

        # 处理后的数据路径
        self.process_train_datapath = self.root_path + "02_rf/final_data/train_process.csv"
        self.process_test_datapath = self.root_path + "02_rf/final_data/test_process.csv"
        self.process_dev_datapath = self.root_path + "02_rf/final_data/dev_process.csv"

        # 保存模型路径
        self.rf_model_save_path = self.root_path + "02_rf/save_model/rf_model.pkl"
        self.tfidf_model_save_path = self.root_path + "02_rf/save_model/tfidf_model.pkl"
        # 保存预测结果
        self.model_predict_result = self.root_path + "02_rf/result/predict_result.csv"


if __name__ == '__main__':
    import pandas as pd
    conf = Config()
    print(conf.train_datapath)
    print(conf.test_datapath)
    print(conf.dev_datapath)
    print(conf.class_path)
    print(conf.stop_words_path)
    print(conf.model_save_path)
    print(conf.process_train_datapath)
    print(conf.process_test_datapath)
    print(conf.process_dev_datapath)
    print(conf.rf_model_save_path)
    print(conf.tfidf_model_save_path)
    print(conf.model_predict_result)

