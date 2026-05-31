# 导包
import jieba
from config import Config

# 加载配置
conf = Config()


# 定义数据预处理函数
def process_data(data_path, process_path, is_char=True):
    """
    数据预处理函数
    :param data_path: 原始数据路径
    :param process_path: 处理后数据保存路径
    :param is_char: bool 变量，True字符级别的分词，False单词级别的分词
    :return: None
    """
    with open(data_path, encoding='utf-8') as f:
        with open(process_path, 'w', encoding='utf-8') as wf:
            # 遍历每一行
            for line in f.readlines():
                # 去除首尾空格
                line = line.strip()
                if not line:
                    continue
                text, label = line.split('\t')
                if is_char:
                    # 字符级别分词
                    words = ' '.join(list(text))
                else:
                    # 单词级别分词
                    words = ' '.join(jieba.lcut(text))
                # 分类ID 转具体类别名称
                label_str = conf.id2class_dict[int(label)]
                words_line = "__label__"+label_str+" "+words+"\n"
                wf.write(words_line)

    print(f"数据处理完成，保存路径：{process_path}")


if __name__ == '__main__':
    process_data(conf.train_datapath, conf.process_train_datapath_char, is_char=True)
    process_data(conf.test_datapath, conf.process_test_datapath_char, is_char=True)
    process_data(conf.dev_datapath, conf.process_dev_datapath_char, is_char=True)
    process_data(conf.train_datapath, conf.process_train_datapath_word, is_char=False)
    process_data(conf.test_datapath, conf.process_test_datapath_word, is_char=False)
    process_data(conf.dev_datapath, conf.process_dev_datapath_word, is_char=False)



