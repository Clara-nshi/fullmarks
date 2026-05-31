# 导包
import pandas as pd

from config import Config
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# 初始化配置
conf = Config()

# 加载数据