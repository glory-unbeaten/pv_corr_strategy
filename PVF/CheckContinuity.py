import re
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class CFG:
    read_path = r'D:\Data\Crypto\resample_data'
    gz_files = os.listdir(read_path)


def check_day(file):
    df = pd.read_csv(os.path.join(CFG.read_path, file), usecols=['symbol'])
    coins = df.squeeze().unique()
    date = re.search(r'\d{4}-\d{2}-\d{2}', file)  # 从文件名中匹配日期
    return pd.Series(data=['True'] * len(coins), index=coins, name=str(date.group()))


with ThreadPoolExecutor() as executor:
    check_results = pd.concat(list(executor.map(check_day, CFG.gz_files)), axis=1)

