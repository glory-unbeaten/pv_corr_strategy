import numpy as np
import re
import csv
import os
import pandas as pd
from tqdm import tqdm
csv_path = r'D:\Data\Crypto\resample_data'
hdf5_path = r'D:\Data\Crypto\hdf5\crypto_data_table.h5'

files = os.listdir(csv_path)

# with pd.HDFStore(hdf5_path, mode="w") as store:
#     for filename in tqdm(files):
#         csv_data = pd.read_csv(os.path.join(csv_path, filename))
#         match = re.search(r'_\d{4}-\d{2}-\d{2}', filename)
#         dataset_name = match.group().replace('-', '_')
#         store.put(key=dataset_name, value=csv_data, complevel=0, format='table')

# with pd.HDFStore(hdf5_path, mode="r") as store:
#     # 获取 HDF5 文件中所有数据集的名称
#     dataset_names = store.keys()
#     # 遍历数据集并读取数据

df = pd.read_hdf(path_or_buf=hdf5_path,
                 key='_2021_01_01',
                 columns=['symbol', 'timestamp', 'price'])

df['timestamp'] = pd.to_datetime(df['timestamp'])
tn = pd.to_datetime('2021-01-01').replace(hour=23, minute=59, second=00)
df = df[df.timestamp == tn]
