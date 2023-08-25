import os
import gc
import re
import time

import pandas as pd
import threading


class CFG:
    read_path = r'D:\Data\Crypto\read_data'
    gz_files = os.listdir(read_path)
    save_path = r'D:\Data\Crypto\resample_data'
    num_threads = 8


# def resample_thread(files):
#     for file in files:
#         resample_save(file)


def resample_save(file: str):
    """
        对CFG.read_path下的多个csv文件进行处理并保存至CFG.save_path
    """
    begin = time.time()
    df = pd.read_csv(os.path.join(CFG.read_path, file))
    df['timestamp'] = pd.to_datetime(df['timestamp'] // 1000, unit='ms')
    df.set_index('timestamp', inplace=True)
    resampled_data = df.groupby('symbol').resample('1T').agg({'price': 'last', 'amount': 'sum'})
    resampled_data.dropna(inplace=True)
    resampled_data.to_csv(os.path.join(CFG.save_path, file.replace('.gz', '')), index=True)
    date = re.search(r'\d{4}-\d{2}-\d{2}', file)
    print(f'Accomplished {date.group()} {round(time.time() - begin, 3)}s!')
    gc.collect()


# threaded_files = [CFG.gz_files[i::CFG.num_threads] for i in range(CFG.num_threads)]
for gz_file in CFG.gz_files:
    resample_save(file=gz_file)

# threads = []
# for files in threaded_files:
#     thread = threading.Thread(target=resample_thread, args=(files,))
#     threads.append(thread)
#     thread.start()
#
# for thread in threads:
#     thread.join()
#
# print("All threads are done!")
