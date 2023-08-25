import pandas as pd

binance = pd.read_csv(r'D:\Data\Crypto\binance-futures_trades_2023-08-01_BTCUSDT.csv',
                      usecols=['timestamp', 'price', 'amount'])

binance.timestamp = binance.timestamp / 1000  # 转换为毫秒
binance.timestamp = pd.to_datetime(binance.timestamp, unit='ms', errors='coerce')
# errors='coerce'将转换超出范围的时间戳设为NaT（Not a Time）

binance.set_index('timestamp', inplace=True)

resample = binance.price.resample('T').ohlc()
resample_amount = binance.amount.resample('T').sum()

fill_close = 0
for index, row in resample.iterrows():
    if pd.isnull(row['close']):
        resample.loc[index] = fill_close
    else:
        fill_close = row['close']

resample.open = resample.close.shift(1)
resample.insert(loc=resample.shape[-1], column='amount', value=resample_amount)

