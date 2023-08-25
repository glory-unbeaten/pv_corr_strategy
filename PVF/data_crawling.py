import nest_asyncio

nest_asyncio.apply()
from tardis_dev import datasets

api_key = 'TD.L9M2r84pFJlqi-Te.vbmdgRxOh7mmRW1.yvv8sfStcf1bF84.Di8MM3XFos4z6oP.UQtLGISKg2zwOob.X40S'
from_date = '2021-01-01'
to_date = '2022-01-01'

datasets.download(
    exchange='binance-futures',
    data_types=['trades'],
    from_date=from_date,
    to_date=to_date,
    symbols=['PERPETUALS'],
    api_key=api_key,
    download_dir=rf'D:/Data/Crypto/binance',
)

print('Accomplished 2021')

from_date = '2023-01-01'
to_date = '2023-08-01'

datasets.download(
    exchange='binance-futures',
    data_types=['trades'],
    from_date=from_date,
    to_date=to_date,
    symbols=['PERPETUALS'],
    api_key=api_key,
    download_dir=rf'D:/Data/Crypto/binance',
)

print('Accomplished 2023')

