import time
import sys

# sys.path.append(r'D:/Quant/PVF')
from ComputeFactorsCounterpart import ComputeFactors, Plot

cf = ComputeFactors(start='2021-01', end='2023-08', group_num=5, freq='week', fee=0.0005)
pt = Plot()
begin = time.time()
quantile_df = cf.quantile(factor_type='pv_corr')
print(round(time.time() - begin, 3))
#%%
ids = cf.index_calculate(factor_type='pv_corr')
#%%
networth_trend_pv_corr = cf.networth_trend_sum(factor_type='pv_corr')
pt.trend_plot(df=networth_trend_pv_corr, title='pv_corr week')


