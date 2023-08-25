import re
import os
import warnings
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class CFG:
    read_path = r'D:\Data\Crypto\resample_data'
    files = os.listdir(read_path)


def correlation_calculate(group: pd.DataFrame) -> pd.Series:
    correlation_coefficient, _ = pearsonr(group['price'], group['amount'])  # _为P-value
    return pd.Series({'correlation_coefficient': correlation_coefficient})


@lru_cache()
def day_calculate(file: str) -> pd.Series:
    df = pd.read_csv(os.path.join(CFG.read_path, file), usecols=['symbol', 'price', 'amount'])
    correlation_results = df.groupby('symbol').apply(correlation_calculate).reset_index()
    date = re.search(r'\d{4}-\d{2}-\d{2}', file)
    return correlation_results.set_index('symbol').rename(columns={'correlation_coefficient': f'{date.group()}'})


def daily_close(file: str) -> pd.Series:
    df = pd.read_csv(os.path.join(CFG.read_path, file), usecols=['symbol', 'timestamp', 'price'])
    date = re.search(r'\d{4}-\d{2}-\d{2}', file)
    tn = pd.to_datetime(date.group()).replace(hour=23, minute=59, second=00)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df.timestamp == tn]
    if df['symbol'].duplicated().any():
        print(f'数据存在问题')
    del df['timestamp']
    return df.set_index('symbol').rename(columns={'price': f'{date.group()}'})


def daily_IC(df1: pd.DataFrame, df2: pd.DataFrame, date: str, rank: bool = False) -> pd.Series:
    date_ = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    x, y = df1[date].values, df2[date_].values
    valid_indices = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x_, y_ = x[valid_indices], y[valid_indices]
    if rank:
        x_ = np.argsort(np.argsort(x_)) + 1
    correlation = pearsonr(x_, y_)[0]
    return pd.Series({f'{date_}': correlation})


class ComputeFactors:
    def __init__(self, start: str, end: str, group_num: int):
        self.end = end
        self.start = start
        self.group_num = group_num
        self.dates = pd.date_range(start=start, end=end, freq='D').strftime('%Y-%m-%d')[:-1]
        self.groups = [f'Group{n}' for n in range(1, group_num + 1)]
        self.months = pd.date_range(start=start, end=end, freq='M').strftime('%Y-%m')

    @staticmethod
    def month_calculate(timestamp: str, factor_type: str = None, concat: bool = False) -> pd.DataFrame:
        specific_month = [file for file in CFG.files if file.startswith(f'binance-futures_trades_{timestamp}')]
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(day_calculate, specific_month)), axis=1)
        if concat:
            if factor_type == 'avg':
                return df.mean(axis=1, skipna=True).rename(timestamp, inplace=True)
            if factor_type == 'std':
                return df.std(axis=1, skipna=True, ddof=1).rename(timestamp, inplace=True)
        return df

    @lru_cache()
    def pv_corr_(self, factor_type: str) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(lambda dt: self.month_calculate(timestamp=dt,
                                                                             concat=True,
                                                                             factor_type=factor_type),
                                             self.months)), axis=1)
        return df

    def pv_corr(self) -> pd.DataFrame:
        pv_corr_avg = self.pv_corr_(factor_type='avg').apply(lambda x: (x - x.mean()) / x.std(),
                                                             axis=1)
        pv_corr_std = self.pv_corr_(factor_type='std').apply(lambda x: (x - x.mean()) / x.std(),
                                                             axis=1)
        return pv_corr_avg + pv_corr_std

    @lru_cache()
    def return_calculate(self):
        specific_day = list(map(lambda date: f'binance-futures_trades_{date}_PERPETUALS.csv', self.dates))
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(daily_close, specific_day)), axis=1)
        returns = df.pct_change(axis=1)
        return returns

    def daily_correlation(self):
        specific_day = list(map(lambda date: f'binance-futures_trades_{date}_PERPETUALS.csv', self.dates))
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(day_calculate, specific_day)), axis=1)
        return df

    def IC(self, rank: bool = False):
        dc = self.daily_correlation()
        returns = self.return_calculate()
        with ThreadPoolExecutor() as executor:
            df = pd.concat(
                list(executor.map(
                    lambda date: daily_IC(df1=dc, df2=returns, date=date, rank=rank),
                    self.dates[:-1]))
            )
        name = 'RankIC' if rank else 'IC'
        return df.rename(f'{name}', inplace=True)

    @lru_cache()
    def quantile(self, factor_type: str) -> pd.DataFrame:

        def group_by_quantiles(column):
            return pd.qcut(column, q=self.group_num, labels=self.groups, duplicates='drop')

        if factor_type == 'pv_corr':
            grouped_df = self.pv_corr().apply(group_by_quantiles, axis=0)
        else:
            grouped_df = self.pv_corr_(factor_type=factor_type).apply(group_by_quantiles, axis=0)
        return grouped_df

    def group_calculate(self, timestamp: str, factor_type: str) -> pd.DataFrame:
        returns = self.return_calculate()
        dates = (pd.date_range(start=timestamp,
                               end=pd.to_datetime(timestamp) + pd.DateOffset(months=1),
                               freq='D')).strftime('%Y-%m-%d').tolist()
        return_month = returns.loc[:, dates[:-1]]
        quantile_month = (pd.to_datetime(timestamp) - pd.DateOffset(months=1)).strftime('%Y-%m')
        quantile_df = self.quantile(factor_type=factor_type)
        df = quantile_df.loc[:, [quantile_month, timestamp]]
        df.dropna(how='any', axis=0, inplace=True)

        def group_(group: str) -> pd.DataFrame:
            common_index = df[df[quantile_month] == group].index
            return_group = return_month.loc[common_index, :]
            cumulate_return = (return_group + 1).cumprod(axis=1).mean(axis=0)
            return cumulate_return.rename(group, inplace=True)

        with ThreadPoolExecutor() as executor:
            return_groups = pd.concat(
                list(executor.map(group_, self.groups)),
                axis=1
            )
        return return_groups

    def networth_trend(self, factor_type: str):
        swap_assets = [1] * self.group_num
        res = []
        for month in tqdm(self.months[1:]):
            df = self.group_calculate(timestamp=month, factor_type=factor_type).mul(swap_assets, axis=1)
            swap_assets = df.iloc[-1, :].tolist()
            res.append(df)
        groups_trend = pd.concat(res, axis=0)
        return groups_trend


class Plot:
    @staticmethod
    def boxplot(df: pd.Series, y_label: str, title: str, threshold: float):
        ratio = (df.abs() > threshold).mean()
        plt.figure(figsize=(13, 5), dpi=540)
        plt.boxplot(df, showmeans=True)
        plt.ylabel(f'{y_label}',
                   fontsize=15,
                   fontweight='bold'
                   )
        plt.title(label=f'{title} Ratio(abs>{threshold}): {round(ratio * 100, 1)}%',
                  fontsize=15,
                  fontweight='bold'
                  )
        plt.axhspan(ymin=-threshold, ymax=threshold, facecolor='purple', alpha=0.3)
        plt.axhline(y=0.0, c='red')
        plt.show()

    @staticmethod
    def scatterplot(df: pd.Series, y_label: str, title: str, threshold: float):
        ratio = (df.abs() > threshold).mean()
        plt.figure(figsize=(13, 5), dpi=540)
        plt.scatter(y=df.values, x=pd.to_datetime(df.index))
        plt.ylabel(f'{y_label}',
                   fontsize=15,
                   fontweight='bold'
                   )
        plt.title(label=f'{title} Ratio(abs>{threshold}): {round(ratio * 100, 1)}%',
                  fontsize=15,
                  fontweight='bold'
                  )
        plt.axhspan(ymin=-threshold, ymax=threshold, facecolor='purple', alpha=0.3)
        plt.axhline(y=0.0, c='red')
        plt.show()

    @staticmethod
    def trend_plot(df: pd.DataFrame, title: str):
        ax = df.plot()
        ax.figure.set_size_inches(13, 5)
        ax.figure.set_dpi(540)
        ax.legend(loc='best')
        ax.set_ylabel('NetWorth',
                      fontsize=15,
                      fontweight='bold'
                      )
        ax.set_title(label=f'Trend of net worth ({title})',
                     fontsize=15,
                     fontweight='bold'
                     )
        plt.show()
