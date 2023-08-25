import re
import os
import warnings
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class CFG:
    # read_path = r'D:\Data\Crypto\resample_data'
    read_path = r'D:\resampled_data'
    files = os.listdir(read_path)


def find_first_monday(start: str, end: str) -> Tuple[int, int]:
    first_day = pd.to_datetime(start)
    weekday = first_day.weekday()
    days_to_monday = (7 - weekday) % 7
    first_monday = first_day + pd.Timedelta(days=days_to_monday)
    num_weeks = (pd.to_datetime(end) - first_monday).days // 7
    string = f"binance-futures_trades_{first_monday.strftime('%Y-%m-%d')}_PERPETUALS.csv"
    return CFG.files.index(string), num_weeks


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
    # if df['symbol'].duplicated().any():
    #     print(f'数据存在问题')
    del df['timestamp']
    return df.set_index('symbol').rename(columns={'price': f'{date.group()}'})


def daily_IC(df1: pd.DataFrame, df2: pd.DataFrame, date: str, window: int, rank: bool = False) -> pd.Series:
    date_ = (pd.to_datetime(date) + pd.Timedelta(days=window)).strftime('%Y-%m-%d')
    dt_ = (pd.to_datetime(date) + pd.Timedelta(days=window - 1)).strftime('%Y-%m-%d')
    x, y = df1.loc[:, date: dt_], df2[date_]
    df = pd.concat([x, y], axis=1)
    df.dropna(how='any', axis=0, inplace=True)
    # valid_indices = np.logical_and(~np.isnan(x), ~np.isnan(y))
    # x_, y_ = x[valid_indices], y[valid_indices]
    x_avg, x_std, y_ = df.iloc[:, :-1].mean(axis=1), df.iloc[:, :-1].std(axis=1), df.iloc[:, -1].values
    x_pv_corr = (x_avg - x_avg.mean()) / x_avg.std() + (x_std - x_std.mean()) / x_std.std()
    if rank:
        x_pv_corr = np.argsort(np.argsort(x_pv_corr.values)) + 1
    correlation = pearsonr(x_pv_corr, y_)[0]
    return pd.Series({f'{date_}': correlation})


def ratio(float_: float) -> str:
    return '{:.2%}'.format(float_)


class ComputeFactors:
    def __init__(self, start: str, end: str, group_num: int, freq: str, fee: float):
        self.fee = fee
        self.end = end
        self.start = start
        self.freq = freq
        self.group_num = group_num
        self.dates = pd.date_range(start=start, end=end, freq='D').strftime('%Y-%m-%d')[:-1]
        self.groups = [f'Group{n}' for n in range(1, group_num + 1)]
        self.months = pd.date_range(start=start, end=end, freq='M').strftime('%Y-%m')
        self.first_monday, self.weeks = find_first_monday(start=start, end=end)

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

    def week_calculate(self, week: int, factor_type: str = None, concat: bool = False) -> pd.DataFrame:

        specific_week = CFG.files[self.first_monday + 7 * week: self.first_monday + 7 * (week + 1)]
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(day_calculate, specific_week)), axis=1)
        if concat:
            if factor_type == 'avg':
                return df.mean(axis=1, skipna=True).rename(week, inplace=True)
            if factor_type == 'std':
                return df.std(axis=1, skipna=True, ddof=1).rename(week, inplace=True)
        return df

    @lru_cache()
    def pv_corr_(self, factor_type: str) -> pd.DataFrame:
        if self.freq == 'month':
            with ThreadPoolExecutor() as executor:
                df = pd.concat(list(executor.map(lambda dt: self.month_calculate(timestamp=dt,
                                                                                 concat=True,
                                                                                 factor_type=factor_type),
                                                 self.months)), axis=1)
            return df
        elif self.freq == 'week':
            with ThreadPoolExecutor() as executor:
                df = pd.concat(list(executor.map(lambda wk: self.week_calculate(week=wk,
                                                                                concat=True,
                                                                                factor_type=factor_type),
                                                 range(self.weeks))), axis=1)
            return df

    def pv_corr(self) -> pd.DataFrame:
        pv_corr_avg = self.pv_corr_(factor_type='avg').apply(lambda x: (x - x.mean()) / x.std(),
                                                             axis=1)
        pv_corr_std = self.pv_corr_(factor_type='std').apply(lambda x: (x - x.mean()) / x.std(),
                                                             axis=1)
        return pv_corr_avg + pv_corr_std

    @lru_cache()
    def return_calculate(self):
        if self.freq == 'month':
            specific_day = list(map(lambda date: f'binance-futures_trades_{date}_PERPETUALS.csv', self.dates))
        elif self.freq == 'week':
            specific_day = CFG.files[self.first_monday: self.first_monday + self.weeks * 7]
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(daily_close, specific_day)), axis=1)
        return df.pct_change(axis=1)

    def daily_correlation(self):
        if self.freq == 'month':
            specific_day = list(map(lambda date: f'binance-futures_trades_{date}_PERPETUALS.csv', self.dates))
        elif self.freq == 'week':
            specific_day = CFG.files[self.first_monday: self.first_monday + self.weeks * 7]
        with ThreadPoolExecutor() as executor:
            df = pd.concat(list(executor.map(day_calculate, specific_day)), axis=1)
        return df

    def IC(self, window: int, rank: bool = False):
        dc = self.daily_correlation()
        returns = self.return_calculate()
        if self.freq == 'month':
            with ThreadPoolExecutor() as executor:
                df = pd.concat(
                    list(executor.map(
                        lambda date: daily_IC(df1=dc, df2=returns, date=date, rank=rank, window=window),
                        self.dates[:-window]))
                )
            name = 'RankIC' if rank else 'IC'
            return df.rename(f'{name}', inplace=True)
        elif self.freq == 'week':
            with ThreadPoolExecutor() as executor:
                df = pd.concat(
                    list(executor.map(
                        lambda date: daily_IC(df1=dc, df2=returns, date=date, rank=rank, window=window),
                        dc.columns[:-window]))
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

    def group_calculate_month_sum(self, timestamp: str, factor_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        returns = self.return_calculate()
        dates = (pd.date_range(start=timestamp,
                               end=pd.to_datetime(timestamp) + pd.DateOffset(months=1),
                               freq='D')).strftime('%Y-%m-%d').tolist()
        return_month = returns.loc[:, dates[:-1]]
        quantile_month = (pd.to_datetime(timestamp) - pd.DateOffset(months=1)).strftime('%Y-%m')
        quantile_df = self.quantile(factor_type=factor_type)
        df = quantile_df.loc[:, [quantile_month, timestamp]]
        df.dropna(how='any', axis=0, inplace=True)

        def group_(group: str) -> pd.Series:
            common_index = df[df[quantile_month] == group].index
            return_group = return_month.loc[common_index, :]
            return_group.iloc[:, [0, -1]] = return_group.iloc[:, [0, -1]] - self.fee
            cumulate_return = return_group.cumsum(axis=1, skipna=True).mean(axis=0)
            return cumulate_return.rename(group, inplace=True)

        def group5() -> pd.Series:
            common_index = df[df[quantile_month] == self.groups[-1]].index
            return_group = return_month.loc[common_index, :]
            daily_return = return_group.cumsum(axis=1, skipna=True)
            return_group.iloc[:, [0, -1]] = return_group.iloc[:, [0, -1]] + self.fee
            cumulate_return = daily_return.mean(axis=0)
            return cumulate_return

        with ThreadPoolExecutor() as executor:
            return_groups = pd.concat(
                list(executor.map(group_, self.groups)),
                axis=1
            )
        return return_groups, group5()

    def group_calculate_week_sum(self, week: int, factor_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        returns = self.return_calculate()
        return_week = returns.iloc[:, week * 7: (week + 1) * 7]
        quantile_df = self.quantile(factor_type=factor_type)
        df = quantile_df.loc[:, [week - 1, week]]
        df.dropna(how='any', axis=0, inplace=True)

        # def group_(group: str) -> pd.Series:
        #     common_index = df[df[week - 1] == group].index
        #     return_group = return_week.loc[common_index, :]
        #     return_group.iloc[:, [0, -1]] = return_group.iloc[:, [0, -1]] - self.fee
        #     cumulate_return = return_group.cumsum(axis=1, skipna=True).mean(axis=0)
        #     return cumulate_return.rename(group, inplace=True)

        def group_(group: str) -> pd.Series:
            common_index = df[df[week - 1] == group].index
            return_group = return_week.loc[common_index, :]

            group_idx = int(group[-1]) - 1  # Extract numeric index from the string
            prev_group = self.groups[group_idx - 1] if group_idx - 1 >= 0 else None
            next_group = self.groups[group_idx + 1] if group_idx + 1 < len(self.groups) else None

            if prev_group is not None and next_group is not None:
                if df.loc[common_index, week - 2] == prev_group and df.loc[common_index, week] == next_group:
                    cumulate_return = return_group.cumsum(axis=1, skipna=True).mean(axis=0)
                else:
                    return_group.iloc[:, [0, -1]] = return_group.iloc[:, [0, -1]] - self.fee
                    cumulate_return = return_group.cumsum(axis=1, skipna=True).mean(axis=0)
            else:
                return_group.iloc[:, [0, -1]] = return_group.iloc[:, [0, -1]] - self.fee
                cumulate_return = return_group.cumsum(axis=1, skipna=True).mean(axis=0)
            return cumulate_return.rename(group)

        def group5() -> pd.Series:
            common_index = df[df[week - 1] == self.groups[-1]].index
            return_group = return_week.loc[common_index, :]


            return_group.iloc[:, [0, -1]] = return_group.iloc[:, [0, -1]] + self.fee
            cumulate_return = return_group.cumsum(axis=1, skipna=True).mean(axis=0)
            return cumulate_return

        with ThreadPoolExecutor() as executor:
            return_groups = pd.concat(
                list(executor.map(group_, self.groups)),
                axis=1
            )
        return return_groups, group5()

    @lru_cache()
    def networth_trend_sum(self, factor_type: str):
        swap_assets = [0] * (self.group_num + 1)
        res = []
        if self.freq == 'month':
            for month in tqdm(self.months[1:]):
                df, hedge_group5 = self.group_calculate_month_sum(timestamp=month, factor_type=factor_type)
                ls = df[self.groups[0]] - hedge_group5
                df.insert(column='Group1 hedges Group5', loc=self.group_num, value=ls.values)
                df = df.add(swap_assets, axis=1)
                swap_assets = df.iloc[-1, :].tolist()
                res.append(df)
        elif self.freq == 'week':
            for week in trange(1, self.weeks):
                df, hedge_group5 = self.group_calculate_week_sum(week=week, factor_type=factor_type)
                ls = df[self.groups[0]] - hedge_group5
                df.insert(column='Group1 hedges Group5', loc=self.group_num, value=ls.values)
                df = df.add(swap_assets, axis=1)
                swap_assets = df.iloc[-1, :].tolist()
                res.append(df)
        groups_trend = pd.concat(res, axis=0)
        return groups_trend

    def group_calculate_month_prod(self, timestamp: str, factor_type: str) -> pd.DataFrame:
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

    def group_calculate_week_prod(self, week: int, factor_type: str) -> pd.DataFrame:
        returns = self.return_calculate()
        return_week = returns.iloc[:, week * 7: (week + 1) * 7]
        quantile_df = self.quantile(factor_type=factor_type)
        df = quantile_df.loc[:, [week - 1, week]]
        df.dropna(how='any', axis=0, inplace=True)

        def group_(group: str) -> pd.DataFrame:
            common_index = df[df[week - 1] == group].index
            return_group = return_week.loc[common_index, :]
            cumulate_return = (return_group + 1).cumprod(axis=1).mean(axis=0)
            return cumulate_return.rename(group, inplace=True)

        with ThreadPoolExecutor() as executor:
            return_groups = pd.concat(
                list(executor.map(group_, self.groups)),
                axis=1
            )
        return return_groups

    def networth_trend_prod(self, factor_type: str):
        swap_assets = [1] * (self.group_num + 1)
        res = []
        if self.freq == 'month':
            for month in tqdm(self.months[1:]):
                df = self.group_calculate_month_prod(timestamp=month, factor_type=factor_type)
                ls = df[self.groups[0]] - df[self.groups[-1]] + 1
                df.insert(column='Group1 hedges Group5', loc=self.group_num, value=ls.values)
                df = df.mul(swap_assets, axis=1)
                swap_assets = df.iloc[-1, :].tolist()
                res.append(df)
        elif self.freq == 'week':
            for week in trange(1, self.weeks):
                df = self.group_calculate_week_prod(week=week, factor_type=factor_type)
                ls = df[self.groups[0]] - df[self.groups[-1]] + 1
                df.insert(column='Group1 hedges Group5', loc=self.group_num, value=ls.values)
                df = df.mul(swap_assets, axis=1)
                swap_assets = df.iloc[-1, :].tolist()
                res.append(df)
        groups_trend = pd.concat(res, axis=0)
        return groups_trend

    def hedges_return_week(self, week: int, factor_type: str):
        returns = self.return_calculate()
        return_week = returns.iloc[:, week * 7: (week + 1) * 7]
        quantile_df = self.quantile(factor_type=factor_type)
        df = quantile_df.loc[:, [week - 1, week]]
        df.dropna(how='any', axis=0, inplace=True)

        def group1() -> pd.Series:
            common_index = df[df[week - 1] == self.groups[0]].index
            return_group = return_week.loc[common_index, :]
            return return_group.mean(axis=0)

        def group5() -> pd.Series:
            common_index = df[df[week - 1] == self.groups[-1]].index
            return_group = return_week.loc[common_index, :]
            return return_group.mean(axis=0)

        return group1() - group5()

    @lru_cache()
    def index_calculate(self, factor_type: str) -> pd.Series:

        net_value = self.networth_trend_sum(factor_type=factor_type)['Group1 hedges Group5']
        final_portfolio = net_value.iloc[-1]
        annualized_return = final_portfolio * (365 / len(net_value))
        with ThreadPoolExecutor() as executor:
            hedges_return = pd.concat(list(executor.map(
                lambda wk: self.hedges_return_week(week=wk, factor_type=factor_type), range(1, self.weeks))),
                axis=0)
        annualized_std = np.sqrt(np.var(hedges_return.values, ddof=0) * 365)
        sharpe = (annualized_return - 0.03) / annualized_std
        rolling_max = net_value.expanding(min_periods=1).max()
        draw_down = np.abs(net_value - rolling_max) / rolling_max
        maximum_draw_down = draw_down.max()
        calmar = annualized_return / maximum_draw_down
        wins = net_value.diff(periods=1)
        wins.iloc[0] = net_value.iloc[0]
        win_rate = (wins > 0).mean()

        ids = pd.Series(
            {
                'final_portfolio': round(final_portfolio, 2),
                'annualized_return': ratio(annualized_return),
                'annualized_std': ratio(annualized_std),
                'sharpe': round(sharpe, 2),
                'maximum_draw_down': ratio(maximum_draw_down),
                'calmar': round(calmar, 2),
                'win_rate': ratio(win_rate),
            }
        )
        return ids


class Plot:
    @staticmethod
    def boxplot(df: pd.Series, y_label: str, title: str, threshold: float):
        ratio_ = (df.abs() > threshold).mean()
        plt.figure(figsize=(13, 5), dpi=540)
        plt.boxplot(df, showmeans=True)
        plt.ylabel(f'{y_label}',
                   fontsize=15,
                   fontweight='bold'
                   )
        plt.title(label=f'{title} Ratio(abs>{threshold}): {round(ratio_ * 100, 1)}%',
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
        ax.set_ylabel('NetValue',
                      fontsize=15,
                      fontweight='bold'
                      )
        ax.set_title(label=f'Trend of net value ({title})',
                     fontsize=15,
                     fontweight='bold'
                     )
        plt.show()
