# -*- coding: utf-8 -*-
"""
价格-成交量因子家族 (Price-Volume Factors)

包含：close_position, skew20, kurt20
"""

from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd


class PriceVolumeFactors:
    """价格-成交量因子计算类"""
    
    def __init__(self, market_data_path: Optional[str] = None, output_path: Optional[str] = None):
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent.parent
        
        self.market_data_path = Path(market_data_path) if market_data_path else factor_lib_root / 'processed_data' / 'market_data'
        self.output_path = Path(output_path) if output_path else factor_lib_root / 'processed_data' / 'factors' / 'technical'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._cache = {}
        self._dates = None
        self._stocks = None
    
    def _load(self, field: str):
        if field not in self._cache:
            path = self.market_data_path / f'{field}.parquet'
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            self._cache[field] = pq.read_table(path)
        return self._cache[field]
    
    def _to_numpy(self, field='close'):
        table = self._load(field)
        columns = table.column_names
        self._dates = columns[0]
        self._stocks = columns[1:]
        data = [table.column(s).to_pylist() for s in self._stocks]
        matrix = np.array(data, dtype=np.float64).T
        dates = table.column(self._dates).to_pylist()
        return dates, self._stocks, matrix
    
    def _save(self, name, matrix, dates, stocks):
        arrays = [pa.array(dates, type=pa.timestamp('ns'))]
        names = ['time']
        for i, s in enumerate(stocks):
            col = [None if (v != v or np.isinf(v)) else float(v) for v in matrix[:, i]]
            arrays.append(pa.array(col, type=pa.float64()))
            names.append(s)
        output_file = self.output_path / f'{name}.parquet'
        pq.write_table(pa.table(arrays, names=names), output_file)
        return output_file
    
    def factor_close_position(self, save=True):
        """收盘价位置因子 - 向量化实现"""
        print("计算因子: close_position (收盘价位置)")
        dates, stocks, close = self._to_numpy('close')
        _, _, high = self._to_numpy('high')
        _, _, low = self._to_numpy('low')
        
        result = np.full_like(close, np.nan)
        price_range = high - low
        valid = price_range > 0
        result[valid] = (close[valid] - low[valid]) / price_range[valid]
        result = np.where((result >= 0) & (result <= 1), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('close_position', result, dates, stocks)
        return result
    
    def factor_intraday_return_ma5(self, save=True):
        """
        日内收益率5日平滑因子
        
        公式: mean((close - open) / open, 5)
        逻辑: 5日平均日内收益，平滑单日噪声，提取短期日内趋势
        """
        print("计算因子: intraday_return_ma5 (日内收益5日均)")
        dates, stocks, close = self._to_numpy('close')
        _, _, open_price = self._to_numpy('open')
        
        # 计算日内收益率
        intraday = np.full_like(close, np.nan)
        mask = open_price > 0
        intraday[mask] = (close[mask] - open_price[mask]) / open_price[mask]
        
        # 5日滚动平均
        window = 5
        min_periods = 3
        n_dates, n_stocks = intraday.shape
        result = np.full_like(intraday, np.nan)
        
        for i in range(window - 1, n_dates):
            window_data = intraday[i - window + 1:i + 1, :]
            valid_count = np.sum(~np.isnan(window_data), axis=0)
            valid_mask = valid_count >= min_periods
            
            if np.any(valid_mask):
                result[i, valid_mask] = np.nanmean(window_data[:, valid_mask], axis=0)
        
        # 极端值截断
        result = np.where((result > -0.2) & (result < 0.2), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('intraday_return_ma5', result, dates, stocks)
        return result
    
    def factor_intraday_return_ma20(self, save=True):
        """
        日内收益率20日平滑因子
        
        公式: mean((close - open) / open, 20)
        逻辑: 20日平均日内收益，匹配月频调仓周期，反映中期日内趋势
        """
        print("计算因子: intraday_return_ma20 (日内收益20日均)")
        dates, stocks, close = self._to_numpy('close')
        _, _, open_price = self._to_numpy('open')
        
        # 计算日内收益率
        intraday = np.full_like(close, np.nan)
        mask = open_price > 0
        intraday[mask] = (close[mask] - open_price[mask]) / open_price[mask]
        
        # 20日滚动平均
        window = 20
        min_periods = 10
        n_dates, n_stocks = intraday.shape
        result = np.full_like(intraday, np.nan)
        
        for i in range(window - 1, n_dates):
            window_data = intraday[i - window + 1:i + 1, :]
            valid_count = np.sum(~np.isnan(window_data), axis=0)
            valid_mask = valid_count >= min_periods
            
            if np.any(valid_mask):
                result[i, valid_mask] = np.nanmean(window_data[:, valid_mask], axis=0)
        
        # 极端值截断
        result = np.where((result > -0.1) & (result < 0.1), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('intraday_return_ma20', result, dates, stocks)
        return result
    
    def factor_close_position_ma5(self, save=True):
        """
        收盘价位置5日平滑因子
        
        公式: mean((close - low) / (high - low), 5)
        逻辑: 5日平均收盘位置，平滑单日噪声，反映短期强弱趋势
        """
        print("计算因子: close_position_ma5 (收盘位置5日均)")
        dates, stocks, close = self._to_numpy('close')
        _, _, high = self._to_numpy('high')
        _, _, low = self._to_numpy('low')
        
        # 计算当日收盘位置
        close_pos = np.full_like(close, np.nan)
        price_range = high - low
        valid = price_range > 0
        close_pos[valid] = (close[valid] - low[valid]) / price_range[valid]
        
        # 5日滚动平均
        window = 5
        min_periods = 3
        n_dates, n_stocks = close_pos.shape
        result = np.full_like(close_pos, np.nan)
        
        for i in range(window - 1, n_dates):
            window_data = close_pos[i - window + 1:i + 1, :]
            valid_count = np.sum(~np.isnan(window_data), axis=0)
            valid_mask = valid_count >= min_periods
            
            if np.any(valid_mask):
                result[i, valid_mask] = np.nanmean(window_data[:, valid_mask], axis=0)
        
        # 极端值截断（平滑后应该在0-1之间，但允许轻微超出）
        result = np.where((result > -0.2) & (result < 1.2), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('close_position_ma5', result, dates, stocks)
        return result
    
    def factor_close_position_ma20(self, save=True):
        """
        收盘价位置20日平滑因子
        
        公式: mean((close - low) / (high - low), 20)
        逻辑: 20日平均收盘位置，匹配月频调仓周期，反映中期强弱趋势
        """
        print("计算因子: close_position_ma20 (收盘位置20日均)")
        dates, stocks, close = self._to_numpy('close')
        _, _, high = self._to_numpy('high')
        _, _, low = self._to_numpy('low')
        
        # 计算当日收盘位置
        close_pos = np.full_like(close, np.nan)
        price_range = high - low
        valid = price_range > 0
        close_pos[valid] = (close[valid] - low[valid]) / price_range[valid]
        
        # 20日滚动平均
        window = 20
        min_periods = 10
        n_dates, n_stocks = close_pos.shape
        result = np.full_like(close_pos, np.nan)
        
        for i in range(window - 1, n_dates):
            window_data = close_pos[i - window + 1:i + 1, :]
            valid_count = np.sum(~np.isnan(window_data), axis=0)
            valid_mask = valid_count >= min_periods
            
            if np.any(valid_mask):
                result[i, valid_mask] = np.nanmean(window_data[:, valid_mask], axis=0)
        
        # 极端值截断
        result = np.where((result > -0.1) & (result < 1.1), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('close_position_ma20', result, dates, stocks)
        return result
    
    def factor_skew20(self, save=True):
        """收益率偏度因子 (20日) - NumPy向量化实现"""
        print("计算因子: skew20 (20日收益偏度)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        
        # 计算 ret1
        ret1 = np.full_like(close, np.nan)
        mask = pre_close > 0
        ret1[mask] = close[mask] / pre_close[mask] - 1
        
        # NumPy向量化滚动偏度计算
        window = 20
        min_periods = 10
        n_dates, n_stocks = ret1.shape
        result = np.full_like(ret1, np.nan)
        
        for i in range(window - 1, n_dates):
            # 获取窗口数据
            window_data = ret1[i - window + 1:i + 1, :]
            
            # 计算有效数据数
            valid_count = np.sum(~np.isnan(window_data), axis=0)
            valid_mask = valid_count >= min_periods
            
            if not np.any(valid_mask):
                continue
            
            # 对有足够数据的股票计算偏度
            valid_data = window_data[:, valid_mask]
            
            # 计算均值和标准差（忽略NaN）
            mean = np.nanmean(valid_data, axis=0)
            std = np.nanstd(valid_data, axis=0, ddof=1)
            
            # 避免除零
            valid_std = std > 1e-10
            valid_mask[valid_mask] = valid_std
            
            if not np.any(valid_mask):
                continue
            
            # 只保留有效标准差的股票
            valid_data = valid_data[:, valid_std]
            mean = mean[valid_std]
            std = std[valid_std]
            n = np.sum(~np.isnan(valid_data), axis=0)
            
            # 计算三阶中心矩 (x - mean)^3
            centered = valid_data - mean
            centered = np.where(np.isnan(centered), 0, centered)  # NaN设为0
            m3 = np.sum(centered ** 3, axis=0) / n
            
            # 偏度 = m3 / std^3
            skew = m3 / (std ** 3)
            
            # 应用到结果
            valid_stocks = np.where(valid_mask)[0]
            result[i, valid_stocks] = skew
        
        # 极端值截断
        result = np.where((result > -5) & (result < 5), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('skew20', result, dates, stocks)
        return result
    
    def factor_kurt20(self, save=True):
        """收益率峰度因子 (20日) - NumPy向量化实现"""
        print("计算因子: kurt20 (20日收益峰度)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        
        # 计算 ret1
        ret1 = np.full_like(close, np.nan)
        mask = pre_close > 0
        ret1[mask] = close[mask] / pre_close[mask] - 1
        
        # NumPy向量化滚动峰度计算
        window = 20
        min_periods = 10
        n_dates, n_stocks = ret1.shape
        result = np.full_like(ret1, np.nan)
        
        for i in range(window - 1, n_dates):
            # 获取窗口数据
            window_data = ret1[i - window + 1:i + 1, :]
            
            # 计算有效数据数
            valid_count = np.sum(~np.isnan(window_data), axis=0)
            valid_mask = valid_count >= min_periods
            
            if not np.any(valid_mask):
                continue
            
            # 对有足够数据的股票计算峰度
            valid_data = window_data[:, valid_mask]
            
            # 计算均值和标准差（忽略NaN）
            mean = np.nanmean(valid_data, axis=0)
            std = np.nanstd(valid_data, axis=0, ddof=1)
            
            # 避免除零
            valid_std = std > 1e-10
            valid_mask[valid_mask] = valid_std
            
            if not np.any(valid_mask):
                continue
            
            # 只保留有效标准差的股票
            valid_data = valid_data[:, valid_std]
            mean = mean[valid_std]
            std = std[valid_std]
            n = np.sum(~np.isnan(valid_data), axis=0)
            
            # 计算四阶中心矩 (x - mean)^4
            centered = valid_data - mean
            centered = np.where(np.isnan(centered), 0, centered)  # NaN设为0
            m4 = np.sum(centered ** 4, axis=0) / n
            
            # 峰度 = m4 / std^4 - 3（超值峰度）
            kurt = m4 / (std ** 4) - 3
            
            # 应用到结果
            valid_stocks = np.where(valid_mask)[0]
            result[i, valid_stocks] = kurt
        
        # 极端值截断
        result = np.where((result > -10) & (result < 20), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('kurt20', result, dates, stocks)
        return result
    
    def compute_all(self, factors: Optional[List[str]] = None):
        available = {
            'close_position': self.factor_close_position,
            'intraday_return_ma5': self.factor_intraday_return_ma5,
            'intraday_return_ma20': self.factor_intraday_return_ma20,
            'close_position_ma5': self.factor_close_position_ma5,
            'close_position_ma20': self.factor_close_position_ma20,
            'skew20': self.factor_skew20,
            'kurt20': self.factor_kurt20,
        }
        
        if factors is None:
            factors = list(available.keys())
        
        output_files = []
        for name in factors:
            if name in available:
                try:
                    result = available[name](save=True)
                    if result is not None:
                        output_files.append(result)
                except Exception as e:
                    print(f"计算因子 {name} 失败: {e}")
        
        print(f"\n完成！共计算 {len(output_files)} 个因子")
        return output_files


if __name__ == "__main__":
    print("=" * 60)
    print("价格-成交量因子计算")
    print("=" * 60)
    
    try:
        pv = PriceVolumeFactors()
        pv.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
