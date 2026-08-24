# -*- coding: utf-8 -*-
"""
流动性因子家族 (Liquidity Factors)

包含：amihud, pv_corr20, vol_trend, amount_ratio
"""

from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


class LiquidityFactors:
    """流动性因子计算类"""
    
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
    
    def factor_amihud(self, save=True):
        """
        Amihud非流动性因子
        
        公式: |ret1| / amount
        逻辑: 单位成交额带来的价格冲击，值越大流动性越差
        无未来函数: 使用当日收益率和当日成交额
        """
        print("计算因子: amihud (Amihud非流动性)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        _, _, amount = self._to_numpy('amount')
        
        # 计算 ret1
        ret1 = np.full_like(close, np.nan)
        mask = pre_close > 0
        ret1[mask] = close[mask] / pre_close[mask] - 1
        
        # Amihud = |ret| / amount
        result = np.full_like(close, np.nan)
        amount_safe = np.where(amount > 0, amount, np.nan)
        result = np.abs(ret1) / amount_safe
        
        # 极端值处理
        result = np.where(result > 1, np.nan, result)  # Amihud通常很小
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('amihud', result, dates, stocks)
        return result
    
    def factor_pv_corr20(self, save=True):
        """
        量价相关性因子 (20日) - NumPy向量化实现
        
        公式: corr(ret1, volume, 20)
        逻辑: 过去20日收益率与成交量的相关系数
        """
        print("计算因子: pv_corr20 (20日量价相关性)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        _, _, volume = self._to_numpy('volume')
        
        # 计算 ret1
        ret1 = np.full_like(close, np.nan)
        mask = pre_close > 0
        ret1[mask] = close[mask] / pre_close[mask] - 1
        
        # NumPy向量化滚动相关系数计算
        window = 20
        min_periods = 10
        n_dates, n_stocks = ret1.shape
        result = np.full_like(ret1, np.nan)
        
        # 预计算滚动窗口的统计量（向量化）
        for i in range(window - 1, n_dates):
            # 获取窗口数据 (window, n_stocks)
            ret_window = ret1[i - window + 1:i + 1, :]
            vol_window = volume[i - window + 1:i + 1, :]
            
            # 计算每只股票的有效数据数（非NaN）
            ret_valid = np.sum(~np.isnan(ret_window), axis=0)
            vol_valid = np.sum(~np.isnan(vol_window), axis=0)
            valid_mask = (ret_valid >= min_periods) & (vol_valid >= min_periods)
            
            if not np.any(valid_mask):
                continue
            
            # 只对有足够数据的股票计算（向量化）
            ret_valid_window = ret_window[:, valid_mask]
            vol_valid_window = vol_window[:, valid_mask]
            
            # 计算均值（忽略NaN）
            ret_mean = np.nanmean(ret_valid_window, axis=0)
            vol_mean = np.nanmean(vol_valid_window, axis=0)
            
            # 中心化
            ret_centered = ret_valid_window - ret_mean
            vol_centered = vol_valid_window - vol_mean
            
            # 将NaN设为0以便计算（不影响结果因为对应位置会被mask）
            ret_centered = np.where(np.isnan(ret_centered), 0, ret_centered)
            vol_centered = np.where(np.isnan(vol_centered), 0, vol_centered)
            
            # 计算协方差和方差
            cov = np.sum(ret_centered * vol_centered, axis=0)
            ret_var = np.sum(ret_centered ** 2, axis=0)
            vol_var = np.sum(vol_centered ** 2, axis=0)
            
            # 相关系数
            denom = np.sqrt(ret_var * vol_var)
            nonzero_mask = denom > 1e-10
            
            valid_stocks = np.where(valid_mask)[0]
            valid_nonzero = valid_stocks[nonzero_mask]
            result[i, valid_nonzero] = cov[nonzero_mask] / denom[nonzero_mask]
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('pv_corr20', result, dates, stocks)
        return result
    
    def factor_vol_trend(self, save=True):
        """
        成交量趋势因子
        
        公式: mean(volume, 5) / mean(volume, 20)
        逻辑: 短期量能相对长期量能的趋势，>1表示放量
        无未来函数: 只用历史成交量
        """
        print("计算因子: vol_trend (成交量趋势)")
        dates, stocks, volume = self._to_numpy('volume')
        
        n_dates, n_stocks = volume.shape
        result = np.full_like(volume, np.nan, dtype=np.float64)
        
        short_period = 5
        long_period = 20
        
        for i in range(long_period - 1, n_dates):
            short_vol = volume[i - short_period + 1:i + 1, :]
            long_vol = volume[i - long_period + 1:i + 1, :]
            
            short_mean = np.nanmean(short_vol, axis=0)
            long_mean = np.nanmean(long_vol, axis=0)
            
            # 处理全NaN情况（避免警告）
            short_mean = np.where(np.all(np.isnan(short_vol), axis=0), np.nan, short_mean)
            long_mean = np.where(np.all(np.isnan(long_vol), axis=0), np.nan, long_mean)
            
            valid = long_mean > 0
            result[i, valid] = short_mean[valid] / long_mean[valid]
        
        # 极端值处理
        result = np.where((result > 10) | (result < 0.1), np.nan, result)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('vol_trend', result, dates, stocks)
        return result
    
    def factor_amount_ratio(self, save=True):
        """
        成交额比率因子
        
        公式: amount / mean(amount, 20)
        逻辑: 当日成交额相对20日均值的比率，放量/缩量指标
        无未来函数: 只用历史成交额
        """
        print("计算因子: amount_ratio (成交额比率)")
        dates, stocks, amount = self._to_numpy('amount')
        
        n_dates, n_stocks = amount.shape
        result = np.full_like(amount, np.nan, dtype=np.float64)
        
        period = 20
        
        for i in range(period - 1, n_dates):
            window = amount[i - period + 1:i + 1, :]
            mean_val = np.nanmean(window, axis=0)
            
            # 处理全NaN情况
            mean_val = np.where(np.all(np.isnan(window), axis=0), np.nan, mean_val)
            
            valid = mean_val > 0
            result[i, valid] = amount[i, valid] / mean_val[valid]
        
        # 极端值处理
        result = np.where(result > 20, np.nan, result)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('amount_ratio', result, dates, stocks)
        return result
    
    def compute_all(self, factors: Optional[List[str]] = None):
        available = {
            'amihud': self.factor_amihud,
            'pv_corr20': self.factor_pv_corr20,
            'vol_trend': self.factor_vol_trend,
            'amount_ratio': self.factor_amount_ratio,
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
    print("流动性因子计算")
    print("=" * 60)
    
    try:
        lf = LiquidityFactors()
        lf.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
