# -*- coding: utf-8 -*-
"""
动量因子家族 (Momentum Factors)

包含：ret1, ret5, ret20, ret60, ret120, ret20_60
"""

from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


class MomentumFactors:
    """动量因子计算类"""
    
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
    
    def factor_ret1(self, save=True):
        """1日收益率 - 短期反转因子"""
        print("计算因子: ret1 (1日收益率)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        
        result = np.full_like(close, np.nan)
        mask = pre_close > 0
        result[mask] = close[mask] / pre_close[mask] - 1
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('ret1', result, dates, stocks)
        return result
    
    def factor_ret5(self, save=True):
        """5日收益率 - 周度反转因子"""
        print("计算因子: ret5 (5日收益率)")
        dates, stocks, close = self._to_numpy('close')
        
        period = 5
        result = np.full_like(close, np.nan)
        if len(close) > period:
            result[period:] = close[period:] / close[:-period] - 1
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('ret5', result, dates, stocks)
        return result
    
    def factor_ret20(self, save=True):
        """20日收益率 - 短期动量"""
        print("计算因子: ret20 (20日收益率)")
        dates, stocks, close = self._to_numpy('close')
        
        period = 20
        result = np.full_like(close, np.nan)
        if len(close) > period:
            valid = close[:-period] > 0
            result[period:] = np.where(valid, close[period:] / close[:-period] - 1, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('ret20', result, dates, stocks)
        return result
    
    def factor_ret60(self, save=True):
        """60日收益率 - 中期动量"""
        print("计算因子: ret60 (60日收益率)")
        dates, stocks, close = self._to_numpy('close')
        
        period = 60
        result = np.full_like(close, np.nan)
        if len(close) > period:
            valid = close[:-period] > 0
            result[period:] = np.where(valid, close[period:] / close[:-period] - 1, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('ret60', result, dates, stocks)
        return result
    
    def factor_ret120(self, save=True):
        """120日收益率 - 长期动量"""
        print("计算因子: ret120 (120日收益率)")
        dates, stocks, close = self._to_numpy('close')
        
        period = 120
        result = np.full_like(close, np.nan)
        if len(close) > period:
            valid = close[:-period] > 0
            result[period:] = np.where(valid, close[period:] / close[:-period] - 1, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('ret120', result, dates, stocks)
        return result
    
    def factor_ret20_60(self, save=True):
        """动量差 - 动量加速度"""
        print("计算因子: ret20_60 (动量差)")
        dates, stocks, close = self._to_numpy('close')
        
        period_short = 20
        period_long = 60
        result = np.full_like(close, np.nan)
        
        if len(close) > period_long:
            ret20 = np.where(close[:-period_short] > 0, 
                           close[period_short:] / close[:-period_short] - 1, np.nan)
            ret60 = np.where(close[:-period_long] > 0,
                           close[period_long:] / close[:-period_long] - 1, np.nan)
            result[period_long:] = ret20[period_long - period_short:] - ret60
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('ret20_60', result, dates, stocks)
        return result
    
    def compute_all(self, factors: Optional[List[str]] = None):
        available = {
            'ret1': self.factor_ret1,
            'ret5': self.factor_ret5,
            'ret20': self.factor_ret20,
            'ret60': self.factor_ret60,
            'ret120': self.factor_ret120,
            'ret20_60': self.factor_ret20_60,
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
    print("动量因子计算")
    print("=" * 60)
    
    try:
        mf = MomentumFactors()
        mf.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
