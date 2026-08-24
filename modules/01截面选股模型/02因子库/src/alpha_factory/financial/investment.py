# -*- coding: utf-8 -*-
"""
投资因子家族 (Investment Factors)

包含：
1. asset_growth: 总资产增长率 = (本期 - 上期) / 上期
2. capex_to_assets: 资本支出强度 = CAPEX_TTM / 总资产

因果逻辑：
- 过度投资的公司往往未来收益更低（投资效应）
- 资产快速扩张可能伴随并购溢价、过度扩张

学术支撑：
- Cooper, Gulen & Schill (2008): Asset Growth Anomaly
"""

import sys
from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

factor_lib_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(factor_lib_root))


class InvestmentFactors:
    """投资因子计算类"""
    
    def __init__(self, processed_data_path: Optional[str] = None):
        if processed_data_path is None:
            self.processed_data_path = factor_lib_root / "processed_data"
        else:
            self.processed_data_path = Path(processed_data_path)
        
        self.output_path = self.processed_data_path / "factors" / "financial"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._tot_assets = None
        self._capex_ttm = None
    
    def _load_tot_assets(self):
        if self._tot_assets is None:
            file_path = self.processed_data_path / "financial_data" / "tot_assets.parquet"
            table = pq.read_table(file_path)
            self._tot_assets = table.to_pandas()
            print(f"  总资产: {self._tot_assets.shape}")
        return self._tot_assets
    
    def _load_capex_ttm(self):
        if self._capex_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "capex_ttm.parquet"
            table = pq.read_table(file_path)
            self._capex_ttm = table.to_pandas()
            print(f"  CAPEX_TTM: {self._capex_ttm.shape}")
        return self._capex_ttm
    
    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'time' in df.columns:
            df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        return df
    
    def factor_asset_growth(self, save: bool = True) -> pd.DataFrame:
        """总资产增长率，使用shift(252)近似年度同比"""
        print("\n计算因子: asset_growth (总资产增长率)")
        
        tot_assets = self._load_tot_assets()
        tot_assets = self._prepare_index(tot_assets)
        
        tot_assets_lag = tot_assets.shift(252)
        tot_assets_lag_safe = tot_assets_lag.copy()
        tot_assets_lag_safe[tot_assets_lag_safe <= 0] = np.nan
        
        growth = (tot_assets - tot_assets_lag_safe) / tot_assets_lag_safe
        growth[(growth > 10) | (growth < -0.9)] = np.nan
        
        print(f"  非空值比例: {growth.notna().sum().sum() / (growth.shape[0] * growth.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "asset_growth.parquet"
            growth.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return growth
    
    def factor_capex_to_assets(self, save: bool = True) -> pd.DataFrame:
        """资本支出强度 = CAPEX_TTM / 总资产"""
        print("\n计算因子: capex_to_assets (资本支出强度)")
        
        capex = self._load_capex_ttm()
        tot_assets = self._load_tot_assets()
        
        capex, tot_assets = self._align_dataframes(capex, tot_assets)
        
        tot_assets_safe = tot_assets.copy()
        tot_assets_safe[tot_assets_safe <= 0] = np.nan
        
        ratio = capex / tot_assets_safe
        ratio[(ratio > 1) | (ratio < 0)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "capex_to_assets.parquet"
            ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def _align_dataframes(self, *dfs) -> tuple:
        dfs = [self._prepare_index(df) for df in dfs]
        common_cols = dfs[0].columns
        for df in dfs[1:]:
            common_cols = common_cols.intersection(df.columns)
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
        aligned = [df.loc[common_index, common_cols] for df in dfs]
        print(f"  对齐后: {len(common_index)} 交易日 x {len(common_cols)} 只股票")
        return tuple(aligned)
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """批量计算所有因子"""
        available = {
            'asset_growth': self.factor_asset_growth,
            'capex_to_assets': self.factor_capex_to_assets,
        }
        
        if factors is None:
            factors = list(available.keys())
        
        print("=" * 60)
        print("投资因子计算")
        print("=" * 60)
        
        results = []
        for name in factors:
            if name in available:
                try:
                    result = available[name](save=True)
                    results.append(result)
                except Exception as e:
                    print(f"计算因子 {name} 失败: {e}")
        
        print(f"\n完成！共计算 {len(results)} 个因子")
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("测试 InvestmentFactors")
    print("=" * 60)
    
    try:
        inv = InvestmentFactors()
        inv.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
