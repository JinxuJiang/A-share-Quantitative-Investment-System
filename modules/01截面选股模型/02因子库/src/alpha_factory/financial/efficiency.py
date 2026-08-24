# -*- coding: utf-8 -*-
"""
效率因子家族 (Efficiency Factors)

包含：
1. asset_turnover: 资产周转率 = 营收_TTM / 总资产
2. working_capital_ratio: 营运资本占比 = (流动资产 - 流动负债) / 总资产

因果逻辑：
- 效率因子反映企业资产使用和运营效率
- 资产周转率高意味着轻资产模式或高运营效率

学术支撑：
- 杜邦分析：ROE = 净利润率 x 资产周转率 x 权益乘数
"""

import sys
from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

factor_lib_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(factor_lib_root))


class EfficiencyFactors:
    """效率因子计算类"""
    
    def __init__(self, processed_data_path: Optional[str] = None):
        if processed_data_path is None:
            self.processed_data_path = factor_lib_root / "processed_data"
        else:
            self.processed_data_path = Path(processed_data_path)
        
        self.output_path = self.processed_data_path / "factors" / "financial"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._revenue_ttm = None
        self._tot_assets = None
        self._total_current_assets = None
        self._total_current_liability = None
    
    def _load_revenue_ttm(self):
        if self._revenue_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "revenue_ttm.parquet"
            table = pq.read_table(file_path)
            self._revenue_ttm = table.to_pandas()
            print(f"  营收TTM: {self._revenue_ttm.shape}")
        return self._revenue_ttm
    
    def _load_tot_assets(self):
        if self._tot_assets is None:
            file_path = self.processed_data_path / "financial_data" / "tot_assets.parquet"
            table = pq.read_table(file_path)
            self._tot_assets = table.to_pandas()
            print(f"  总资产: {self._tot_assets.shape}")
        return self._tot_assets
    
    def _load_total_current_assets(self):
        if self._total_current_assets is None:
            file_path = self.processed_data_path / "financial_data" / "total_current_assets.parquet"
            table = pq.read_table(file_path)
            self._total_current_assets = table.to_pandas()
            print(f"  流动资产: {self._total_current_assets.shape}")
        return self._total_current_assets
    
    def _load_total_current_liability(self):
        if self._total_current_liability is None:
            file_path = self.processed_data_path / "financial_data" / "total_current_liability.parquet"
            table = pq.read_table(file_path)
            self._total_current_liability = table.to_pandas()
            print(f"  流动负债: {self._total_current_liability.shape}")
        return self._total_current_liability
    
    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'time' in df.columns:
            df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        return df
    
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
    
    def factor_asset_turnover(self, save: bool = True) -> pd.DataFrame:
        """资产周转率 = 营收_TTM / 总资产"""
        print("\n计算因子: asset_turnover (资产周转率)")
        
        revenue = self._load_revenue_ttm()
        tot_assets = self._load_tot_assets()
        
        revenue, tot_assets = self._align_dataframes(revenue, tot_assets)
        
        tot_assets_safe = tot_assets.copy()
        tot_assets_safe[tot_assets_safe <= 0] = np.nan
        
        turnover = revenue / tot_assets_safe
        turnover[(turnover > 5) | (turnover < 0)] = np.nan
        
        print(f"  非空值比例: {turnover.notna().sum().sum() / (turnover.shape[0] * turnover.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "asset_turnover.parquet"
            turnover.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return turnover
    
    def factor_working_capital_ratio(self, save: bool = True) -> pd.DataFrame:
        """营运资本占比 = (流动资产 - 流动负债) / 总资产"""
        print("\n计算因子: working_capital_ratio (营运资本占比)")
        
        current_assets = self._load_total_current_assets()
        current_liab = self._load_total_current_liability()
        tot_assets = self._load_tot_assets()
        
        current_assets, current_liab, tot_assets = self._align_dataframes(
            current_assets, current_liab, tot_assets
        )
        
        working_capital = current_assets - current_liab
        tot_assets_safe = tot_assets.copy()
        tot_assets_safe[tot_assets_safe <= 0] = np.nan
        
        ratio = working_capital / tot_assets_safe
        ratio[(ratio > 1) | (ratio < -0.5)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "working_capital_ratio.parquet"
            ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """批量计算所有因子"""
        available = {
            'asset_turnover': self.factor_asset_turnover,
            'working_capital_ratio': self.factor_working_capital_ratio,
        }
        
        if factors is None:
            factors = list(available.keys())
        
        print("=" * 60)
        print("效率因子计算")
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
    print("测试 EfficiencyFactors")
    print("=" * 60)
    
    try:
        ef = EfficiencyFactors()
        ef.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
