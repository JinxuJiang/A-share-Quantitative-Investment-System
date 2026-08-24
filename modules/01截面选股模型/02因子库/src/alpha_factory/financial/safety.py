# -*- coding: utf-8 -*-
"""
安全因子家族 (Safety Factors)

包含：
1. debt_to_equity: 产权比率 = 总负债 / 归母权益
2. current_ratio: 流动比率 = 流动资产 / 流动负债
3. cash_ratio: 现金比率 = 货币资金 / 流动负债

因果逻辑：
- 安全因子反映企业的财务稳健性和偿债能力
- 低杠杆、高流动性公司抗风险能力更强

学术支撑：
- Fama-French (1992): 杠杆因子
- 财务报表分析经典指标
"""

import sys
from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

factor_lib_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(factor_lib_root))


class SafetyFactors:
    """安全因子计算类"""
    
    def __init__(self, processed_data_path: Optional[str] = None):
        if processed_data_path is None:
            self.processed_data_path = factor_lib_root / "processed_data"
        else:
            self.processed_data_path = Path(processed_data_path)
        
        self.output_path = self.processed_data_path / "factors" / "financial"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._tot_liab = None
        self._tot_shrhldr_eqy = None
        self._total_current_assets = None
        self._total_current_liability = None
        self._cash_equivalents = None
    
    def _load_tot_liab(self):
        if self._tot_liab is None:
            file_path = self.processed_data_path / "financial_data" / "tot_liab.parquet"
            table = pq.read_table(file_path)
            self._tot_liab = table.to_pandas()
            print(f"  总负债: {self._tot_liab.shape}")
        return self._tot_liab
    
    def _load_tot_shrhldr_eqy(self):
        if self._tot_shrhldr_eqy is None:
            file_path = self.processed_data_path / "financial_data" / "tot_shrhldr_eqy.parquet"
            table = pq.read_table(file_path)
            self._tot_shrhldr_eqy = table.to_pandas()
            print(f"  净资产: {self._tot_shrhldr_eqy.shape}")
        return self._tot_shrhldr_eqy
    
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
    
    def _load_cash_equivalents(self):
        if self._cash_equivalents is None:
            file_path = self.processed_data_path / "financial_data" / "cash_equivalents.parquet"
            table = pq.read_table(file_path)
            self._cash_equivalents = table.to_pandas()
            print(f"  货币资金: {self._cash_equivalents.shape}")
        return self._cash_equivalents
    
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
    
    def factor_debt_to_equity(self, save: bool = True) -> pd.DataFrame:
        """产权比率 = 总负债 / 归母权益"""
        print("\n计算因子: debt_to_equity (产权比率)")
        
        tot_liab = self._load_tot_liab()
        equity = self._load_tot_shrhldr_eqy()
        
        tot_liab, equity = self._align_dataframes(tot_liab, equity)
        
        equity_safe = equity.copy()
        equity_safe[equity_safe <= 0] = np.nan
        
        de_ratio = tot_liab / equity_safe
        de_ratio[(de_ratio > 100) | (de_ratio < 0)] = np.nan
        
        print(f"  非空值比例: {de_ratio.notna().sum().sum() / (de_ratio.shape[0] * de_ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "debt_to_equity.parquet"
            de_ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return de_ratio
    
    def factor_current_ratio(self, save: bool = True) -> pd.DataFrame:
        """流动比率 = 流动资产 / 流动负债"""
        print("\n计算因子: current_ratio (流动比率)")
        
        current_assets = self._load_total_current_assets()
        current_liab = self._load_total_current_liability()
        
        current_assets, current_liab = self._align_dataframes(current_assets, current_liab)
        
        current_liab_safe = current_liab.copy()
        current_liab_safe[current_liab_safe <= 0] = np.nan
        
        ratio = current_assets / current_liab_safe
        ratio[(ratio > 50) | (ratio < 0)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "current_ratio.parquet"
            ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def factor_cash_ratio(self, save: bool = True) -> pd.DataFrame:
        """现金比率 = 货币资金 / 流动负债"""
        print("\n计算因子: cash_ratio (现金比率)")
        
        cash = self._load_cash_equivalents()
        current_liab = self._load_total_current_liability()
        
        cash, current_liab = self._align_dataframes(cash, current_liab)
        
        current_liab_safe = current_liab.copy()
        current_liab_safe[current_liab_safe <= 0] = np.nan
        
        ratio = cash / current_liab_safe
        ratio[(ratio > 10) | (ratio < 0)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "cash_ratio.parquet"
            ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """批量计算所有因子"""
        available = {
            'debt_to_equity': self.factor_debt_to_equity,
            'current_ratio': self.factor_current_ratio,
            'cash_ratio': self.factor_cash_ratio,
        }
        
        if factors is None:
            factors = list(available.keys())
        
        print("=" * 60)
        print("安全因子计算")
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
    print("测试 SafetyFactors")
    print("=" * 60)
    
    try:
        sf = SafetyFactors()
        sf.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
