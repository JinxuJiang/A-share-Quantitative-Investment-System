# -*- coding: utf-8 -*-
"""
成长因子家族 (Growth Factors)

成长因子反映企业盈利能力和收入规模的增长速度，是成长股投资的核心。

因果逻辑：
- GARP策略：合理价格成长（Growth at Reasonable Price）
- 动量效应：成长加速往往伴随估值提升（戴维斯双击）
- 预期差：实际成长超预期时股价上涨

注意：
- 成长因子与价值因子负相关（成长股往往估值高）
- 高成长不可持续，需结合质量因子筛选

参考：
- Lakonishok et al. (1994): Contrarian Investment, Extrapolation, and Risk
- Fama-French (2015): A Five-Factor Asset Pricing Model (CMA投资因子)
"""

import sys
from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# 添加项目路径
factor_lib_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(factor_lib_root))


class GrowthFactors:
    """
    成长因子计算类
    
    包含：利润增长、营收增长等成长性指标
    """
    
    def __init__(self, processed_data_path: Optional[str] = None):
        """
        初始化
        
        参数:
        -----
        processed_data_path : str, optional
            processed_data 路径，默认使用 因子库/processed_data/
        """
        if processed_data_path is None:
            self.processed_data_path = factor_lib_root / "processed_data"
        else:
            self.processed_data_path = Path(processed_data_path)
        
        self.output_path = self.processed_data_path / "factors" / "financial"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self._net_profit_ttm = None
        self._revenue_ttm = None
        self._oper_profit_ttm = None
    
    def _load_net_profit_ttm(self):
        """加载净利润_TTM"""
        if self._net_profit_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "net_profit_ttm.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"净利润TTM数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._net_profit_ttm = table.to_pandas()
            print(f"  净利润TTM: {self._net_profit_ttm.shape}")
        return self._net_profit_ttm
    
    def _load_revenue_ttm(self):
        """加载营业收入_TTM"""
        if self._revenue_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "revenue_ttm.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"营收TTM数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._revenue_ttm = table.to_pandas()
            print(f"  营收TTM: {self._revenue_ttm.shape}")
        return self._revenue_ttm
    
    def _load_oper_profit_ttm(self):
        """加载营业利润_TTM"""
        if self._oper_profit_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "oper_profit_ttm.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"营业利润TTM数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._oper_profit_ttm = table.to_pandas()
            print(f"  营业利润TTM: {self._oper_profit_ttm.shape}")
        return self._oper_profit_ttm
    
    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """将time列设为索引"""
        if 'time' in df.columns:
            df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        return df
    
    def _calculate_yoy_growth(self, df: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
        """
        计算同比增长率
        
        参数:
        -----
        df : pd.DataFrame
            财务数据宽表
        periods : int
            同比周期，默认252个交易日（约1年）
            
        返回:
        ------
        pd.DataFrame : 同比增长率
        """
        df = self._prepare_index(df)
        
        # 计算同比变化
        df_lag = df.shift(periods)
        
        # 增长率 = (当期 - 上期) / |上期|
        # 上期<=0时设为NaN（无法计算有效增长率）
        df_lag_safe = df_lag.copy()
        df_lag_safe[df_lag_safe <= 0] = np.nan
        
        growth = (df - df_lag_safe) / df_lag_safe.abs()
        
        return growth
    
    def factor_profit_growth(self, save: bool = True) -> pd.DataFrame:
        """
        净利润增长率 (Profit Growth, YoY)
        
        因果逻辑：
        ---------
        净利润的同比增速，反映企业盈利能力的增长趋势。
        
        GARP策略核心：
        - 寻找盈利增长稳定但估值合理的股票
        - 避免过高估值的成长股（估值陷阱）
        - 避免零增长的价值股（价值陷阱）
        
        注意：
        - 净利润波动大，单季度异常值多，建议用TTM平滑
        - 高成长难以持续，需结合ROE看质量
        
        公式：
        ------
        Profit_Growth(t) = (净利润_TTM(t) - 净利润_TTM(t-252)) / |净利润_TTM(t-252)|
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 净利润增长率宽表
        """
        print("\n计算因子: profit_growth (净利润增长率 YoY)")
        
        net_profit = self._load_net_profit_ttm()
        
        # 计算同比增长
        growth = self._calculate_yoy_growth(net_profit, periods=252)
        
        # 极端值处理（增长率通常在-100%到+200%之间）
        growth[(growth > 5) | (growth < -1)] = np.nan
        
        print(f"  非空值比例: {growth.notna().sum().sum() / (growth.shape[0] * growth.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "profit_growth.parquet"
            growth_reset = growth.reset_index()
            growth_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return growth
    
    def factor_revenue_growth(self, save: bool = True) -> pd.DataFrame:
        """
        营收增长率 (Revenue Growth, YoY)
        
        因果逻辑：
        ---------
        营业收入的同比增速，反映企业市场扩张能力。
        
        营收增长 vs 利润增长：
        - 营收增长更稳定，不易被财务手段操纵
        - 高营收增长+低利润增长 = 可能是价格战或扩张期
        - 营收下滑+利润增长 = 可能是裁员或收缩，不可持续
        
        适用场景：
        - 成长股筛选：营收高增长是成长股的必要条件
        - 周期判断：营收增速拐点往往领先利润拐点
        
        公式：
        ------
        Revenue_Growth(t) = (营收_TTM(t) - 营收_TTM(t-252)) / |营收_TTM(t-252)|
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 营收增长率宽表
        """
        print("\n计算因子: revenue_growth (营收增长率 YoY)")
        
        revenue = self._load_revenue_ttm()
        
        # 计算同比增长
        growth = self._calculate_yoy_growth(revenue, periods=252)
        
        # 极端值处理
        growth[(growth > 5) | (growth < -1)] = np.nan
        
        print(f"  非空值比例: {growth.notna().sum().sum() / (growth.shape[0] * growth.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "revenue_growth.parquet"
            growth_reset = growth.reset_index()
            growth_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return growth
    
    def factor_oper_profit_growth(self, save: bool = True) -> pd.DataFrame:
        """
        营业利润增长率 (Operating Profit Growth, YoY)
        
        因果逻辑：
        ---------
        营业利润的同比增速，反映核心业务的内生增长。
        
        相比净利润增长的优势：
        - 排除非经常性损益（投资收益、政府补贴等）
        - 更纯粹反映主营业务的改善
        - 不易被一次性因素扭曲
        
        应用场景：
        - 识别盈利质量改善：营业利润增长 > 净利润增长（核心能力增强）
        - 剔除利润操纵：只看主营，排除变卖资产等一次性收益
        
        公式：
        ------
        Oper_Profit_Growth(t) = (营业利润_TTM(t) - 营业利润_TTM(t-252)) / |营业利润_TTM(t-252)|
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 营业利润增长率宽表
        """
        print("\n计算因子: oper_profit_growth (营业利润增长率 YoY)")
        
        oper_profit = self._load_oper_profit_ttm()
        
        # 计算同比增长
        growth = self._calculate_yoy_growth(oper_profit, periods=252)
        
        # 极端值处理
        growth[(growth > 5) | (growth < -1)] = np.nan
        
        print(f"  非空值比例: {growth.notna().sum().sum() / (growth.shape[0] * growth.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "oper_profit_growth.parquet"
            growth_reset = growth.reset_index()
            growth_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return growth
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """
        批量计算所有成长因子
        
        参数:
        -----
        factors : List[str], optional
            要计算的因子列表，默认计算所有
            
        返回:
        ------
        List[pd.DataFrame] : 因子宽表列表
        """
        available_factors = {
            'profit_growth': self.factor_profit_growth,
            'revenue_growth': self.factor_revenue_growth,
            'oper_profit_growth': self.factor_oper_profit_growth,
        }
        
        if factors is None:
            factors = list(available_factors.keys())
        
        print("=" * 60)
        print("成长因子计算")
        print("=" * 60)
        
        results = []
        
        for factor_name in factors:
            if factor_name in available_factors:
                try:
                    result = available_factors[factor_name](save=True)
                    results.append(result)
                except Exception as e:
                    print(f"计算因子 {factor_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"未知因子: {factor_name}，可用因子: {list(available_factors.keys())}")
        
        print(f"\n全部完成！共计算 {len(results)} 个因子")
        return results


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试 GrowthFactors")
    print("=" * 60)
    
    try:
        gf = GrowthFactors()
        gf.compute_all()
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 data_engine 准备数据")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
