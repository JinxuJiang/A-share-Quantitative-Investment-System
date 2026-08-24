# -*- coding: utf-8 -*-
"""
估值因子家族 (Valuation Factors)

估值因子反映股票的市场价格与基本面价值的比率，是价值投资的核心指标。

因果逻辑：
- 价值效应：低估值股票长期跑赢高估值股票（价值溢价）
- 均值回归：价格围绕价值波动，极端估值会回归
- 安全边际：低估值提供下行保护

参考：
- Fama-French (1992): The Cross-Section of Expected Stock Returns
- Lakonishok et al. (1994): Contrarian Investment, Extrapolation, and Risk
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


class ValuationFactors:
    """
    估值因子计算类
    
    包含：PE、PB、PS 等经典估值指标
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
        self._close = None
        self._cap_stk = None
        self._net_profit_ttm = None
        self._net_assets = None
        self._revenue_ttm = None
    
    def _load_market_data(self):
        """加载行情数据（收盘价）"""
        if self._close is None:
            close_file = self.processed_data_path / "market_data" / "close.parquet"
            if not close_file.exists():
                raise FileNotFoundError(f"收盘价数据不存在: {close_file}")
            
            table = pq.read_table(close_file)
            self._close = table.to_pandas()
            print(f"  收盘价: {self._close.shape}")
        return self._close
    
    def _load_cap_stk(self):
        """加载总股本"""
        if self._cap_stk is None:
            cap_stk_file = self.processed_data_path / "financial_data" / "cap_stk.parquet"
            if not cap_stk_file.exists():
                raise FileNotFoundError(f"总股本数据不存在: {cap_stk_file}")
            
            table = pq.read_table(cap_stk_file)
            self._cap_stk = table.to_pandas()
            print(f"  总股本: {self._cap_stk.shape}")
        return self._cap_stk
    
    def _load_net_profit_ttm(self):
        """加载净利润_TTM"""
        if self._net_profit_ttm is None:
            net_profit_file = self.processed_data_path / "financial_data" / "net_profit_ttm.parquet"
            if not net_profit_file.exists():
                raise FileNotFoundError(f"净利润TTM数据不存在: {net_profit_file}")
            
            table = pq.read_table(net_profit_file)
            self._net_profit_ttm = table.to_pandas()
            print(f"  净利润TTM: {self._net_profit_ttm.shape}")
        return self._net_profit_ttm
    
    def _load_net_assets(self):
        """加载净资产（归母股东权益）"""
        if self._net_assets is None:
            net_assets_file = self.processed_data_path / "financial_data" / "tot_shrhldr_eqy.parquet"
            if not net_assets_file.exists():
                raise FileNotFoundError(f"净资产数据不存在: {net_assets_file}")
            
            table = pq.read_table(net_assets_file)
            self._net_assets = table.to_pandas()
            print(f"  净资产: {self._net_assets.shape}")
        return self._net_assets
    
    def _load_revenue_ttm(self):
        """加载营业收入_TTM"""
        if self._revenue_ttm is None:
            revenue_file = self.processed_data_path / "financial_data" / "revenue_ttm.parquet"
            if not revenue_file.exists():
                raise FileNotFoundError(f"营业收入TTM数据不存在: {revenue_file}")
            
            table = pq.read_table(revenue_file)
            self._revenue_ttm = table.to_pandas()
            print(f"  营业收入TTM: {self._revenue_ttm.shape}")
        return self._revenue_ttm
    
    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """将time列设为索引"""
        if 'time' in df.columns:
            df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        return df
    
    def _align_dataframes(self, *dfs) -> tuple:
        """对齐多个DataFrame的行列"""
        # 先统一处理索引
        dfs = [self._prepare_index(df) for df in dfs]
        
        # 对齐列（股票代码）
        common_cols = dfs[0].columns
        for df in dfs[1:]:
            common_cols = common_cols.intersection(df.columns)
        
        # 对齐行（日期）
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
        
        # 截取共同部分
        aligned = [df.loc[common_index, common_cols] for df in dfs]
        
        print(f"  对齐后: {len(common_index)} 交易日 × {len(common_cols)} 只股票")
        return tuple(aligned)
    
    def factor_pe(self, save: bool = True) -> pd.DataFrame:
        """
        PE (市盈率, Price-to-Earnings Ratio)
        
        因果逻辑：
        ---------
        股价与每股收益的比率，反映市场愿意为每单位盈利支付的价格。
        
        低PE策略（价值投资）：
        - 市场可能对某些股票过度悲观，导致价格低于内在价值
        - 低PE股票往往有更高的安全边际和分红率
        - 长期看，低PE股票组合能跑赢市场（价值溢价）
        
        注意：PE为负（亏损股）或极高（泡沫股）应剔除
        
        公式：
        ------
        PE = 市值 / 净利润_TTM = close × cap_stk / net_profit_ttm
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : PE因子宽表
        """
        print("\n计算因子: PE (市盈率)")
        
        # 加载数据
        close = self._load_market_data()
        cap_stk = self._load_cap_stk()
        net_profit = self._load_net_profit_ttm()
        
        # 对齐
        close, cap_stk, net_profit = self._align_dataframes(close, cap_stk, net_profit)
        
        # 计算市值
        market_cap = close * cap_stk
        
        # 计算PE（净利润<=0设为NaN）
        net_profit_safe = net_profit.copy()
        net_profit_safe[net_profit_safe <= 0] = np.nan
        
        pe = market_cap / net_profit_safe
        
        # 极端值处理
        pe[(pe > 1000) | (pe < 0)] = np.nan
        
        print(f"  非空值比例: {pe.notna().sum().sum() / (pe.shape[0] * pe.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "pe_ratio.parquet"
            pe_reset = pe.reset_index()
            pe_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return pe
    
    def factor_pb(self, save: bool = True) -> pd.DataFrame:
        """
        PB (市净率, Price-to-Book Ratio)
        
        因果逻辑：
        ---------
        股价与每股净资产的比率，反映市场愿意为每单位账面价值支付的价格。
        
        PB因子的优势：
        - 比PE更稳定（净资产波动小于净利润）
        - 适用于周期性行业和亏损企业
        - Fama-French三因子中的HML（高账面市值比）核心指标
        
        低PB策略：
        - 清算价值视角：股价接近或低于净资产，下行空间有限
        - 反转效应：低PB股票往往是暂时失宠的价值股
        - 长期看，低PB组合有显著超额收益
        
        公式：
        ------
        PB = 市值 / 净资产 = close × cap_stk / tot_shrhldr_eqy
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : PB因子宽表
        """
        print("\n计算因子: PB (市净率)")
        
        # 加载数据
        close = self._load_market_data()
        cap_stk = self._load_cap_stk()
        net_assets = self._load_net_assets()
        
        # 对齐
        close, cap_stk, net_assets = self._align_dataframes(close, cap_stk, net_assets)
        
        # 计算市值
        market_cap = close * cap_stk
        
        # 计算PB（净资产<=0设为NaN）
        net_assets_safe = net_assets.copy()
        net_assets_safe[net_assets_safe <= 0] = np.nan
        
        pb = market_cap / net_assets_safe
        
        # 极端值处理（PB通常不会超过50）
        pb[(pb > 50) | (pb < 0)] = np.nan
        
        print(f"  非空值比例: {pb.notna().sum().sum() / (pb.shape[0] * pb.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "pb_ratio.parquet"
            pb_reset = pb.reset_index()
            pb_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return pb
    
    def factor_ps(self, save: bool = True) -> pd.DataFrame:
        """
        PS (市销率, Price-to-Sales Ratio)
        
        因果逻辑：
        ---------
        股价与每股销售额的比率，反映市场愿意为每单位营收支付的价格。
        
        PS因子的适用场景：
        - 亏损企业：PE为负无法使用，PS可替代
        - 成长初期：营收增长但尚未盈利（如科技股、成长股）
        - 收入质量：营收比利润更难操纵，PS更稳健
        
        低PS策略：
        - 价值投资补充：对于高成长但暂未盈利的公司的估值
        - 结合营收增长率使用效果更佳（PS低+增长高 = 被低估的成长股）
        
        注意：不同行业PS差异大（零售低、软件高），需行业中性化
        
        公式：
        ------
        PS = 市值 / 营收_TTM = close × cap_stk / revenue_ttm
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : PS因子宽表
        """
        print("\n计算因子: PS (市销率)")
        
        # 加载数据
        close = self._load_market_data()
        cap_stk = self._load_cap_stk()
        revenue = self._load_revenue_ttm()
        
        # 对齐
        close, cap_stk, revenue = self._align_dataframes(close, cap_stk, revenue)
        
        # 计算市值
        market_cap = close * cap_stk
        
        # 计算PS（营收<=0设为NaN）
        revenue_safe = revenue.copy()
        revenue_safe[revenue_safe <= 0] = np.nan
        
        ps = market_cap / revenue_safe
        
        # 极端值处理
        ps[(ps > 100) | (ps < 0)] = np.nan
        
        print(f"  非空值比例: {ps.notna().sum().sum() / (ps.shape[0] * ps.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "ps_ratio.parquet"
            ps_reset = ps.reset_index()
            ps_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ps
    
    def factor_ey(self, save: bool = True) -> pd.DataFrame:
        """
        EY (盈利收益率, Earnings Yield)
        
        因果逻辑：
        ---------
        PE的倒数，即净利润与市值的比率。
        
        EY = 1/PE，逻辑与PE相反：
        - EY越高（PE越低），股票越"便宜"
        - EY为正比PE为负更易处理（盈利股票EY>0，亏损EY<0）
        
        优势：
        - 与收益率同向：高EY ≈ 高收益预期
        - 可加性：可以组合不同公司的EY计算组合收益率
        - 格林布拉特神奇公式的核心指标之一
        
        公式：
        ------
        EY = 净利润_TTM / 市值 = net_profit_ttm / (close × cap_stk)
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : EY因子宽表
        """
        print("\n计算因子: EY (盈利收益率)")
        
        # 加载数据
        close = self._load_market_data()
        cap_stk = self._load_cap_stk()
        net_profit = self._load_net_profit_ttm()
        
        # 对齐
        close, cap_stk, net_profit = self._align_dataframes(close, cap_stk, net_profit)
        
        # 计算市值
        market_cap = close * cap_stk
        market_cap_safe = market_cap.copy()
        market_cap_safe[market_cap_safe <= 0] = np.nan
        
        # 计算EY
        ey = net_profit / market_cap_safe
        
        # 极端值处理（EY范围通常在-1到1之间）
        ey[(ey > 2) | (ey < -1)] = np.nan
        
        print(f"  非空值比例: {ey.notna().sum().sum() / (ey.shape[0] * ey.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "earnings_yield.parquet"
            ey_reset = ey.reset_index()
            ey_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ey
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """
        批量计算所有估值因子
        
        参数:
        -----
        factors : List[str], optional
            要计算的因子列表，默认计算所有
            
        返回:
        ------
        List[pd.DataFrame] : 因子宽表列表
        """
        available_factors = {
            'pe': self.factor_pe,
            'pb': self.factor_pb,
            'ps': self.factor_ps,
            'ey': self.factor_ey,
        }
        
        if factors is None:
            factors = list(available_factors.keys())
        
        print("=" * 60)
        print("估值因子计算")
        print("=" * 60)
        
        results = []
        
        for factor_name in factors:
            if factor_name in available_factors:
                try:
                    result = available_factors[factor_name](save=True)
                    results.append(result)
                except Exception as e:
                    print(f"计算因子 {factor_name} 失败: {e}")
            else:
                print(f"未知因子: {factor_name}，可用因子: {list(available_factors.keys())}")
        
        print(f"\n全部完成！共计算 {len(results)} 个因子")
        return results


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试 ValuationFactors")
    print("=" * 60)
    
    try:
        vf = ValuationFactors()
        vf.compute_all()
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 data_engine 准备数据")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
