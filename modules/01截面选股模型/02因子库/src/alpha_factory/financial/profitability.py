# -*- coding: utf-8 -*-
"""
盈利因子家族 (Profitability Factors)

盈利因子反映企业的盈利能力和资本使用效率，是质量投资的核心。

因果逻辑：
- 盈利溢价：高盈利公司长期跑赢低盈利公司（Novy-Marx, 2013）
- 竞争优势：持续高盈利往往源于护城河（品牌、规模、网络效应）
- 复利效应：高ROE公司的利润再投资产生复利增长

参考：
- Fama-French (2015): A Five-Factor Asset Pricing Model (RMW盈利因子)
- Novy-Marx (2013): The Other Side of Value: The Gross Profitability Premium
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


class ProfitabilityFactors:
    """
    盈利因子计算类
    
    包含：ROE、ROA、ROE_Growth 等盈利指标
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
        self._net_assets = None
        self._tot_assets = None
        self._oper_profit_ttm = None
                    
    def _load_net_profit_ttm(self):
        """加载净利润_TTM（归母净利润）"""
        if self._net_profit_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "net_profit_ttm.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"净利润TTM数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._net_profit_ttm = table.to_pandas()
            print(f"  净利润TTM: {self._net_profit_ttm.shape}")
        return self._net_profit_ttm
    
    def _load_net_assets(self):
        """加载净资产（归母股东权益）"""
        if self._net_assets is None:
            file_path = self.processed_data_path / "financial_data" / "tot_shrhldr_eqy.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"净资产数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._net_assets = table.to_pandas()
            print(f"  净资产: {self._net_assets.shape}")
        return self._net_assets
    
    def _load_tot_assets(self):
        """加载总资产"""
        if self._tot_assets is None:
            file_path = self.processed_data_path / "financial_data" / "tot_assets.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"总资产数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._tot_assets = table.to_pandas()
            print(f"  总资产: {self._tot_assets.shape}")
        return self._tot_assets
    
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
    
    def _align_dataframes(self, *dfs) -> tuple:
        """对齐多个DataFrame的行列"""
        dfs = [self._prepare_index(df) for df in dfs]
        
        common_cols = dfs[0].columns
        for df in dfs[1:]:
            common_cols = common_cols.intersection(df.columns)
        
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
        
        aligned = [df.loc[common_index, common_cols] for df in dfs]
        print(f"  对齐后: {len(common_index)} 交易日 × {len(common_cols)} 只股票")
        return tuple(aligned)
    
    def factor_roe(self, save: bool = True) -> pd.DataFrame:
        """
        ROE (净资产收益率, Return on Equity)
        
        因果逻辑：
        ---------
        净利润与净资产的比率，反映股东投入资本的盈利能力。
        
        巴菲特最看重的指标：
        - 高ROE意味着公司能用少量资本创造大量利润
        - 持续高ROE（>15%）往往是竞争优势的体现
        - 复利效应：高ROE公司的留存收益再投资，推动长期增长
        
        杜邦分析拆解：
        ROE = 净利润率 × 资产周转率 × 权益乘数
            = (净利润/营收) × (营收/总资产) × (总资产/净资产)
        
        高ROE来源：
        - 高利润率（品牌溢价，如茅台）
        - 高周转率（薄利多销，如零售）
        - 高杠杆（金融属性，如银行）
        
        公式：
        ------
        ROE = 净利润_TTM / 净资产
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : ROE因子宽表
        """
        print("\n计算因子: ROE (净资产收益率)")
        
        net_profit = self._load_net_profit_ttm()
        net_assets = self._load_net_assets()
        
        net_profit, net_assets = self._align_dataframes(net_profit, net_assets)
        
        # 净资产<=0设为NaN
        net_assets_safe = net_assets.copy()
        net_assets_safe[net_assets_safe <= 0] = np.nan
        
        roe = net_profit / net_assets_safe
        
        # 极端值处理（ROE通常在-100%到100%之间）
        roe[(roe > 2) | (roe < -1)] = np.nan
        
        print(f"  非空值比例: {roe.notna().sum().sum() / (roe.shape[0] * roe.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "roe.parquet"
            roe_reset = roe.reset_index()
            roe_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return roe
    
    def factor_roa(self, save: bool = True) -> pd.DataFrame:
        """
        ROA (总资产收益率, Return on Assets)
        
        因果逻辑：
        ---------
        净利润与总资产的比率，反映企业整体资产的盈利能力。
        
        ROA vs ROE：
        - ROA不考虑财务杠杆，反映资产本身的使用效率
        - ROE高可能是高杠杆导致，ROA更能反映真实盈利质量
        - ROA持续高的公司往往有更可持续的竞争优势
        
        适用场景：
        - 比较不同杠杆率的公司（如制造业 vs 金融业）
        - 识别高ROE但低ROA的"杠杆陷阱"
        
        公式：
        ------
        ROA = 净利润_TTM / 总资产
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : ROA因子宽表
        """
        print("\n计算因子: ROA (总资产收益率)")
        
        net_profit = self._load_net_profit_ttm()
        tot_assets = self._load_tot_assets()
        
        net_profit, tot_assets = self._align_dataframes(net_profit, tot_assets)
        
        # 总资产<=0设为NaN
        tot_assets_safe = tot_assets.copy()
        tot_assets_safe[tot_assets_safe <= 0] = np.nan
        
        roa = net_profit / tot_assets_safe
        
        # 极端值处理
        roa[(roa > 1) | (roa < -0.5)] = np.nan
        
        print(f"  非空值比例: {roa.notna().sum().sum() / (roa.shape[0] * roa.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "roa.parquet"
            roa_reset = roa.reset_index()
            roa_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return roa
    
    def factor_roe_growth(self, save: bool = True) -> pd.DataFrame:
        """
        ROE增长率 (ROE Growth, YoY)
        
        因果逻辑：
        ---------
        ROE的同比变化，反映公司盈利能力的改善趋势。
        
        增长 vs 绝对值：
        - 高ROE但下降：可能竞争加剧或行业周期顶部
        - 中等ROE但上升：可能是改善中的公司，市场可能低估
        - ROE增长往往伴随股价加速（戴维斯双击）
        
        应用：
        - 结合ROE绝对值使用：ROE高+增长高 = 质量成长
        - 捕捉盈利改善的早期信号
        
        公式：
        ------
        ROE_Growth(t) = ROE(t) - ROE(t-252)  # 同比一年前（约252交易日）
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : ROE_Growth因子宽表
        """
        print("\n计算因子: ROE_Growth (ROE同比增长率)")
        
        # 先计算ROE
        roe = self.factor_roe(save=False)
        
        # 同比252个交易日（约一年）
        period = 252
        roe_growth = roe.diff(period)
        
        print(f"  非空值比例: {roe_growth.notna().sum().sum() / (roe_growth.shape[0] * roe_growth.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "roe_growth.parquet"
            roe_growth_reset = roe_growth.reset_index()
            roe_growth_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return roe_growth
    
    def factor_opm(self, save: bool = True) -> pd.DataFrame:
        """
        OPM (营业利润率, Operating Profit Margin)
        
        因果逻辑：
        ---------
        营业利润与营业收入的比率，反映核心业务的盈利能力。
        
        OPM vs NPM（净利润率）：
        - OPM只看主营业务，排除非经常性损益和投资收益
        - 更稳定，更能反映核心竞争力的变化
        - 高OPM通常意味着强定价权或成本优势
        
        公式：
        ------
        OPM = 营业利润_TTM / 营收_TTM
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : OPM因子宽表
        """
        print("\n计算因子: OPM (营业利润率)")
        
        oper_profit = self._load_oper_profit_ttm()
        
        # 营收_TTM，需要从financial_data加载
        revenue_file = self.processed_data_path / "financial_data" / "revenue_ttm.parquet"
        if not revenue_file.exists():
            raise FileNotFoundError(f"营收TTM数据不存在: {revenue_file}")
        
        revenue = pq.read_table(revenue_file).to_pandas()
        print(f"  营收TTM: {revenue.shape}")
        
        oper_profit, revenue = self._align_dataframes(oper_profit, revenue)
        
        # 营收<=0设为NaN
        revenue_safe = revenue.copy()
        revenue_safe[revenue_safe <= 0] = np.nan
        
        opm = oper_profit / revenue_safe
        
        # 极端值处理
        opm[(opm > 1) | (opm < -0.5)] = np.nan
        
        print(f"  非空值比例: {opm.notna().sum().sum() / (opm.shape[0] * opm.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "opm.parquet"
            opm_reset = opm.reset_index()
            opm_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return opm
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """
        批量计算所有盈利因子
        
        参数:
        -----
        factors : List[str], optional
            要计算的因子列表，默认计算所有
            
        返回:
        ------
        List[pd.DataFrame] : 因子宽表列表
        """
        available_factors = {
            'roe': self.factor_roe,
            'roa': self.factor_roa,
            'roe_growth': self.factor_roe_growth,
            'opm': self.factor_opm,
        }
        
        if factors is None:
            factors = list(available_factors.keys())
        
        print("=" * 60)
        print("盈利因子计算")
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
    print("测试 ProfitabilityFactors")
    print("=" * 60)
    
    try:
        pf = ProfitabilityFactors()
        pf.compute_all()
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 data_engine 准备数据")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
