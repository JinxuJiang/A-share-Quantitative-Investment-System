# -*- coding: utf-8 -*-
"""
质量因子家族 (Quality Factors)

质量因子反映企业的财务健康状况、盈利质量和运营效率。

因果逻辑：
- 质量溢价：高质量公司长期跑赢低质量公司（Novy-Marx, 2013）
- 防御属性：质量股在市场下跌时表现更稳健
- 复利效应：高质量公司能持续将利润再投资并获得高回报

质量维度：
1. 盈利能力：ROE、ROA、毛利率（已在profitability.py）
2. 盈利稳定性：利润波动率、应计项
3. 运营效率：资产周转率、存货周转
4. 财务安全：杠杆水平、流动性

参考：
- Novy-Marx (2013): The Other Side of Value: The Gross Profitability Premium
- Asness et al. (2019): Quality Minus Junk
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


class QualityFactors:
    """
    质量因子计算类
    
    包含：资产周转率、财务杠杆、利润质量等指标
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
        self._revenue_ttm = None
        self._tot_assets = None
        self._tot_shrhldr_eqy = None
        self._total_current_assets = None
        self._oper_profit_ttm = None
        self._net_profit_ttm = None
        self._close = None
        self._operating_cash_flow_ttm = None  # 新增
                                    
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
    
    def _load_tot_shrhldr_eqy(self):
        """加载净资产（归母股东权益）"""
        if self._tot_shrhldr_eqy is None:
            file_path = self.processed_data_path / "financial_data" / "tot_shrhldr_eqy.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"净资产数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._tot_shrhldr_eqy = table.to_pandas()
            print(f"  净资产: {self._tot_shrhldr_eqy.shape}")
        return self._tot_shrhldr_eqy
    
    def _load_total_current_assets(self):
        """加载流动资产"""
        if self._total_current_assets is None:
            file_path = self.processed_data_path / "financial_data" / "total_current_assets.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"流动资产数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._total_current_assets = table.to_pandas()
            print(f"  流动资产: {self._total_current_assets.shape}")
        return self._total_current_assets
    
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
    
    def _load_close(self):
        """加载收盘价（用于计算现金流收益率）"""
        if self._close is None:
            file_path = self.processed_data_path / "market_data" / "close.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"收盘价数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._close = table.to_pandas()
            print(f"  收盘价: {self._close.shape}")
        return self._close
    
    def _load_operating_cash_flow_ttm(self):
        """加载经营现金流_TTM（新增）"""
        if self._operating_cash_flow_ttm is None:
            file_path = self.processed_data_path / "financial_data" / "operating_cash_flow_ttm.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"经营现金流TTM数据不存在: {file_path}")
            
            table = pq.read_table(file_path)
            self._operating_cash_flow_ttm = table.to_pandas()
            print(f"  经营现金流TTM: {self._operating_cash_flow_ttm.shape}")
        return self._operating_cash_flow_ttm
    
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
    
    def factor_asset_turnover(self, save: bool = True) -> pd.DataFrame:
        """
        资产周转率 (Asset Turnover Ratio)
        
        因果逻辑：
        ---------
        营业收入与总资产的比率，反映资产使用效率。
        
        杜邦分析拆解 ROE：
        ROE = 净利润率 × 资产周转率 × 权益乘数
        
        高资产周转率意味着：
        - 单位资产创造更多收入（轻资产模式，如零售、服务）
        - 运营效率高，库存周转快
        - 与净利润率负相关（薄利多销 vs 厚利少销）
        
        适用场景：
        - 比较同行业公司（不同行业差异大）
        - 识别运营效率提升的公司
        
        公式：
        ------
        Asset_Turnover = 营收_TTM / 总资产
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 资产周转率宽表
        """
        print("\n计算因子: asset_turnover (资产周转率)")
        
        revenue = self._load_revenue_ttm()
        tot_assets = self._load_tot_assets()
        
        revenue, tot_assets = self._align_dataframes(revenue, tot_assets)
        
        # 总资产<=0设为NaN
        tot_assets_safe = tot_assets.copy()
        tot_assets_safe[tot_assets_safe <= 0] = np.nan
        
        # 计算周转率
        turnover = revenue / tot_assets_safe
        
        # 极端值处理（资产周转率通常在0-5之间）
        turnover[(turnover > 5) | (turnover < 0)] = np.nan
        
        print(f"  非空值比例: {turnover.notna().sum().sum() / (turnover.shape[0] * turnover.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "asset_turnover.parquet"
            turnover_reset = turnover.reset_index()
            turnover_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return turnover
    
    def factor_financial_leverage(self, save: bool = True) -> pd.DataFrame:
        """
        财务杠杆 (Financial Leverage)
        
        因果逻辑：
        ---------
        总资产与净资产的比率，反映公司的负债程度。
        
        杠杆的双刃剑：
        - 高杠杆放大ROE（ROE = ROA × 杠杆），但增加财务风险
        - 低杠杆更稳健，抗风险能力强，但可能错失增长机会
        - 金融危机时，高杠杆公司更容易破产
        
        作为反向因子：
        - 质量投资通常偏好低杠杆公司（防御性）
        - 低杠杆 = 高质量 = 低风险溢价
        
        注意：不同行业杠杆水平差异大（银行高、科技低），需行业中性化
        
        公式：
        ------
        Financial_Leverage = 总资产 / 净资产
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 财务杠杆宽表
        """
        print("\n计算因子: financial_leverage (财务杠杆)")
        
        tot_assets = self._load_tot_assets()
        net_assets = self._load_tot_shrhldr_eqy()
        
        tot_assets, net_assets = self._align_dataframes(tot_assets, net_assets)
        
        # 净资产<=0设为NaN（资不抵债）
        net_assets_safe = net_assets.copy()
        net_assets_safe[net_assets_safe <= 0] = np.nan
        
        # 计算杠杆
        leverage = tot_assets / net_assets_safe
        
        # 极端值处理（杠杆通常在1-10之间）
        leverage[(leverage > 20) | (leverage < 1)] = np.nan
        
        print(f"  非空值比例: {leverage.notna().sum().sum() / (leverage.shape[0] * leverage.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "financial_leverage.parquet"
            leverage_reset = leverage.reset_index()
            leverage_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return leverage
    
    def factor_profit_quality(self, save: bool = True) -> pd.DataFrame:
        """
        利润质量 (Profit Quality)
        
        因果逻辑：
        ---------
        营业利润与净利润的比率，反映核心利润的占比。
        
        利润构成分析：
        - 营业利润：主营业务的盈利能力，可持续、可复制
        - 非经常性损益：投资收益、政府补贴、资产处置等，一次性
        - 利润质量高 = 营业利润占比高 = 盈利可持续
        
        应用场景：
        - 识别利润操纵：净利润高但营业利润低（靠非经常性损益粉饰）
        - 价值投资：偏好利润质量高的公司（现金流质量通常也更好）
        
        公式：
        ------
        Profit_Quality = 营业利润_TTM / 净利润_TTM
        
        解释：
        - > 1: 有非经常性亏损（如资产减值）
        - = 1: 无非经常性损益
        - < 1: 有非经常性收益（如投资收益）
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 利润质量宽表
        """
        print("\n计算因子: profit_quality (利润质量)")
        
        oper_profit = self._load_oper_profit_ttm()
        net_profit = self._load_net_profit_ttm()
        
        oper_profit, net_profit = self._align_dataframes(oper_profit, net_profit)
        
        # 净利润=0设为NaN
        net_profit_safe = net_profit.copy()
        net_profit_safe[net_profit_safe == 0] = np.nan
        
        # 计算利润质量
        quality = oper_profit / net_profit_safe
        
        # 极端值处理（通常在-2到5之间）
        quality[(quality > 10) | (quality < -5)] = np.nan
        
        print(f"  非空值比例: {quality.notna().sum().sum() / (quality.shape[0] * quality.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "profit_quality.parquet"
            quality_reset = quality.reset_index()
            quality_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return quality
    
    def factor_current_asset_ratio(self, save: bool = True) -> pd.DataFrame:
        """
        流动资产占比 (Current Asset Ratio)
    
        流动资产与总资产的比率，反映资产结构。
        
        因果逻辑：
        ---------
        流动资产与总资产的比率，反映资产的流动性。
        
        注意：
        - 这里没有流动负债数据，用总资产代替
        - 实际流动比率 = 流动资产 / 流动负债
        - 这里的变形 = 流动资产 / 总资产，反映资产结构
        
        高流动资产占比意味着：
        - 资产流动性好，短期偿债能力强
        - 但可能资产利用效率低（现金过多未投资）
        - 轻资产模式（如软件、服务）占比通常较低
        
        公式：
        ------
        Current_Asset_Ratio = 流动资产 / 总资产
        
        参数:
        -----
        save : bool
            是否保存到文件
            
        返回:
        ------
        pd.DataFrame : 流动资产占比宽表
        """
        print("\n计算因子: current_asset_ratio (流动资产占比)")
        
        current_assets = self._load_total_current_assets()
        tot_assets = self._load_tot_assets()
        
        current_assets, tot_assets = self._align_dataframes(current_assets, tot_assets)
        
        # 总资产<=0设为NaN
        tot_assets_safe = tot_assets.copy()
        tot_assets_safe[tot_assets_safe <= 0] = np.nan
        
        # 计算流动比率
        ratio = current_assets / tot_assets_safe
        
        # 极端值处理（流动比率应在0-1之间）
        ratio[(ratio > 1.5) | (ratio < 0)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "current_ratio.parquet"
            ratio_reset = ratio.reset_index()
            ratio_reset.to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def factor_accrual(self, save: bool = True) -> pd.DataFrame:
        """
        Accrual (应计利润比) = (净利润_TTM - 经营现金流_TTM) / 总资产
        
        因果逻辑：高应计意味着利润质量差，Sloan (1996) 发现高应计股票未来收益更低
        """
        print("\n计算因子: accrual (应计利润比)")
        
        net_profit = self._load_net_profit_ttm()
        ocf = self._load_operating_cash_flow_ttm()
        tot_assets = self._load_tot_assets()
        
        net_profit, ocf, tot_assets = self._align_dataframes(net_profit, ocf, tot_assets)
        
        accrual = (net_profit - ocf) / tot_assets
        accrual[(accrual > 1) | (accrual < -1)] = np.nan
        
        print(f"  非空值比例: {accrual.notna().sum().sum() / (accrual.shape[0] * accrual.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "accrual.parquet"
            accrual.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return accrual
    
    def factor_cashflow_to_profit(self, save: bool = True) -> pd.DataFrame:
        """
        现金流利润比 = 经营现金流_TTM / 净利润_TTM
        
        因果逻辑：衡量利润含金量，>1说明现金流好于利润（预收款多）
        """
        print("\n计算因子: cashflow_to_profit (现金流利润比)")
        
        ocf = self._load_operating_cash_flow_ttm()
        net_profit = self._load_net_profit_ttm()
        
        ocf, net_profit = self._align_dataframes(ocf, net_profit)
        
        net_profit_safe = net_profit.copy()
        net_profit_safe[net_profit_safe == 0] = np.nan
        
        ratio = ocf / net_profit_safe
        ratio[(ratio > 10) | (ratio < -10)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "cashflow_to_profit.parquet"
            ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def factor_ocf_to_revenue(self, save: bool = True) -> pd.DataFrame:
        """
        收现率 = 经营现金流_TTM / 营收_TTM
        
        因果逻辑：反映收入质量，高收现率代表议价能力强
        """
        print("\n计算因子: ocf_to_revenue (收现率)")
        
        ocf = self._load_operating_cash_flow_ttm()
        revenue = self._load_revenue_ttm()
        
        ocf, revenue = self._align_dataframes(ocf, revenue)
        
        revenue_safe = revenue.copy()
        revenue_safe[revenue_safe <= 0] = np.nan
        
        ratio = ocf / revenue_safe
        ratio[(ratio > 5) | (ratio < -2)] = np.nan
        
        print(f"  非空值比例: {ratio.notna().sum().sum() / (ratio.shape[0] * ratio.shape[1]) * 100:.2f}%")
        
        if save:
            output_file = self.output_path / "ocf_to_revenue.parquet"
            ratio.reset_index().to_parquet(output_file, index=False)
            print(f"  已保存: {output_file}")
        
        return ratio
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """
        批量计算所有质量因子
        
        参数:
        -----
        factors : List[str], optional
            要计算的因子列表，默认计算所有
            
        返回:
        ------
        List[pd.DataFrame] : 因子宽表列表
        """
        available_factors = {
            'asset_turnover': self.factor_asset_turnover,
            'financial_leverage': self.factor_financial_leverage,
            'profit_quality': self.factor_profit_quality,
            'current_asset_ratio': self.factor_current_asset_ratio,
            'accrual': self.factor_accrual,
            'cashflow_to_profit': self.factor_cashflow_to_profit,
            'ocf_to_revenue': self.factor_ocf_to_revenue,
        }
        
        if factors is None:
            factors = list(available_factors.keys())
        
        print("=" * 60)
        print("质量因子计算")
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
    print("测试 QualityFactors")
    print("=" * 60)
    
    try:
        qf = QualityFactors()
        qf.compute_all()
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 data_engine 准备数据")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
