# -*- coding: utf-8 -*-
"""
波动率因子家族 (Volatility Factors)

波动率因子反映股票价格波动的幅度和特征，是风险管理的重要指标。

因果逻辑：
- 低波动异象：理论上高风险应高收益，但实证发现低波动股票长期跑赢
- 波动率聚类：高波动后往往跟随高波动（GARCH效应）
- 波动率与收益负相关：高波动往往对应高估值、高投机性

参考：
- Ang et al. (2006): The Cross-Section of Volatility and Expected Returns
- Blitz & van Vliet (2007): The Volatility Effect
"""

from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


class VolatilityFactors:
    """
    波动率因子计算类
    
    功能：
    1. 从 market_data 加载价格数据（open/high/low/close）
    2. 计算各种波动率因子
    3. 保存为独立的 parquet 文件
    
    参数：
    -----
    market_data_path : str, optional
        市场数据路径，默认使用 processed_data/market_data/
    output_path : str, optional
        因子输出路径，默认使用 processed_data/technical_factors/
    """
    
    def __init__(
        self,
        market_data_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        初始化波动率因子计算器
        """
        # 路径设置
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent.parent  # 因子库
        
        if market_data_path is None:
            self.market_data_path = factor_lib_root / 'processed_data' / 'market_data'
        else:
            self.market_data_path = Path(market_data_path)
            
        if output_path is None:
            self.output_path = factor_lib_root / 'processed_data' / 'factors' / 'technical'
        else:
            self.output_path = Path(output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 缓存数据
        self._close_table: Optional[pa.Table] = None
        self._high_table: Optional[pa.Table] = None
        self._low_table: Optional[pa.Table] = None
        self._dates = None
        self._stocks = None
    
    def _load_price_data(self, field: str) -> pa.Table:
        """
        加载价格数据
        
        参数：
        -----
        field : str
            价格字段名：'close', 'high', 'low', 'open'
            
        返回：
        ------
        pa.Table : 价格宽表
        """
        cache_attr = f'_{field}_table'
        
        if getattr(self, cache_attr) is None:
            price_file = self.market_data_path / f'{field}.parquet'
            if not price_file.exists():
                raise FileNotFoundError(
                    f"{field}数据不存在: {price_file}\n"
                    f"请先运行 main_prepare_market_data.py 准备数据"
                )
            setattr(self, cache_attr, pq.read_table(price_file))
            print(f"已加载{field}数据")
        
        return getattr(self, cache_attr)
    
    def _get_numpy_matrix(self, field: str = 'close') -> tuple:
        """
        将 PyArrow Table 转换为 NumPy 矩阵
        
        参数：
        -----
        field : str, default='close'
            要加载的价格字段
        
        返回：
        ------
        (dates, stocks, matrix) : 日期列表、股票列表、价格矩阵
        """
        # 统一使用 close 的列结构作为基准
        if self._dates is None:
            table = self._load_price_data('close')
            columns = table.column_names
            self._dates = columns[0]  # time
            self._stocks = columns[1:]  # 股票代码
        
        # 加载指定字段的数据
        table = self._load_price_data(field)
        
        # 转换为 NumPy 矩阵
        data_list = []
        for stock in self._stocks:
            col_data = table.column(stock).to_pylist()
            data_list.append(col_data)
        
        # shape: (n_stocks, n_dates) -> 转置为 (n_dates, n_stocks)
        matrix = np.array(data_list, dtype=np.float64).T
        
        # 获取日期数组
        dates = table.column(self._dates).to_pylist()
        
        return dates, self._stocks, matrix
    
    def _save_factor(self, factor_name: str, factor_matrix: np.ndarray, dates: list, stocks: list):
        """
        保存因子数据为 parquet 文件
        
        参数：
        -----
        factor_name : str
            因子名称
        factor_matrix : np.ndarray
            因子值矩阵，shape: (n_dates, n_stocks)
        dates : list
            日期列表
        stocks : list
            股票代码列表
        """
        arrays = [pa.array(dates, type=pa.timestamp('ns'))]
        names = ['time']
        
        for i, stock in enumerate(stocks):
            col_data = factor_matrix[:, i]
            col_list = [
                None if (v != v or np.isinf(v)) else float(v)
                for v in col_data
            ]
            arrays.append(pa.array(col_list, type=pa.float64()))
            names.append(stock)
        
        factor_table = pa.table(arrays, names=names)
        
        output_file = self.output_path / f"{factor_name}.parquet"
        pq.write_table(factor_table, output_file)
        print(f"已保存因子: {output_file} ({factor_table.num_rows} 行 × {factor_table.num_columns} 列)")
        
        return output_file
    
    def _compute_daily_return(self, close_matrix: np.ndarray) -> np.ndarray:
        """
        计算日收益率矩阵
        
        参数：
        -----
        close_matrix : np.ndarray
            收盘价矩阵，shape: (n_dates, n_stocks)
            
        返回：
        ------
        np.ndarray : 日收益率矩阵
        """
        daily_ret = np.full_like(close_matrix, np.nan)
        n_dates = close_matrix.shape[0]
        
        if n_dates > 1:
            mask = close_matrix[:-1] > 0
            daily_ret[1:] = np.where(mask, close_matrix[1:] / close_matrix[:-1] - 1, np.nan)
        
        return daily_ret
    
    def factor_std20(self, save: bool = True) -> Optional[Path]:
        """
        20日波动率因子 (20-Day Volatility)
        
        因果逻辑：
        ---------
        过去20个交易日日收益率的标准差，反映股票的短期波动风险。
        
        根据 "低波动异象"（Low Volatility Anomaly）：
        - 传统金融理论认为高风险应获得高收益补偿
        - 但实证发现低波动股票长期跑赢了高波动股票
        - 可能原因：
          1. 杠杆限制：投资者无法充分杠杆化低风险资产
          2. 彩票偏好：投资者喜欢高波动股票的"彩票特性"
          3. 机构行为：benchmark约束导致追逐高波动成长股
        
        在截面选股中，std20 通常作为反向因子使用
        （选择波动率最低的分组，规避高风险股票）。
        
        计算方法：
        ---------
        1. 计算日收益率: r(t) = close(t)/close(t-1) - 1
        2. 滚动20日标准差: std20(t) = std(r(t-19:t), ddof=1)
        
        参数：
        -----
        save : bool
            是否保存到文件，默认 True
            
        返回：
        ------
        Path : 输出文件路径（如果 save=True）
        np.ndarray : 因子矩阵（如果 save=False）
        """
        print("\n计算因子: std20 (20日波动率)")
        
        dates, stocks, close_matrix = self._get_numpy_matrix('close')
        n_dates, n_stocks = close_matrix.shape
        
        print(f"数据维度: {n_dates} 天 × {n_stocks} 只股票")
        
        # 计算日收益率
        daily_ret = self._compute_daily_return(close_matrix)
        
        # 滚动20日标准差
        period = 20
        factor_matrix = np.full_like(close_matrix, np.nan)
        
        for i in range(period - 1, n_dates):
            window = daily_ret[i - period + 1:i + 1]
            factor_matrix[i] = np.nanstd(window, axis=0, ddof=1)
        
        # 年化（可选）：std20 * sqrt(252)
        # factor_matrix = factor_matrix * np.sqrt(252)
        
        print(f"因子计算完成，非NaN值比例: {np.sum(~np.isnan(factor_matrix)) / factor_matrix.size:.2%}")
        
        if save:
            return self._save_factor('std20', factor_matrix, dates, stocks)
        else:
            return factor_matrix
    
    def factor_std60(self, save: bool = True) -> Optional[Path]:
        """
        60日波动率因子 (60-Day Volatility)
        
        因果逻辑：
        ---------
        过去60个交易日日收益率的标准差，反映股票的中期波动风险。
        
        相比 std20（短期），std60 能更好地反映：
        - 结构性波动特征（而非短期噪音）
        - 公司基本面的不确定性
        - 市场参与者的长期分歧程度
        
        std60 与 std20 的比值（volatility_regime）可以识别
        波动率的状态变化（当前波动 vs 历史正常波动）。
        
        计算方法：
        ---------
        1. 计算日收益率: r(t) = close(t)/close(t-1) - 1
        2. 滚动60日标准差: std60(t) = std(r(t-59:t), ddof=1)
        
        参数：
        -----
        save : bool
            是否保存到文件，默认 True
            
        返回：
        ------
        Path : 输出文件路径（如果 save=True）
        np.ndarray : 因子矩阵（如果 save=False）
        """
        print("\n计算因子: std60 (60日波动率)")
        
        dates, stocks, close_matrix = self._get_numpy_matrix('close')
        n_dates, n_stocks = close_matrix.shape
        
        print(f"数据维度: {n_dates} 天 × {n_stocks} 只股票")
        
        # 计算日收益率
        daily_ret = self._compute_daily_return(close_matrix)
        
        # 滚动60日标准差
        period = 60
        factor_matrix = np.full_like(close_matrix, np.nan)
        
        for i in range(period - 1, n_dates):
            window = daily_ret[i - period + 1:i + 1]
            factor_matrix[i] = np.nanstd(window, axis=0, ddof=1)
        
        print(f"因子计算完成，非NaN值比例: {np.sum(~np.isnan(factor_matrix)) / factor_matrix.size:.2%}")
        
        if save:
            return self._save_factor('std60', factor_matrix, dates, stocks)
        else:
            return factor_matrix
    
    def factor_atr20(self, save: bool = True) -> Optional[Path]:
        """
        20日平均真实波幅 (Average True Range, ATR)
        
        因果逻辑：
        ---------
        ATR 是技术分析中衡量波动率的经典指标，由 J. Welles Wilder 提出。
        
        与标准差不同，ATR 考虑了价格的跳空和日内波动：
        - 真实波幅 TR = max(high-low, |high-close_prev|, |low-close_prev|)
        - 反映了市场参与者的日内交易活跃度和价格分歧
        
        ATR 的应用：
        - 止损设置：ATR 倍数止损（如 2x ATR）
        - 仓位管理：高 ATR 股票减仓，低 ATR 股票加仓
        - 趋势确认：ATR 放大常伴随趋势突破或反转
        
        在截面选股中，低 ATR 股票通常代表：
        - 市场共识度高，价格波动小
        - 机构持仓稳定，投机性低
        - 长期持有体验更好（低回撤）
        
        计算方法：
        ---------
        1. 计算真实波幅 TR:
           TR(t) = max(
               high(t) - low(t),
               |high(t) - close(t-1)|,
               |low(t) - close(t-1)|
           )
        2. 滚动20日平均: ATR20(t) = mean(TR(t-19:t))
        
        参数：
        -----
        save : bool
            是否保存到文件，默认 True
            
        返回：
        ------
        Path : 输出文件路径（如果 save=True）
        np.ndarray : 因子矩阵（如果 save=False）
        """
        print("\n计算因子: atr20 (20日平均真实波幅)")
        
        # 加载三种价格数据
        dates, stocks, close_matrix = self._get_numpy_matrix('close')
        _, _, high_matrix = self._get_numpy_matrix('high')
        _, _, low_matrix = self._get_numpy_matrix('low')
        
        n_dates, n_stocks = close_matrix.shape
        print(f"数据维度: {n_dates} 天 × {n_stocks} 只股票")
        
        # 计算真实波幅 TR
        tr_matrix = np.full_like(close_matrix, np.nan)
        
        for i in range(1, n_dates):
            # TR = max(high-low, |high-close_prev|, |low-close_prev|)
            range1 = high_matrix[i] - low_matrix[i]  # 当日振幅
            range2 = np.abs(high_matrix[i] - close_matrix[i-1])  # 高-昨收
            range3 = np.abs(low_matrix[i] - close_matrix[i-1])   # 低-昨收
            
            tr_matrix[i] = np.maximum(np.maximum(range1, range2), range3)
        
        # 滚动20日平均
        period = 20
        factor_matrix = np.full_like(close_matrix, np.nan)
        
        for i in range(period, n_dates):
            window = tr_matrix[i - period + 1:i + 1]
            factor_matrix[i] = np.nanmean(window, axis=0)
        
        print(f"因子计算完成，非NaN值比例: {np.sum(~np.isnan(factor_matrix)) / factor_matrix.size:.2%}")
        
        if save:
            return self._save_factor('atr20', factor_matrix, dates, stocks)
        else:
            return factor_matrix
    
    def factor_volatility_regime(self, save: bool = True) -> Optional[Path]:
        """
        波动率状态因子 (Volatility Regime)
        
        因果逻辑：
        ---------
        当前波动率相对于历史正常波动率的比率，识别异常波动状态。
        
        公式: volatility_regime = std20 / std60
        
        解读：
        - ratio > 1.2: 当前波动放大（可能是事件驱动、公告期、危机）
        - ratio < 0.8: 当前波动压缩（可能是盘整期、等待突破）
        - ratio ≈ 1.0: 波动正常
        
        应用场景：
        1. 风险规避：规避波动率急剧放大的股票（不确定性高）
        2. 波动率回归：高 ratio 股票波动可能回归常态（均值回归）
        3. 事件驱动：ratio 突变常伴随信息披露
        
        注意：这是一个相对因子，与绝对波动率（std20）结合使用效果更佳。
        
        计算方法：
        ---------
        volatility_regime(t) = std20(t) / std60(t)
        
        当 std60 = 0 或缺失时，结果为 NaN。
        
        参数：
        -----
        save : bool
            是否保存到文件，默认 True
            
        返回：
        ------
        Path : 输出文件路径（如果 save=True）
        np.ndarray : 因子矩阵（如果 save=False）
        """
        print("\n计算因子: volatility_regime (波动率状态: std20/std60)")
        
        dates, stocks, close_matrix = self._get_numpy_matrix('close')
        n_dates, n_stocks = close_matrix.shape
        
        print(f"数据维度: {n_dates} 天 × {n_stocks} 只股票")
        
        # 计算日收益率
        daily_ret = self._compute_daily_return(close_matrix)
        
        # 计算 std20 和 std60
        std20_matrix = np.full_like(close_matrix, np.nan)
        std60_matrix = np.full_like(close_matrix, np.nan)
        
        # 计算滚动标准差
        for i in range(59, n_dates):  # 从60日开始
            # std20
            if i >= 19:
                window20 = daily_ret[i - 19:i + 1]
                std20_matrix[i] = np.nanstd(window20, axis=0, ddof=1)
            
            # std60
            window60 = daily_ret[i - 59:i + 1]
            std60_matrix[i] = np.nanstd(window60, axis=0, ddof=1)
        
        # 计算比率
        factor_matrix = np.full_like(close_matrix, np.nan)
        mask = (std60_matrix > 0) & ~np.isnan(std60_matrix)
        factor_matrix[mask] = std20_matrix[mask] / std60_matrix[mask]
        
        print(f"因子计算完成，非NaN值比例: {np.sum(~np.isnan(factor_matrix)) / factor_matrix.size:.2%}")
        
        if save:
            return self._save_factor('volatility_regime', factor_matrix, dates, stocks)
        else:
            return factor_matrix
    
    def compute_all(self, factors: Optional[List[str]] = None):
        """
        批量计算所有波动率因子
        
        参数：
        -----
        factors : List[str], optional
            要计算的因子列表，默认计算所有
            
        返回：
        ------
        List[Path] : 输出文件路径列表
        """
        available_factors = {
            'std20': self.factor_std20,
            'std60': self.factor_std60,
            'atr20': self.factor_atr20,
            'volatility_regime': self.factor_volatility_regime,
        }
        
        if factors is None:
            factors = list(available_factors.keys())
        
        output_files = []
        
        print("=" * 60)
        print("波动率因子计算")
        print("=" * 60)
        
        for factor_name in factors:
            if factor_name in available_factors:
                try:
                    result = available_factors[factor_name](save=True)
                    if result is not None:
                        output_files.append(result)
                except Exception as e:
                    print(f"计算因子 {factor_name} 失败: {e}")
            else:
                print(f"未知因子: {factor_name}，可用因子: {list(available_factors.keys())}")
        
        print(f"\n全部完成！共计算 {len(output_files)} 个因子")
        return output_files


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试 VolatilityFactors")
    print("=" * 60)
    
    try:
        vf = VolatilityFactors()
        output_files = vf.compute_all()
        
        print("\n✓ 测试完成，生成文件:")
        for f in output_files:
            print(f"  - {f}")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 main_prepare_market_data.py 准备数据")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
