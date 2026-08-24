# -*- coding: utf-8 -*-
"""
行业数据加载器

将行业映射数据（industry_map.csv）转换为宽表格式（时间 × 股票），
与行情数据格式保持一致，便于后续因子计算和清洗。

输入：
------
raw_data/industry_map.csv
格式: order_book_id, industry_name

输出：
------
processed_data/financial_data/industry.parquet
格式: 宽表，index=datetime64[ns], columns=股票代码, values=行业名称

注意：
-----
行业数据是静态的（不随时间变化），但输出格式与行情数据一致（时间×股票），
方便后续按日期切片使用。
"""

import csv
import datetime
from pathlib import Path
from typing import Optional, Dict
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np


class IndustryLoader:
    """
    行业数据加载器
    
    功能：
    1. 读取 industry_map.csv
    2. 转换为宽表格式（时间 × 股票代码）
    3. 保存为 parquet 文件
    
    参数：
    -----
    raw_data_path : str, optional
        原始数据路径
    market_data_path : str, optional
        行情数据路径（用于获取交易日历）
    output_path : str, optional
        输出路径
    """
    
    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        market_data_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        初始化行业数据加载器
        """
        # 路径设置
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent  # 因子库
        project_root = factor_lib_root.parent  # 截面多因子模型
        
        if raw_data_path is None:
            self.raw_data_path = project_root / '01数据' / 'data' / 'tushare_data'
        else:
            self.raw_data_path = Path(raw_data_path)
            
        if market_data_path is None:
            self.market_data_path = factor_lib_root / 'processed_data' / 'market_data'
        else:
            self.market_data_path = Path(market_data_path)
            
        if output_path is None:
            self.output_path = factor_lib_root / 'processed_data' / 'financial_data'
        else:
            self.output_path = Path(output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.industry_file = self.raw_data_path / 'industry_map.csv'
    
    def _load_trading_calendar(self) -> list:
        """
        从 market_data/close.parquet 加载交易日历
        
        返回：
        ------
        list : 交易日列表（datetime.datetime 格式）
        """
        close_file = self.market_data_path / 'close.parquet'
        if not close_file.exists():
            raise FileNotFoundError(
                f"收盘价数据不存在: {close_file}\n"
                f"请先运行 main_prepare_market_data.py 准备行情数据"
            )
        
        table = pq.read_table(close_file, columns=['time'])
        time_list = table.column('time').to_pylist()
        
        # 确保是 datetime 格式
        calendar = []
        for t in time_list:
            if isinstance(t, datetime.date) and not isinstance(t, datetime.datetime):
                calendar.append(datetime.datetime.combine(t, datetime.time.min))
            else:
                calendar.append(t)
        
        print(f"已加载交易日历: {len(calendar)} 个交易日")
        print(f"  日期范围: {calendar[0]} ~ {calendar[-1]}")
        return calendar
    
    def load_industry_map(self) -> Dict[str, str]:
        """
        读取行业映射 CSV 文件
        
        返回：
        ------
        Dict[str, str] : {股票代码: 行业名称}
        """
        if not self.industry_file.exists():
            raise FileNotFoundError(f"行业数据文件不存在: {self.industry_file}")
        
        industry_map = {}
        
        # 尝试不同编码读取
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                with open(self.industry_file, 'r', encoding=encoding) as f:
                    reader = csv.reader(f)
                    header = next(reader)  # 跳过表头
                    
                    print(f"行业文件表头: {header}")
                    
                    for row in reader:
                        if len(row) >= 2:
                            stock_code = row[0].strip()
                            industry_name = row[1].strip()
                            industry_map[stock_code] = industry_name
                
                print(f"成功读取行业数据，共 {len(industry_map)} 只股票")
                return industry_map
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"使用编码 {encoding} 读取失败: {e}")
                continue
        
        raise ValueError("无法读取行业数据文件，请检查文件编码")
    
    def prepare_industry_data(self, overwrite: bool = False) -> Path:
        """
        准备行业数据宽表
        
        格式：
        ------
        宽表，行为交易日（datetime64[ns]），列为股票代码，值为行业名称
        
        参数：
        -----
        overwrite : bool
            是否覆盖已存在的文件
            
        返回：
        ------
        Path : 输出文件路径
        """
        output_file = self.output_path / 'industry.parquet'
        
        if output_file.exists() and not overwrite:
            print(f"文件已存在，跳过: {output_file}")
            return output_file
        
        # 1. 读取交易日历
        trading_calendar = self._load_trading_calendar()
        
        # 2. 读取行业映射
        industry_map = self.load_industry_map()
        
        # 3. 构建宽表（使用 pandas 更方便处理字符串）
        # 创建 DataFrame：每行是一个交易日，每列是一个股票
        stock_codes = list(industry_map.keys())
        industry_names = [industry_map[code] for code in stock_codes]
        
        # 广播：每个交易日，所有股票的行业都一样
        # 使用 numpy 广播
        data = np.tile(industry_names, (len(trading_calendar), 1))
        
        df = pd.DataFrame(
            data,
            index=pd.DatetimeIndex(trading_calendar, name='time'),
            columns=stock_codes
        )
        
        # 4. 转换为 PyArrow Table 并保存
        # 将 index 转为列
        df_reset = df.reset_index()
        
        # 构建 PyArrow Table
        arrays = [pa.array(df_reset['time'], type=pa.timestamp('ns'))]
        names = ['time']
        
        for col in df_reset.columns[1:]:
            arrays.append(pa.array(df_reset[col], type=pa.string()))
            names.append(col)
        
        table = pa.table(arrays, names=names)
        
        pq.write_table(table, output_file)
        
        print(f"已保存: {output_file} ({table.num_rows} 行 × {table.num_columns} 列)")
        print(f"  行业数量: {len(set(industry_names))}")
        print(f"  股票数量: {len(stock_codes)}")
        print(f"  前5个行业: {list(set(industry_names))[:5]}")
        
        return output_file


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试 IndustryLoader")
    print("=" * 60)
    
    try:
        loader = IndustryLoader()
        output = loader.prepare_industry_data(overwrite=True)
        print(f"\n✓ 行业数据准备完成: {output}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
