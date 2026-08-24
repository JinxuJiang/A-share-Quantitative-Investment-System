# -*- coding: utf-8 -*-
"""
行情数据加载器

将原始行情数据（个股时间序列）转换为宽表格式（日期×股票），
便于截面因子计算。

原始数据结构（长格式）：
    time, open, high, low, close, volume, amount, preClose, suspendFlag
    
转换后数据结构（宽表格式）：
    行索引：日期（datetime）
    列：股票代码（如 000001.SZ, 000002.SZ...）
    值：对应字段的数值
"""

import os
import datetime
from pathlib import Path
from typing import List, Optional, Union
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


class MarketDataLoader:
    """
    市场数据加载器
    
    功能：
    1. 读取 tushare_data/market_data/ 下的个股 parquet 文件
    2. 提取指定字段（open/high/low/close/volume/amount/preClose）
    3. 拼接成宽表（日期×股票）
    4. 保存到 processed_data/market_data/
    
    参数：
    -----
    raw_data_path : str
        原始数据路径，默认 '01数据/data/tushare_data/market_data/'
    output_path : str
        输出路径，默认 '因子库/processed_data/market_data/'
    """
    
    # 支持的字段列表
    VALID_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose', 'suspendFlag']
    
    def __init__(
        self, 
        raw_data_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        初始化数据加载器
        
        参数：
        -----
        raw_data_path : str, optional
            原始数据路径，默认使用 '01数据/data/tushare_data/market_data/'
        output_path : str, optional  
            输出路径，默认使用相对路径 '因子库/processed_data/market_data/'
        """
        # 获取项目根目录
        # 当前文件: 因子库/src/data_engine/market_data_loader.py
        # 项目根目录: 截面多因子模型 (因子库的父目录)
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent  # 因子库
        project_root = factor_lib_root.parent  # 截面多因子模型
        
        # 设置默认路径
        if raw_data_path is None:
            self.raw_data_path = project_root / '01数据' / 'data' / 'tushare_data' / 'market_data'
        else:
            self.raw_data_path = Path(raw_data_path)
            
        if output_path is None:
            self.output_path = factor_lib_root / 'processed_data' / 'market_data'
        else:
            self.output_path = Path(output_path)
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 存储股票列表
        self.stock_list: List[str] = []
        
    def _get_stock_files(self) -> List[Path]:
        """
        获取所有股票数据文件列表
        
        返回：
        ------
        List[Path] : 所有 .parquet 文件路径列表
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"原始数据路径不存在: {self.raw_data_path}")
        
        files = sorted(self.raw_data_path.glob('*.parquet'))
        print(f"找到 {len(files)} 个股票文件")
        return files
    
    def _read_single_stock(
        self, 
        file_path: Path, 
        field: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pa.Table]:
        """
        读取单个股票的指定字段数据
        
        参数：
        -----
        file_path : Path
            股票数据文件路径
        field : str
            要读取的字段名
        start_date : str, optional
            开始日期，格式 'YYYY-MM-DD'
        end_date : str, optional
            结束日期，格式 'YYYY-MM-DD'
            
        返回：
        ------
        pa.Table : 包含 time 和 field 两列的 PyArrow Table
        """
        try:
            # 只读取需要的列，减少内存占用
            columns = ['time', field]
            table = pq.read_table(file_path, columns=columns)
            
            # 日期过滤
            if start_date or end_date:
                time_col = table.column('time')
                # 转换为 pandas 进行日期过滤（更灵活）
                # 这里用 pyarrow 的 compute 函数
                import pyarrow.compute as pc
                
                if start_date:
                    mask = pc.greater_equal(
                        pc.cast(time_col, pa.date32()),
                        pa.scalar(np.datetime64(start_date).astype('datetime64[D]').astype('int32'), pa.date32())
                    )
                    table = table.filter(mask)
                    
                if end_date:
                    mask = pc.less_equal(
                        pc.cast(time_col, pa.date32()),
                        pa.scalar(np.datetime64(end_date).astype('datetime64[D]').astype('int32'), pa.date32())
                    )
                    table = table.filter(mask)
            
            return table
            
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None
    
    def _build_wide_table(
        self, 
        field: str,
        stock_files: List[Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pa.Table:
        """
        构建宽表：将所有股票的指定字段拼接成一张表
        
        参数：
        -----
        field : str
            字段名（如 'close', 'volume'）
        stock_files : List[Path]
            股票文件列表
        start_date, end_date : str, optional
            日期范围
            
        返回：
        ------
        pa.Table : 宽表，列名为 [time, 股票代码1, 股票代码2, ...]
        """
        print(f"\n正在处理字段: {field}")
        
        # 收集所有数据
        data_dict = {}  # {stock_code: (times, values)}
        all_times = set()
        
        for i, file_path in enumerate(stock_files):
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(stock_files)} 个股票...")
            
            stock_code = file_path.stem  # 000001.SZ
            table = self._read_single_stock(file_path, field, start_date, end_date)
            
            if table is None or len(table) == 0:
                continue
            
            # 提取时间和数值
            raw_times = table.column('time').to_pylist()
            # 毫秒时间戳转换为 datetime 对象 (1262534400000 -> 2010-01-04 00:00:00)
            # 使用 datetime.datetime 而不是 date，保存为 datetime64[ns]
            times = [datetime.datetime.fromtimestamp(t / 1000) for t in raw_times]
            values = table.column(field).to_pylist()
            
            data_dict[stock_code] = (times, values)
            all_times.update(times)
        
        if not data_dict:
            raise ValueError(f"没有有效数据用于字段 {field}")
        
        print(f"  共 {len(data_dict)} 个股票有数据，时间范围包含 {len(all_times)} 个日期")
        
        # 构建统一的日期索引
        sorted_times = sorted(all_times)
        time_to_idx = {t: i for i, t in enumerate(sorted_times)}
        
        # 构建 PyArrow Table
        # 策略：为每列创建数组，然后组合成表
        # time 列使用 timestamp[ns] 类型，便于后续作为 datetime64 索引
        arrays = [pa.array(sorted_times, type=pa.timestamp('ns'))]  # 第一列是时间
        names = ['time']
        
        for stock_code, (times, values) in data_dict.items():
            # 创建一个与 sorted_times 对齐的数组
            aligned_values = [None] * len(sorted_times)
            for t, v in zip(times, values):
                aligned_values[time_to_idx[t]] = v
            
            # 根据字段类型选择合适的类型
            if field in ['volume']:
                # volume 可能是整数
                arrays.append(pa.array(aligned_values, type=pa.int64()))
            else:
                # 其他用浮点数
                arrays.append(pa.array(aligned_values, type=pa.float64()))
            
            names.append(stock_code)
        
        wide_table = pa.table(arrays, names=names)
        return wide_table
    
    def prepare_field(
        self, 
        field: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False
    ) -> Path:
        """
        准备单个字段的宽表数据
        
        参数：
        -----
        field : str
            字段名，必须是 VALID_FIELDS 之一
        start_date, end_date : str, optional
            日期范围，格式 'YYYY-MM-DD'
        overwrite : bool
            是否覆盖已存在的文件
            
        返回：
        ------
        Path : 输出文件路径
        """
        if field not in self.VALID_FIELDS:
            raise ValueError(f"不支持的字段: {field}，支持的字段: {self.VALID_FIELDS}")
        
        output_file = self.output_path / f"{field}.parquet"
        
        # 检查是否已存在
        if output_file.exists() and not overwrite:
            print(f"文件已存在，跳过: {output_file}")
            return output_file
        
        # 获取股票文件列表
        stock_files = self._get_stock_files()
        
        # 构建宽表
        wide_table = self._build_wide_table(field, stock_files, start_date, end_date)
        
        # 保存
        pq.write_table(wide_table, output_file)
        print(f"已保存: {output_file} (形状: {wide_table.num_rows} 行 × {wide_table.num_columns} 列)")
        
        return output_file
    
    def prepare_all_fields(
        self,
        fields: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False
    ) -> List[Path]:
        """
        批量准备所有字段的宽表数据
        
        参数：
        -----
        fields : List[str], optional
            要处理的字段列表，默认处理所有 VALID_FIELDS
        start_date, end_date : str, optional
            日期范围
        overwrite : bool
            是否覆盖已存在文件
            
        返回：
        ------
        List[Path] : 所有输出文件路径列表
        """
        if fields is None:
            fields = self.VALID_FIELDS
        
        output_files = []
        
        for field in fields:
            try:
                output_file = self.prepare_field(field, start_date, end_date, overwrite)
                output_files.append(output_file)
            except Exception as e:
                print(f"处理字段 {field} 失败: {e}")
                continue
        
        print(f"\n全部完成！共处理 {len(output_files)} 个字段")
        return output_files
    
    def load(self, field: str) -> pa.Table:
        """
        加载已准备好的宽表数据
        
        参数：
        -----
        field : str
            字段名
            
        返回：
        ------
        pa.Table : 宽表数据
        """
        file_path = self.output_path / f"{field}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}，请先调用 prepare_field")
        
        return pq.read_table(file_path)
    
    def get_stock_list(self) -> List[str]:
        """
        获取股票代码列表
        
        返回：
        ------
        List[str] : 股票代码列表
        """
        if not self.stock_list:
            stock_files = self._get_stock_files()
            self.stock_list = [f.stem for f in stock_files]
        return self.stock_list
