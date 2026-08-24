# -*- coding: utf-8 -*-
"""
PIT (Point-in-Time) 对齐器

核心功能：将季度/半年度的财务数据对齐到日度交易日历。

逻辑：
-----
对于每个股票，按照公告日期 (m_anntime) 排序：
    财报1(2010Q1, 4/29公告) ─────── 财报2(2010Q2, 8/25公告) ─────── ...
         │                              │
         ▼                              ▼
    2010-04-30 ~ 2010-08-25       2010-08-26 ~ 2010-10-28
        使用财报1数据                 使用财报2数据

使用方法：
---------
    aligner = PITAligner(trading_calendar)
    daily_data = aligner.align(financial_df, 'm_anntime', 'cap_stk')

输入格式：
---------
financial_df: DataFrame with columns [m_anntime, field1, field2, ...]
    m_anntime: str, 格式 'YYYYMMDD' (如 '20100429')

输出格式：
---------
DataFrame with columns [date, field1, field2, ...]
    date: datetime.date 对象
"""

import datetime
from typing import List, Optional, Dict, Tuple
import numpy as np


class PITAligner:
    """
    PIT 对齐器
    
    将不规则的财务公告数据对齐到规则的交易日历。
    
    参数：
    -----
    trading_calendar : List[datetime.date]
        交易日列表（从 market_data/close.parquet 的 time 列获取）
    """
    
    def __init__(self, trading_calendar: List):
        """
        初始化对齐器
        
        参数：
        -----
        trading_calendar : List[datetime.date] 或 List[datetime.datetime]
            标准交易日历，用于对齐
        """
        # 统一转换为 date 类型（去除时间部分）
        self.trading_calendar = sorted([
            d.date() if isinstance(d, datetime.datetime) else d 
            for d in trading_calendar
        ])
        self.date_to_idx = {d: i for i, d in enumerate(self.trading_calendar)}
        
    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
        """
        解析日期字符串或日期对象为 date 对象
        
        支持格式：
        - 'YYYYMMDD' (如 '20100429')
        - 'YYYY-MM-DD' (如 '2010-04-29')
        - datetime.date 或 datetime.datetime 对象
        
        参数：
        -----
        date_str : str 或 date/datetime 对象
            日期字符串或对象
            
        返回：
        ------
        datetime.date 或 None（解析失败）
        """
        # 如果已经是 date 或 datetime 对象
        # 注意：datetime.datetime 是 datetime.date 的子类，先判断 datetime
        if isinstance(date_str, datetime.datetime):
            return date_str.date()
        if isinstance(date_str, datetime.date) and not isinstance(date_str, datetime.datetime):
            return date_str
        if date_str is None or date_str == '':
            return None
        
        try:
            if len(date_str) == 8:  # YYYYMMDD
                return datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            elif len(date_str) == 10 and date_str[4] == '-':  # YYYY-MM-DD
                return datetime.date.fromisoformat(date_str)
        except:
            return None
        
        return None
    
    def align(
        self,
        financial_records: List[Dict],
        date_field: str,
        value_fields: List[str],
        stock_code: Optional[str] = None
    ) -> List[Tuple]:
        """
        将财务记录对齐到交易日历
        
        参数：
        -----
        financial_records : List[Dict]
            财务记录列表，每个记录是一个字典
            例如: [{'m_anntime': '20100429', 'cap_stk': 3105434000.0, ...}, ...]
        date_field : str
            日期字段名（如 'm_anntime'）
        value_fields : List[str]
            需要对齐的数值字段列表
        stock_code : str, optional
            股票代码，用于日志
            
        返回：
        ------
        List[Tuple] : 对齐后的数据
            [(date, value1, value2, ...), ...]
            长度等于 trading_calendar
            
        注意：
        -----
        数据层已做两步清洗，这里只需对齐到交易日历。
        """
        if not financial_records:
            return [(d,) + tuple([np.nan] * len(value_fields)) for d in self.trading_calendar]
        
        # 解析并排序记录（数据层已清洗，无需去重）
        valid_records = []
        for record in financial_records:
            ann_date = self._parse_date(record.get(date_field))
            if ann_date:
                valid_records.append((ann_date, record))
        
        if not valid_records:
            return [(d,) + tuple([np.nan] * len(value_fields)) for d in self.trading_calendar]
        
        valid_records.sort(key=lambda x: x[0])
        
        # 对齐到交易日历
        result = []
        record_idx = 0
        current_record = valid_records[0]
        
        for trade_date in self.trading_calendar:
            while (record_idx + 1 < len(valid_records) and
                   trade_date > valid_records[record_idx + 1][0]):
                record_idx += 1
                current_record = valid_records[record_idx]
            
            if trade_date <= current_record[0] and record_idx == 0:
                values = [np.nan] * len(value_fields)
            else:
                record = current_record[1]
                values = []
                for field in value_fields:
                    val = record.get(field)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        values.append(np.nan)
                    else:
                        values.append(float(val))
            
            result.append((trade_date,) + tuple(values))
        
        return result
