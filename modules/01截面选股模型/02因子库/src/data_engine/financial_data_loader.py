# -*- coding: utf-8 -*-
"""
财务数据加载器

将原始财务数据（季度报告）转换为 PIT 对齐后的日度宽表。

处理流程：
---------
1. 读取 market_data/close.parquet 获取交易日历
2. 从 tushare_data/financial_full/ 读取三张季度分区表（income/balancesheet/cashflow）
3. 版本事件：保留不同 f_ann_date，同日冲突保守选择 type 5 原始值
4. 对每个股票：
   - 三张表分别按自身 f_ann_date 构建 PIT 事件流，下一交易日生效
   - 如需TTM：严格按连续报告期计算（累计值→单季度→4季度求和）
   - PIT对齐到交易日历
5. 拼接成宽表（行：日期，列：股票代码）
6. 保存为 parquet 文件

输出字段：
---------
期末值字段（资产负债表）:
    - cap_stk: 总股本
    - tot_assets: 总资产
    - tot_shrhldr_eqy: 归属于母公司股东权益合计
    - total_current_assets: 流动资产
    - tot_liab: 总负债
    - total_current_liability: 流动负债
    - cash_equivalents: 货币资金

TTM字段（利润表/现金流量表，4季度滚动求和）:
    - net_profit_ttm: 归母净利润_TTM
    - revenue_ttm: 营业收入_TTM
    - oper_profit_ttm: 营业利润_TTM
    - operating_cash_flow_ttm: 经营现金流_TTM
    - capex_ttm: 购建固定资产现金_TTM

注意：
- ROE_TTM 后续从 net_profit_ttm / tot_shrhldr_eqy 计算得出
- TTM计算逻辑：先算季度TTM，再PIT对齐到日度（确保同财报期内TTM值不变）
- [2026-07-28] 输入侧从 QMT per-stock 文件切换为 Tushare 季度分区表，
  输出契约（processed_data/financial_data/*.parquet 宽表）不变
"""

import datetime
import gc
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd

from pit_aligner import PITAligner


class FinancialDataLoader:
    """
    财务数据加载器

    功能：
    1. 加载原始财务数据（Tushare 季度分区表）
    2. 计算 TTM 字段（在季度层面正确计算）
    3. PIT 对齐到交易日历
    4. 输出宽表格式

    参数：
    -----
    raw_data_path : str, optional
        原始财务数据路径（tushare_data/financial_full）
    market_data_path : str, optional
        行情数据路径（用于获取交易日历）
    output_path : str, optional
        输出路径
    """

    # 需要处理的字段配置
    # format: (输出字段名, 是否需要TTM, 源字段名)
    # 源字段名为内部统一命名（沿用旧 QMT 口径名），由 _TUSHARE_COLUMN_MAP 映射到 Tushare 列
    FIELD_CONFIG = [
        # 资产负债表 - 期末值，不需要TTM
        ('cap_stk', False, 'cap_stk'),
        ('tot_assets', False, 'tot_assets'),
        ('tot_shrhldr_eqy', False, 'tot_shrhldr_eqy_excl_min_int'),  # 归属于母公司股东权益
        ('total_current_assets', False, 'total_current_assets'),

        # 利润表 - 需要TTM（4季度滚动求和）
        ('net_profit', True, 'net_profit_excl_min_int_inc'),  # 归母净利润
        ('revenue', True, 'revenue'),
        ('oper_profit', True, 'oper_profit'),

        # 资产负债表（补充）
        ('tot_liab', False, 'tot_liab'),                          # 总负债
        ('total_current_liability', False, 'total_current_liability'),  # 流动负债
        ('cash_equivalents', False, 'cash_equivalents'),          # 货币资金

        # 现金流量表（TTM）
        ('operating_cash_flow', True, 'net_cash_flows_oper_act'), # 经营现金流
        ('capex', True, 'cash_pay_acq_const_fiolta'),             # 购建固定资产现金
    ]

    # Tushare 分区表 → 内部统一字段名 的映射
    # format: 表名 -> {Tushare列名: 内部字段名}
    _TUSHARE_COLUMN_MAP = {
        'income': {
            'n_income_attr_p': 'net_profit_excl_min_int_inc',  # 归母净利润
            'revenue': 'revenue',
            'operate_profit': 'oper_profit',
        },
        'balancesheet': {
            'total_share': 'cap_stk',                          # 总股本（股，已验证与QMT量纲一致）
            'total_assets': 'tot_assets',
            'total_hldr_eqy_exc_min_int': 'tot_shrhldr_eqy_excl_min_int',
            'total_cur_assets': 'total_current_assets',
            'total_liab': 'tot_liab',
            'total_cur_liab': 'total_current_liability',
            'money_cap': 'cash_equivalents',                   # 货币资金
        },
        'cashflow': {
            'n_cashflow_act': 'net_cash_flows_oper_act',
            'c_pay_acq_const_fiolta': 'cash_pay_acq_const_fiolta',
        },
    }

    _SOURCE_TABLE = {
        internal_name: table_name
        for table_name, column_map in _TUSHARE_COLUMN_MAP.items()
        for internal_name in column_map.values()
    }

    # 各表需要读取的键列
    _KEY_COLS = [
        'ts_code', 'end_date', 'ann_date', 'f_ann_date',
        'report_type', 'update_flag',
    ]

    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        market_data_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        初始化财务数据加载器
        """
        # 路径设置
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent  # 因子库
        project_root = factor_lib_root.parent  # 截面多因子模型

        if raw_data_path is None:
            self.raw_data_path = project_root / '01数据' / 'data' / 'tushare_data' / 'financial_full'
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

        # 交易日历（从 market_data/close.parquet 读取）
        self.trading_calendar: List[datetime.date] = []
        self._load_trading_calendar()

        # PIT 对齐器
        self.aligner = PITAligner(self.trading_calendar)

    def _load_trading_calendar(self):
        """
        从 market_data/close.parquet 加载交易日历
        """
        close_file = self.market_data_path / 'close.parquet'
        if not close_file.exists():
            raise FileNotFoundError(
                f"收盘价数据不存在: {close_file}\n"
                f"请先运行 main_prepare_market_data.py 准备行情数据"
            )

        table = pq.read_table(close_file, columns=['time'])
        # 转换为 datetime 对象，确保与 market_data 格式一致
        time_list = table.column('time').to_pylist()
        # 如果是 datetime.date，转换为 datetime.datetime
        self.trading_calendar = []
        for t in time_list:
            if isinstance(t, datetime.date) and not isinstance(t, datetime.datetime):
                self.trading_calendar.append(datetime.datetime.combine(t, datetime.time.min))
            else:
                self.trading_calendar.append(t)
        print(f"已加载交易日历: {len(self.trading_calendar)} 个交易日")
        print(f"  日期范围: {self.trading_calendar[0]} ~ {self.trading_calendar[-1]}")

    def _read_partition_table(self, table_name: str) -> pd.DataFrame:
        """
        读取一张 Tushare 季度分区表（全部分区），保留 PIT 版本事件并重命名列

        同一报告期可以保留多个不同 f_ann_date。仅当同一股票、报告期和
        公告日存在并列版本时，保守选择 type 5 调整前数据；没有 type 5
        时优先 update_flag=0 的初始记录。

        参数：
        -----
        table_name : str
            'income' / 'balancesheet' / 'cashflow'

        返回：
        ------
        pd.DataFrame : [ts_code, report_date, m_anntime, 内部字段名...]
        """
        table_dir = self.raw_data_path / table_name
        if not table_dir.exists():
            raise FileNotFoundError(f"分区表目录不存在: {table_dir}")

        col_map = self._TUSHARE_COLUMN_MAP[table_name]
        value_cols = list(col_map.keys())
        read_cols = self._KEY_COLS + value_cols

        files = sorted(table_dir.glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f"分区表为空: {table_dir}")

        df = pd.concat(
            (pd.read_parquet(f, columns=read_cols) for f in files),
            ignore_index=True
        )
        print(f"  {table_name}: {len(files)} 个分区, {len(df)} 行（版本标准化前）")

        df = self._select_statement_versions(df)
        print(f"  {table_name}: PIT 版本事件 {len(df)} 行")

        # 统一命名；严格使用该表自身的实际公告日期，不借用 ann_date 或其他表日期。
        df = df.rename(columns={'end_date': 'report_date', **col_map})
        df['m_anntime'] = df['f_ann_date']
        keep_cols = [
            'ts_code', 'report_date', 'm_anntime',
            'ann_date', 'report_type', 'update_flag',
            *col_map.values(),
        ]
        return df[keep_cols]

    @staticmethod
    def _select_statement_versions(df: pd.DataFrame) -> pd.DataFrame:
        """Preserve one conservative PIT event per report period and announcement."""
        required = {
            'ts_code', 'end_date', 'f_ann_date',
            'report_type', 'update_flag',
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"财务版本选择缺少必要列: {missing}")

        work = df.copy()
        work['_report_type'] = (
            work['report_type']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
        )
        work = work.loc[work['_report_type'].isin(['1', '5'])].copy()
        work['_ann'] = pd.to_datetime(
            work['f_ann_date'].astype('string'),
            format='%Y%m%d',
            errors='coerce',
        )
        missing_f_ann = int(work['_ann'].isna().sum())
        if missing_f_ann:
            raise ValueError(
                f"发现 {missing_f_ann} 条 type 1/type 5 财务记录缺少有效 f_ann_date"
            )

        first_type1 = (
            work.loc[work['_report_type'].eq('1')]
            .groupby(['ts_code', 'end_date'])['_ann']
            .min()
        )
        period_keys = pd.MultiIndex.from_frame(work[['ts_code', 'end_date']])
        work['_first_type1_ann'] = first_type1.reindex(period_keys).to_numpy()
        late_type5 = (
            work['_report_type'].eq('5')
            & work['_first_type1_ann'].notna()
            & work['_ann'].gt(work['_first_type1_ann'])
        )
        work = work.loc[~late_type5].copy()

        work = work.drop_duplicates().copy()
        work['_type_order'] = work['_report_type'].map({'5': 0, '1': 1})
        work['_update_order'] = pd.to_numeric(
            work['update_flag'], errors='coerce'
        ).fillna(99)
        work = work.sort_values(
            by=[
                'ts_code', 'end_date', '_ann',
                '_type_order', '_update_order',
            ],
            kind='mergesort',
        )
        work = work.drop_duplicates(
            subset=['ts_code', 'end_date', 'f_ann_date'], keep='first'
        )
        work['report_type'] = work['_report_type']
        return work.drop(
            columns=[
                '_report_type', '_ann', '_first_type1_ann',
                '_type_order', '_update_order',
            ],
            errors='ignore',
        )

    def _load_all_records(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        分别读取三张分区表，按股票重组；三表在季度层绝不合并。

        返回：
        ------
        Dict[str, Dict[str, List[Dict]]]
            {股票代码: {表名: 该表财务记录列表}}
        """
        print("\n读取 Tushare 财务分区表...")
        all_records: Dict[str, Dict[str, List[Dict]]] = {}
        for table_name in ['income', 'balancesheet', 'cashflow']:
            frame = self._read_partition_table(table_name)
            for stock_code, group in frame.groupby('ts_code', sort=True):
                all_records.setdefault(stock_code, {})[table_name] = (
                    group.drop(columns=['ts_code'])
                    .sort_values(['report_date', 'm_anntime'], kind='mergesort')
                    .to_dict('records')
                )
            del frame
            gc.collect()

        print(
            f"三表独立读取后共 {len(all_records)} 只股票"
        )

        return all_records

    def _calculate_ttm_from_cumulative(
        self,
        records: List[Dict],
        value_field: str
    ) -> List[Dict]:
        """
        从累计值计算 TTM (Trailing Twelve Months)

        正确算法：
        1. 从累计值计算单季度值（必须存在同年准确的上一报告期）
        2. 只对连续四个季度的单季度值求和得到TTM

        参数：
        -----
        records : List[Dict]
            财务记录列表，每个记录包含 report_date 和累计值
        value_field : str
            需要计算TTM的字段名（如 'net_profit_excl_min_int_inc'）

        返回：
        ------
        List[Dict] : 添加了单季度和TTM字段的记录
            每个记录新增：
            - {value_field}_quarter: 单季度值
            - {value_field}_ttm: TTM值
        """
        if not records:
            return []

        available = {
            str(record.get('report_date', '')): record
            for record in records
            if self._is_standard_report_period(record.get('report_date'))
        }
        result = []
        for report_period in sorted(available):
            record = available[report_period]
            new_record = record.copy()
            new_record[f'{value_field}_quarter'] = self._single_quarter_value(
                available, report_period, value_field
            )
            new_record[f'{value_field}_ttm'] = self._ttm_value(
                available, report_period, value_field
            )
            result.append(new_record)
        return result

    @staticmethod
    def _is_standard_report_period(report_period) -> bool:
        value = str(report_period or '')
        return (
            len(value) == 8
            and value.isdigit()
            and value[4:] in {'0331', '0630', '0930', '1231'}
        )

    @staticmethod
    def _previous_report_period(report_period: str) -> Optional[str]:
        year = int(report_period[:4])
        suffix = report_period[4:]
        previous = {
            '0331': f'{year - 1}1231',
            '0630': f'{year}0331',
            '0930': f'{year}0630',
            '1231': f'{year}0930',
        }
        return previous.get(suffix)

    @classmethod
    def _recent_report_periods(
        cls, latest_period: str, count: int = 4
    ) -> List[str]:
        periods = [latest_period]
        while len(periods) < count:
            previous = cls._previous_report_period(periods[-1])
            if previous is None:
                return []
            periods.append(previous)
        return periods

    @staticmethod
    def _numeric_value(record: Optional[Dict], value_field: str) -> float:
        if not record:
            return np.nan
        value = record.get(value_field)
        if value is None or pd.isna(value):
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @classmethod
    def _single_quarter_value(
        cls,
        available: Dict[str, Dict],
        report_period: str,
        value_field: str,
    ) -> float:
        current = cls._numeric_value(available.get(report_period), value_field)
        if pd.isna(current):
            return np.nan
        if report_period.endswith('0331'):
            return current

        previous_period = cls._previous_report_period(report_period)
        previous = cls._numeric_value(
            available.get(previous_period), value_field
        )
        if pd.isna(previous):
            return np.nan
        return current - previous

    @classmethod
    def _ttm_value(
        cls,
        available: Dict[str, Dict],
        latest_period: str,
        value_field: str,
    ) -> float:
        periods = cls._recent_report_periods(latest_period, count=4)
        if len(periods) != 4 or any(p not in available for p in periods):
            return np.nan
        quarter_values = [
            cls._single_quarter_value(available, period, value_field)
            for period in periods
        ]
        if any(pd.isna(value) for value in quarter_values):
            return np.nan
        return float(sum(quarter_values))

    @staticmethod
    def _same_snapshot(left: Optional[Tuple], right: Tuple) -> bool:
        if left is None or len(left) != len(right):
            return False
        for old, new in zip(left, right):
            if pd.isna(old) and pd.isna(new):
                continue
            if old != new:
                return False
        return True

    def _build_latest_period_events(
        self,
        records: List[Dict],
        value_fields: List[str],
    ) -> List[Dict]:
        """生成期末值事件；晚到的旧报告不会替换当前最新报告期。"""
        valid = [
            record for record in records
            if self._is_standard_report_period(record.get('report_date'))
            and record.get('m_anntime') not in (None, '')
        ]
        valid.sort(
            key=lambda record: (
                str(record.get('m_anntime')),
                str(record.get('report_date')),
            )
        )
        available: Dict[str, Dict] = {}
        events: List[Dict] = []
        previous_snapshot: Optional[Tuple] = None
        index = 0
        while index < len(valid):
            event_date = str(valid[index]['m_anntime'])
            while (
                index < len(valid)
                and str(valid[index]['m_anntime']) == event_date
            ):
                record = valid[index]
                available[str(record['report_date'])] = record
                index += 1

            latest_period = max(available)
            latest_record = available[latest_period]
            snapshot = tuple(
                self._numeric_value(latest_record, field)
                for field in value_fields
            )
            if self._same_snapshot(previous_snapshot, snapshot):
                continue
            event = {
                'm_anntime': event_date,
                'report_date': latest_period,
            }
            event.update(dict(zip(value_fields, snapshot)))
            events.append(event)
            previous_snapshot = snapshot
        return events

    def _build_ttm_events(
        self,
        records: List[Dict],
        value_fields: List[str],
    ) -> List[Dict]:
        """
        按实际公告时间维护可用报告集合，并始终针对当时最新报告期计算TTM。

        晚到记录若仍在最近四季度窗口内，可从实际到达日起改变当前TTM；
        已在窗口外的旧记录不会覆盖当前TTM。
        """
        valid = [
            record for record in records
            if self._is_standard_report_period(record.get('report_date'))
            and record.get('m_anntime') not in (None, '')
        ]
        valid.sort(
            key=lambda record: (
                str(record.get('m_anntime')),
                str(record.get('report_date')),
            )
        )
        available: Dict[str, Dict] = {}
        events: List[Dict] = []
        previous_snapshot: Optional[Tuple] = None
        index = 0
        while index < len(valid):
            event_date = str(valid[index]['m_anntime'])
            while (
                index < len(valid)
                and str(valid[index]['m_anntime']) == event_date
            ):
                record = valid[index]
                available[str(record['report_date'])] = record
                index += 1

            latest_period = max(available)
            snapshot = tuple(
                self._ttm_value(available, latest_period, field)
                for field in value_fields
            )
            if self._same_snapshot(previous_snapshot, snapshot):
                continue
            event = {
                'm_anntime': event_date,
                'report_date': latest_period,
            }
            event.update({
                f'{field}_ttm': value
                for field, value in zip(value_fields, snapshot)
            })
            events.append(event)
            previous_snapshot = snapshot
        return events

    def _process_single_stock(
        self,
        stock_code: str,
        records_by_table: Dict[str, List[Dict]],
        field_configs: Optional[List[Tuple[str, bool, str]]] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        处理单个股票的所有字段

        流程：
        1. 对于需要TTM的字段：
           - 先计算季度TTM（累计值→单季度→4季度求和）
           - 再PIT对齐到日度
        2. 对于不需要TTM的字段：直接PIT对齐

        参数：
        -----
        stock_code : str
            股票代码（如 000001.SZ）
        records_by_table : Dict[str, List[Dict]]
            该股票按 income / balancesheet / cashflow 分开的财务记录

        返回：
        ------
        Dict[str, np.ndarray] : {字段名: 日频数值数组}
        """
        if not records_by_table:
            return None

        configs = field_configs if field_configs is not None else self.FIELD_CONFIG
        configs_by_table: Dict[str, List[Tuple[str, bool, str]]] = {}
        for field_name, need_ttm, source_field in configs:
            src_field = source_field if source_field else field_name
            table_name = self._SOURCE_TABLE.get(src_field)
            if table_name is None:
                raise ValueError(f"无法确定字段 {src_field} 所属财务表")
            configs_by_table.setdefault(table_name, []).append(
                (field_name, need_ttm, src_field)
            )

        result: Dict[str, np.ndarray] = {}
        for table_name, table_configs in configs_by_table.items():
            table_records = records_by_table.get(table_name, [])
            ttm_configs = [config for config in table_configs if config[1]]
            direct_configs = [
                config for config in table_configs if not config[1]
            ]

            if ttm_configs:
                source_fields = [config[2] for config in ttm_configs]
                event_fields = [f'{field}_ttm' for field in source_fields]
                events = self._build_ttm_events(
                    table_records, source_fields
                )
                aligned = self.aligner.align(
                    events, 'm_anntime', event_fields, stock_code
                )
                for field_index, config in enumerate(ttm_configs):
                    output_field = f'{config[0]}_ttm'
                    result[output_field] = np.fromiter(
                        (row[field_index + 1] for row in aligned),
                        dtype=np.float64,
                        count=len(aligned),
                    )

            if direct_configs:
                source_fields = [config[2] for config in direct_configs]
                events = self._build_latest_period_events(
                    table_records, source_fields
                )
                aligned = self.aligner.align(
                    events, 'm_anntime', source_fields, stock_code
                )
                for field_index, config in enumerate(direct_configs):
                    output_field = config[0]
                    result[output_field] = np.fromiter(
                        (row[field_index + 1] for row in aligned),
                        dtype=np.float64,
                        count=len(aligned),
                    )

        return result

    def _build_wide_table_from_memmap(
        self,
        field_data: np.memmap,
        stock_codes: List[str],
        field_name: str
    ) -> pa.Table:
        """
        从磁盘映射矩阵构建单个字段的宽表

        参数：
        -----
        field_data : np.memmap
            shape=(股票数, 交易日数) 的磁盘映射矩阵
        stock_codes : List[str]
            股票代码顺序，与 field_data 第一维一致
        field_name : str
            要构建宽表的字段名

        返回：
        ------
        pa.Table : 宽表，列名为 [date, 股票代码1, 股票代码2, ...]
        """
        print(f"\n构建宽表: {field_name}")

        if not stock_codes:
            raise ValueError(f"没有有效数据用于字段 {field_name}")

        print(f"  共 {len(stock_codes)} 个股票有数据")

        arrays = [pa.array(self.trading_calendar, type=pa.timestamp('ns'))]  # 第一列是日期
        names = ['time']

        for stock_idx, stock_code in enumerate(stock_codes):
            # from_pandas=True 保持旧输出语义：NaN 写成 Arrow null。
            arrays.append(pa.array(field_data[stock_idx], type=pa.float64(), from_pandas=True))
            names.append(stock_code)

        return pa.table(arrays, names=names)

    @contextmanager
    def _output_lock(self):
        """阻止多个财务重建进程同时写入同一输出目录。"""
        lock_path = self.output_path / '.financial_prepare.lock'
        lock_file = open(lock_path, 'a+b')
        if lock_path.stat().st_size == 0:
            lock_file.write(b'\0')
            lock_file.flush()

        try:
            if os.name == 'nt':
                import msvcrt
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise RuntimeError(
                f"已有财务数据处理任务正在使用输出目录: {self.output_path}"
            ) from exc

        try:
            yield
        finally:
            if os.name == 'nt':
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def prepare_all_fields(self, fields: Optional[List[str]] = None, overwrite: bool = False):
        """
        批量处理所有财务字段

        参数：
        -----
        fields : List[str], optional
            要处理的字段列表，如 ['cap_stk', 'net_profit_ttm']，默认处理所有
        overwrite : bool
            是否覆盖已存在的文件
        """
        # 确定要处理的字段
        field_configs_to_process = self.FIELD_CONFIG
        if fields is not None:
            # 过滤指定字段
            field_configs_to_process = [
                (fn, need_ttm, src) for fn, need_ttm, src in self.FIELD_CONFIG
                if fn in fields or f"{fn}_ttm" in fields
            ]

        if not field_configs_to_process:
            print("没有匹配的财务字段")
            return []

        output_files = []
        pending_configs = []
        for field_name, need_ttm, source_field in field_configs_to_process:
            output_field = f"{field_name}_ttm" if need_ttm else field_name
            output_file = self.output_path / f"{output_field}.parquet"
            if output_file.exists() and not overwrite:
                print(f"文件已存在，跳过: {output_file}")
                output_files.append(output_file)
            else:
                pending_configs.append((field_name, need_ttm, source_field))

        if not pending_configs:
            print(f"\n全部完成！共找到 {len(output_files)} 个已有文件")
            return output_files

        with self._output_lock():
            # 读取分区表并按股票重组
            all_records = self._load_all_records()
            stock_codes = sorted(all_records.keys())
            num_dates = len(self.trading_calendar)
            output_fields = [
                f"{field_name}_ttm" if need_ttm else field_name
                for field_name, need_ttm, _ in pending_configs
            ]

            estimated_bytes = len(output_fields) * len(stock_codes) * num_dates * 8
            print("\n处理财务数据（TTM + PIT对齐）...")
            print(
                f"  使用磁盘映射缓存约 {estimated_bytes / (1024 ** 3):.2f} GB，"
                "内存不再随处理进度累积"
            )

            temp_dir_obj = tempfile.TemporaryDirectory(
                prefix='.financial_memmap_',
                dir=self.output_path
            )
            memmaps: Dict[str, np.memmap] = {}
            try:
                for output_field in output_fields:
                    memmap_path = Path(temp_dir_obj.name) / f'{output_field}.dat'
                    memmaps[output_field] = np.memmap(
                        memmap_path,
                        dtype=np.float64,
                        mode='w+',
                        shape=(len(stock_codes), num_dates)
                    )

                processed_count = 0
                for stock_idx, stock_code in enumerate(stock_codes):
                    stock_records = all_records.pop(stock_code)
                    result = self._process_single_stock(
                        stock_code,
                        stock_records,
                        pending_configs
                    )
                    if result:
                        for output_field in output_fields:
                            memmaps[output_field][stock_idx, :] = result[output_field]
                        processed_count += 1
                    del stock_records, result

                    if (stock_idx + 1) % 500 == 0:
                        for mmap in memmaps.values():
                            mmap.flush()
                        print(f"  已处理 {stock_idx + 1}/{len(stock_codes)} 个股票...")

                del all_records
                gc.collect()
                print(f"\n成功处理 {processed_count} 个股票")

                for output_field in output_fields:
                    output_file = self.output_path / f"{output_field}.parquet"
                    temp_output_file = self.output_path / f".{output_field}.parquet.tmp"
                    wide_table = None
                    try:
                        memmaps[output_field].flush()
                        wide_table = self._build_wide_table_from_memmap(
                            memmaps[output_field],
                            stock_codes,
                            output_field
                        )
                        pq.write_table(wide_table, temp_output_file)
                        os.replace(temp_output_file, output_file)
                        print(
                            f"已保存: {output_file} "
                            f"({wide_table.num_rows} 行 × {wide_table.num_columns} 列)"
                        )
                        output_files.append(output_file)
                    except Exception as e:
                        if temp_output_file.exists():
                            temp_output_file.unlink()
                        print(f"处理字段 {output_field} 失败: {e}")
                        continue
                    finally:
                        wide_table = None
                        gc.collect()
            finally:
                for mmap in memmaps.values():
                    mmap.flush()
                    mmap._mmap.close()
                memmaps.clear()
                gc.collect()
                temp_dir_obj.cleanup()

        print(f"\n全部完成！共生成或找到 {len(output_files)} 个文件")
        return output_files
