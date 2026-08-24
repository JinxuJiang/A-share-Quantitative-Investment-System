# 01 数据层 (Data Engine) 📥

> 行情与财务数据接入层。当前使用 **Tushare 三件套**；QMT 代码已退役删除（git 历史可找回），旧数据 `raw_data/` 保留作对照。

---

## ✨ 核心能力

| 能力 | 说明 |
|:---|:---|
| 🔌 Tushare数据接入 | `TushareDataEngine` 一个引擎类承载全部逻辑：行情/财务/状态/元数据 |
| 📊 等比前复权 | 原始价 × (adj_factor/最新adj_factor)，避免传统前复权的负数问题 |
| 📅 财务全字段原始层 | 三张报表同时保存 `report_type=1/5`，四表全原生字段按季度分区，PIT 版本选择留给因子层 |
| 🔄 断点续跑 | 行情每250交易日原子落盘分片、财务按表/季度原子写入，已完成自动跳过 |
| 📉 回测基准 | 独立维护中证1000（000852.SH）日行情与完整交易安排 |
| ✔️ 自动验证 | `validate_all` 覆盖行情/财务/状态/元数据四类检查，FAIL 非零退出 |

---

## 📁 目录结构

### 代码文件（本层提交的内容）

```
01数据/
├── Base_TushareEngine.py     # Tushare 核心引擎：连接/抓取/清洗/存储全部逻辑
├── tushare_data_main.py      # 数据入口（--full / --weekly / --refresh-financial-versions）
├── tushare_weekly_update.py  # 每周增量更新（继承引擎，加增量策略）
├── tushare_token.txt         # Tushare token（本地文件，已 gitignore，不提交）
└── README.md                 # 本文档
```

> 旧 QMT 三件套（`Base_DataEngine.py` / `data_main.py` / `monthly_update.py`）
> 已于 2026-07-29 随迁移完成删除，需要时从 git 历史找回。

### 运行时自动生成的目录

首次运行后自动创建，**无需手动创建，也不提交到 Git**：

```
01数据/
└── data/                              # 【运行时自动生成，已 gitignore】
    ├── raw_data/                      # 旧 QMT 数据（保留作对照，QMT 代码已退役）
    │   ├── market_data/               #   行情（个股 parquet）
    │   ├── financial_data/            #   财务（个股 parquet，323列）
    │   ├── st_status.parquet          #   ST状态宽表
    │   ├── stock_info.parquet         #   股票基础信息
    │   ├── industry_map.csv           #   申万一级行业映射
    │   └── update_log.json            #   更新记录
    └── tushare_data/                  # 新 Tushare 数据（当前正式数据源）
        ├── market_data/{code}.parquet #   行情（等比前复权，与旧格式一致）
        ├── benchmark/000852.SH.parquet #  中证1000回测基准（日行情）
        ├── financial_full/{表}/{季度}.parquet # 财务四表全字段季度分区
        ├── st_status.parquet          #   ST状态宽表（0=正常, 1=ST）
        ├── suspend_status.parquet     #   停牌状态宽表（0/1）
        ├── stock_info.parquet         #   股票基础信息
        ├── industry_map.csv           #   申万一级行业映射
        ├── raw/                       #   中间层（断点续跑与复权重建的弹药库）
        │   ├── market/                #     daily / adj_factor / daily_basic（单文件或 *_shards/ 分片）
        │   └── metadata/              #     stock_basic / trade_cal / trade_schedule / stock_st 等
        ├── logs/                      #   验证与对拍报告
        └── update_log.json            #   更新记录
```

---

## 🚀 快速开始

### 0️⃣ 配置 Token

在项目根创建 `01数据/tushare_token.txt`，内容为一行 Tushare Pro token
（该文件已 gitignore，不会提交）。也可用环境变量 `TUSHARE_TOKEN` 覆盖。

### 1️⃣ 首次全量下载

```powershell
conda activate qf

# 首次全量下载（元数据→行情→财务→状态→总验证，断点续跑）
python 01数据/tushare_data_main.py --full

# 全量下载但只到指定日期（避免盘中获取未收盘数据）
python 01数据/tushare_data_main.py --full --end-date 20260727
```

> 直接运行 `python 01数据/tushare_data_main.py` 而不带参数，只会显示帮助，
> 不会下载或更新数据。

只有旧数据目录尚未保存 type 5、需要一次性补抓全部历史版本时，才运行：

```powershell
python 01数据/tushare_data_main.py --refresh-financial-versions
```

正常首次下载和后续周更不需要重复执行这个历史补抓命令。

### 2️⃣ 日常每周更新

下面两个命令是同一套更新流程，**二选一，不要重复运行**：

```powershell
conda activate qf

# 推荐：通过统一入口运行
python 01数据/tushare_data_main.py --weekly

# 等价写法：直接运行周更脚本
python 01数据/tushare_weekly_update.py
```

周更只更新 `01数据/data/tushare_data`，包括元数据、中证1000基准、行情、
财务原始分区、ST/停牌状态和数据层验证；**不会自动重建 `02因子库` 的宽表或计算因子**。

### 3️⃣ 数据更新后刷新因子层

如果希望模型使用刚更新的数据，还要依次执行：

```powershell
conda activate qf

# 1. 将数据层行情转换成因子层市场宽表
python 02因子库/src/data_engine/main_prepare_market_data.py --overwrite

# 2. 重建行业宽表 + 12个财务基础宽表（含新PIT/TTM逻辑）
python 02因子库/src/data_engine/main_prepare_financial_data.py --overwrite

# 3. 重算全部技术因子
python 02因子库/src/alpha_factory/technical/main_compute_technical.py

# 4. 重算全部财务因子
python 02因子库/src/alpha_factory/financial/main_compute_financial.py

# 5. 验收迁移、PIT和因子值
python 02因子库/validate_tushare_factor_migration.py --full-values --pit-samples 30
```

如果本次只更新、修正了财务数据，可以跳过市场宽表和技术因子，只运行：

```powershell
python 02因子库/src/data_engine/main_prepare_financial_data.py --financial-only --overwrite
python 02因子库/src/alpha_factory/financial/main_compute_financial.py
python 02因子库/validate_tushare_factor_migration.py --full-values --pit-samples 30
```

### 4️⃣ 引擎接口（`Base_TushareEngine.py`，代码中直接调用）

| 方法 | 说明 |
|:---|:---|
| `download_metadata()` | 元数据 → `stock_info.parquet` + `industry_map.csv` |
| `download_benchmark_index()` | 中证1000日行情 → `benchmark/000852.SH.parquet` |
| `download_market_data(start, end, missing_only=, build=)` | 行情两步：按日抓取 → 等比前复权构建 `market_data/{code}.parquet` |
| `download_financial_data(start_period, end_period, overwrite=)` | 财务四表全字段；三张报表同时抓 type 1/5 → `financial_full/{表}/{季度}.parquet` |
| `download_status_data(start, end, missing_only=, build=)` | ST/停牌事件表 → `st_status.parquet` + `suspend_status.parquet` |
| `validate_financial_data(...)` | 财务分区校验（schema/键/重复行/披露季完整性） |
| `validate_all(end_date=)` | 总验证：行情覆盖/财务/状态/元数据，报告落盘 `logs/validation_report.json` |

### 5️⃣ 每周更新策略

| 数据类型 | 策略 | 原因 |
|:---|:---|:---|
| 元数据 | 全量重抓 | 股票列表/交易日历会变化，成本低 |
| 中证1000基准 | 全量重抓 | 数据量小，保证回测基准同步更新 |
| 行情 | 缺失补齐 + 全量重建 per-stock | 等比前复权因子随分红除权漂移 |
| 财务 | 重抓最近12个季度的 type 1/5，并与旧分区追加合并 | 周更覆盖最近三年的重述，历史版本不覆盖 |
| 状态 | 缺失补齐 + 重建宽表 | 事件表按日增量 |

---

## 🔄 数据流与逻辑

### 行情数据（两步）

```
输入: 交易日历中的开市日期 (~4022天, 2010至今)
   │
   ▼ 第一步：按交易日抓取（Tushare 按日全市场接口）
┌─────────────────────────────────────────┐
│ daily / adj_factor / daily_basic        │
│  • 每250个交易日原子落盘一个分片          │
│  • 中断最多损失一批，内存占用可控          │
│  • missing_only 模式自动跳过已抓日期      │
└──────────────────┬──────────────────────┘
                   │  存 raw/market/
                   ▼ 第二步：等比前复权 + 构建
┌─────────────────────────────────────────┐
│ qfq_ratio = adj_factor / 最新adj_factor  │
│  • 价格×ratio，成交量×100，成交额×1000    │
│  • 停牌日价格 ffill + suspendFlag=1      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
       market_data/{code}.parquet（与旧 QMT 格式一致）
```

### 财务数据（季度分区）

```
输入: 季度报告期 (2010Q1 至今, 66个季度)
   │
   ▼
┌─────────────────────────────────────────┐
│ 四表全原生字段（不改名、不清洗）           │
│  • income(85列) / balancesheet(153列)   │
│  • cashflow(98列) / fina_indicator(110列)│
│  • 三张报表显式请求 report_type=1 和 5   │
│  • type 1=当前最新；type 5=调整前保留版  │
└──────────────────┬──────────────────────┘
                   │  按 表/季度 版本追加后原子写入
                   ▼
     financial_full/{表}/{季度}.parquet
                   │
                   ▼ 自动校验
     schema一致性 / 必要列 / 重复行 / 披露季完整性
```

> PIT 版本选择**不在本层做**。因子层按每条记录自己的 `f_ann_date` 依次
> 应用版本；同日 type 1/type 5 冲突时保守选择 type 5。每张表只使用自己的
> 实际公告日期，并从公告后的第一个交易日生效。

### 状态数据（ST/停牌）

```
stock_st / suspend_d 事件表（按日抓取，fetch_log 记录已请求日期）
   │
   ▼ 宽表构建（行=全部开市日, 列=全部沪深股票）
st_status.parquet      0=正常, 1=ST（Tushare 不区分 ST/*ST）
suspend_status.parquet 0=正常, 1=停牌
```

> 注意：Tushare `stock_st` 接口数据自 **2016-08-09** 起才有，之前为空属预期。

---

## 🔑 关键设计

### 等比前复权 vs 传统前复权

- **传统前复权**：历史价直接减分红，可能出现负数，收益率计算失真
- **等比前复权**：`adjusted = raw_price × (adj_factor / 最新adj_factor)`，始终为正
- 代价：分红除权后历史复权价会"漂移"，所以周更时用 raw 层**全量重建** per-stock 文件

### 财务为什么在原始层保留 type 1/5

旧 QMT 流程在数据层做 PIT 清洗（同报告期只留最早公告日），会丢掉报告修订版
和调整前版本。新架构在三张报表原始分区中同时保存 `report_type=1/5`，
版本选择逻辑上移到因子层，避免用修订后的最新值回填历史。

### 行情分片原子写

| | 旧方案 | 新方案 |
|:---|:---|:---|
| 写入 | 全部日期攒内存，最后一次写盘 | 每250交易日原子落盘一个分片 |
| 中断 | 全部白跑，且可能写坏文件 | 最多损失一批，重跑同区间安全覆盖 |
| 内存 | 全历史 10GB+ | 与历史规模无关 |

读取时自动合并"旧单文件 + 新分片"并按 `(ts_code, trade_date)` 去重。

---

## 📊 数据规范

### 行情数据

| 属性 | 说明 |
|:---|:---|
| **位置** | `data/tushare_data/market_data/{code}.parquet` |
| **字段** | `time`, `open`, `high`, `low`, `close`, `volume`, `amount`, `preClose`, `suspendFlag` |
| **复权** | 等比前复权（adj_factor 比值） |
| **时间戳** | `time` 为**北京时间午夜的 epoch 毫秒**（按 UTC 解读会差 8 小时） |
| **单位** | 价格=元，volume=股，amount=元 |

### 财务数据

| 属性 | 说明 |
|:---|:---|
| **位置** | `data/tushare_data/financial_full/{income,balancesheet,cashflow,fina_indicator}/{YYYYMMDD}.parquet` |
| **关键字段** | `end_date`=报告期, `f_ann_date`=实际公告日, `report_type`=报表版本, `update_flag`=更新标记 |
| **格式** | 全原生字段（85~153列/表）+ `query_period` |
| **特点** | 三张报表同 `(ts_code, end_date)` 可能同时存在 type 1/5；因子层整行选版，不跨版本补字段 |

### 状态与元数据

| 文件 | 说明 |
|:---|:---|
| `st_status.parquet` | 宽表：行=开市日(已排序), 列=股票, 0=正常 1=ST |
| `suspend_status.parquet` | 宽表：0=正常 1=停牌 |
| `stock_info.parquet` | 股票基础信息（含 list_date/delist_date/list_status） |
| `industry_map.csv` | 申万一级行业映射（SW2021） |

---

## ✅ 验收状态（2026-07-29）

- 数据层 `validate_all`：**12/12 全 PASS**（行情5468只、财务4表×66季度分区、状态/元数据齐全）；
- 三张报表同时保存 `report_type=1/5`，临时文件0、必要列缺失0；
- 因子层迁移验收（`validate_tushare_factor_migration.py`）：**PASS=9 / FAIL=0**；
- 45个因子（21技术+24财务）已全部基于 Tushare 数据重建；
- QMT 三件套代码、迁移临时脚本、调试残留已清理；
- 2026Q2仍处于披露季，后续 `--weekly` 会继续补齐。

---

## 📚 详细文档

- [01.1_设计原理与逻辑架构](../docs/01.1_设计原理与逻辑架构.md) - 架构设计、数据流、核心决策
- [01.2_工程实现与规范](../docs/01.2_工程实现与规范.md) - API说明、数据格式、开发规范
- [01.3_运维与变更日志](../docs/01.3_运维与变更日志.md) - 检查点、性能基准、变更记录

---

*最后更新: 2026-07-29（Tushare 迁移完成，QMT 代码退役，数据层验收通过）*  
*维护者: 蒋大王*
