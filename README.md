# Hyperliquid BTC 滞后性追踪器

一个用于分析 Hyperliquid 交易所中山寨币与 BTC 相关性的量化分析工具,通过识别短期低相关但长期高相关的异常币种,发现潜在的时间差套利机会。

## 项目背景与目标

### 理论基础

本项目是 **统计套利 (Statistical Arbitrage)** 在加密货币市场的创新应用，核心思想是通过识别山寨币与 BTC 之间的**时间延迟相关性**来发现套利机会。

**🎯 策略分类**

- **主策略**：时间延迟配对交易 (Time-Lagged Pair Trading)
  - 以 BTC 为基准资产，山寨币为配对资产
  - 引入时间延迟维度 τ，捕捉价格传导的时间差

- **理论支撑**：
  - **协整理论 (Cointegration)**：长期相关性表明两个资产存在协整关系
  - **均值回归 (Mean Reversion)**：短期相关性破裂后预期回归至长期水平

**🔬 技术实现的三个维度**

1. **跨周期分析 (Cross-Timeframe Analysis)**
   - **长期视角**：5分钟K线 × 7天数据 → 识别稳定的跟随关系
   - **短期视角**：1分钟K线 × 1天数据 → 捕捉即时的价格延迟
   - **套利信号**：长期高相关（>0.6）但短期低相关（<0.4）

2. **延迟优化 (Lag Optimization)**
   - 搜索最优延迟 τ* ∈ [0, 3]，使相关系数最大化
   - **τ* > 0** 表明山寨币滞后于 BTC，存在时间差套利窗口
   - 通过延迟时间预判山寨币价格走势

3. **波动性评估 (Volatility Assessment)**
   - 计算 Beta 系数衡量山寨币相对 BTC 的波动幅度
   - **β ≥ 1.0** 保证足够的套利空间（波动幅度大于 BTC）
   - 使用 Winsorization 方法处理极端值，提高分析稳健性

### 核心需求

本项目旨在研究 **Hyperliquid 交易所**上满足以下三个维度特征的山寨币：

1. **跨周期特征**：长期（7天）高相关 + 短期（1天）低相关
   - 长期协整：ρ(7d) > 0.6，表明存在稳定的跟随关系
   - 短期破裂：ρ(1d) < 0.4，表明存在相关性破裂窗口

2. **延迟特征**：价格传导存在时间差
   - 最优延迟 τ* > 0，山寨币滞后于 BTC
   - 可预判性：通过 BTC 价格变动预测山寨币走势

3. **波动特征**：波动幅度大于 BTC
   - Beta 系数 β ≥ 1.0，保证足够的套利空间
   - 稳健性：异常值处理后的 Beta 仍满足阈值

### 项目定位

- **当前阶段**：研究与探索阶段，专注于识别和验证异常币种特征
- **长期目标**：为后期构建**全自动化套利系统**提供理论基础和早期数据支撑
- **核心价值**：通过量化分析发现时间差套利机会，自动化告警潜在目标币种

### 应用场景

- **研究分析**：识别具有套利潜力的币种，建立候选池
- **风险评估**：通过 Beta 系数评估波动风险，过滤高风险标的
- **策略验证**：为后续自动化交易策略提供历史数据和回测依据
- **实时监控**：自动化检测和告警，减少人工筛选成本

## 项目简介

基于原项目 [related_corrcoef_abnormal_alert](https://github.com/zhajingwen/related_corrcoef_abnormal_alert) 改进开发。

本项目通过计算不同时间周期和时间延迟下山寨币与 BTC 的皮尔逊相关系数,自动识别存在时间差套利空间的异常币种。核心原理是寻找**短期滞后但长期跟随 BTC 走势**的币种,这类币种可能存在价格发现延迟,从而产生套利机会。

### 核心功能

**三维分析引擎**
- **跨周期分析**: 对比 5分钟/7天 与 1分钟/1天 两种K线组合，发现相关性差异
- **延迟优化**: 自动搜索最优延迟 τ* ∈ [0, 3]，捕捉价格传导时差
- **波动评估**: 计算 Beta 系数和异常值处理（Winsorization），确保分析稳健性

**自动化运维**
- **实时监控**: 持续跟踪所有 USDC 永续合约，自动识别异常模式
- **智能告警**: 通过飞书机器人推送套利机会，减少人工筛选成本
- **定时调度**: 支持自定义执行时间和周期，适配不同交易策略

## 技术原理

本章节详细介绍如何将上述理论转化为可执行的量化分析算法。

### 相关系数分析

通过计算不同延迟 τ 下的皮尔逊相关系数，量化山寨币与 BTC 的跟随关系：

```
ρ(τ) = corr(BTC_returns[t], ALT_returns[t+τ])
```

**异常模式识别（套利机会判定）**

同时满足以下任一组合条件时，判定存在套利机会：

**组合1：跨周期相关性破裂**
- 长期高相关：7天周期 ρ(0) > 0.6
- 短期低相关：1天周期 ρ(0) < 0.4
- 显著差异：Δρ = ρ_long - ρ_short > 0.38
- 波动充足：平均 β ≥ 1.0

**组合2：延迟传导模式**
- 长期高相关：7天周期 ρ(0) > 0.6
- 存在延迟：1天周期最优延迟 τ* > 0
- 波动充足：平均 β ≥ 1.0

### Beta 系数

Beta 系数用于衡量山寨币收益率相对 BTC 的跟随幅度:

```
β = Cov(BTC_returns, ALT_returns) / Var(BTC_returns)
```

- **β > 1.0**: 山寨币波动幅度大于 BTC (高风险)
- **β = 1.0**: 与 BTC 同步波动
- **β < 1.0**: 波动幅度小于 BTC (相对稳健)

项目设定 Beta 阈值为 1.0,低于此值的币种不会触发告警。

### 异常值处理

采用 Winsorization 方法处理极端收益率:

- 下分位数: 0.1%
- 上分位数: 99.9%
- 将超出范围的值限制在分位数边界内

这种方法可以有效降低极端价格波动对统计分析的影响。

## 项目结构

```
hyperliquid-btc-lag-tracker/
├── hyperliquid_analyzer.py    # 核心分析模块
├── utils/                      # 工具模块
│   ├── config.py              # 环境配置
│   ├── lark_bot.py            # 飞书机器人集成
│   ├── scheduler.py           # 定时调度器
│   ├── redisdb.py             # Redis 数据库工具
│   └── spider_failed_alert.py # 爬虫失败告警
├── pyproject.toml             # 项目依赖配置
├── README.md                  # 项目文档
└── hyperliquid.log            # 运行日志
```

## 快速开始

### 环境要求

- Python >= 3.12
- Redis (可选,用于数据缓存)

### 安装依赖

使用 uv (推荐):
```bash
uv sync
```

或使用 pip:
```bash
pip install -r requirements.txt
```

### 主要依赖

- **ccxt**: 加密货币交易所 API 统一接口
- **numpy**: 数值计算
- **pandas**: 数据分析
- **matplotlib**: 数据可视化
- **pyinform**: 信息论分析 (可选)
- **retry**: 自动重试机制

### 环境变量配置

创建 `.env` 文件或设置环境变量:

```bash
# 飞书机器人 Webhook ID (必需,用于接收告警通知)
export LARKBOT_ID="your-lark-bot-webhook-id"

# 环境标识 (可选,默认为 local)
export ENV="production"  # 生产环境启用定时调度

# Redis 配置 (可选)
export REDIS_HOST="127.0.0.1"
export REDIS_PASSWORD="your-redis-password"
```

### 运行分析

直接运行主程序:

```bash
python hyperliquid_analyzer.py
```

程序将自动:
1. 连接 Hyperliquid 交易所
2. 获取所有 USDC 永续合约交易对
3. 分析每个币种与 BTC 的相关性
4. 检测异常模式并通过飞书推送告警

## 核心模块说明

### DelayCorrelationAnalyzer

主要分析器类,包含以下核心方法:

#### 初始化参数

```python
analyzer = DelayCorrelationAnalyzer(
    exchange_name="hyperliquid",  # 交易所名称
    timeout=30000,                # 请求超时(毫秒)
    default_combinations=[        # K线组合
        ("5m", "7d"),  # 5分钟K线,7天数据
        ("1m", "1d")   # 1分钟K线,1天数据
    ]
)
```

#### 关键阈值配置

这些阈值直接对应理论基础中的三个维度分析：

```python
# ========== 跨周期分析阈值 ==========
LONG_TERM_CORR_THRESHOLD = 0.6   # 协整关系判定：长期相关性下限
SHORT_TERM_CORR_THRESHOLD = 0.4  # 相关性破裂判定：短期相关性上限
CORR_DIFF_THRESHOLD = 0.38       # 套利信号触发：最小相关性差值

# ========== 波动性评估阈值 ==========
AVG_BETA_THRESHOLD = 1.0         # 波动充足性判定：保证套利空间
WINSORIZE_LOWER_PERCENTILE = 0.1    # 异常值下限（0.1%分位数）
WINSORIZE_UPPER_PERCENTILE = 99.9   # 异常值上限（99.9%分位数）

# ========== 延迟优化配置 ==========
# 最大延迟搜索范围在 find_optimal_delay() 方法中配置，默认为 3
```

### 数据下载与缓存

```python
# 下载历史数据 (带自动重试)
df = analyzer.download_ccxt_data(
    symbol="ETH/USDC:USDC",
    period="7d",
    timeframe="5m"
)

# 获取 BTC 数据 (带缓存)
btc_df = analyzer._get_btc_data(timeframe="5m", period="7d")

# 获取山寨币数据 (带缓存)
alt_df = analyzer._get_alt_data(
    symbol="ETH/USDC:USDC",
    period="7d",
    timeframe="5m",
    coin="ETH"
)
```

### 相关性分析

```python
# 寻找最优延迟
tau_star, corrs, max_corr, beta = DelayCorrelationAnalyzer.find_optimal_delay(
    btc_ret=btc_returns,    # BTC 收益率数组
    alt_ret=alt_returns,    # 山寨币收益率数组
    max_lag=3,              # 最大延迟
    enable_outlier_treatment=True,  # 启用异常值处理
    enable_beta_calc=True   # 计算 Beta 系数
)

# 分析单个币种
is_anomaly = analyzer.one_coin_analysis("ETH/USDC:USDC")

# 批量分析所有币种
analyzer.run()
```

## 输出示例

### 控制台日志

```
2025-12-25 18:00:00 - __main__ - INFO - 启动分析器 | 交易所: hyperliquid | K线组合: [('5m', '7d'), ('1m', '1d')]
2025-12-25 18:00:05 - __main__ - INFO - 发现 150 个 USDC 永续合约交易对
2025-12-25 18:05:30 - __main__ - INFO - 发现异常币种 | 交易所: hyperliquid | 币种: DOGE/USDC:USDC | 差值: 0.42
2025-12-25 18:05:30 - __main__ - INFO - 分析进度: 37/150 (24%)
...
2025-12-25 18:30:00 - __main__ - INFO - 分析完成 | 交易所: hyperliquid | 总数: 150 | 异常: 8 | 跳过: 12 | 耗时: 1800.5s | 平均: 12.00s/币种
```

### 飞书告警通知

```
hyperliquid

DOGE/USDC:USDC 相关系数分析结果
相关系数  时间周期  数据周期  最优延迟  Beta系数
  0.8500      5m      7d       0     1.35
  0.4200      1m      1d       2     1.28

差值: 0.43
⚠️ 中等波动：平均Beta=1.32
```

## 定时调度

### 使用方式

在代码中使用 `@scheduled_task` 装饰器:

```python
from utils.scheduler import scheduled_task

# 每天 09:00 执行
@scheduled_task(start_time="09:00")
def daily_analysis():
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()

# 每周二、周四、周六 14:30 执行
@scheduled_task(start_time="14:30", weekdays=[1, 3, 5])
def weekly_analysis():
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()

# 每隔 3600 秒执行一次
@scheduled_task(duration=3600)
def hourly_analysis():
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")
    analyzer.run()
```

### 调度模式

1. **周几的几点执行**: 提供 `start_time` 和 `weekdays` 参数
2. **每天的几点执行**: 只提供 `start_time` 参数
3. **每隔 N 秒执行**: 只提供 `duration` 参数

注意: 在本地环境 (`ENV=local`) 下会直接执行,跳过定时调度。

## 配置说明

### 阈值调优建议

根据市场环境和策略需求，可以调整阈值来平衡**准确性**与**机会捕获率**：

```python
# ========== 保守策略：强调协整关系稳定性 ==========
# 适用场景：追求高质量信号，降低假阳性
LONG_TERM_CORR_THRESHOLD = 0.7    # 更强的协整关系要求
SHORT_TERM_CORR_THRESHOLD = 0.3   # 更明显的相关性破裂
CORR_DIFF_THRESHOLD = 0.45        # 更大的套利窗口
AVG_BETA_THRESHOLD = 1.2          # 更高的波动性要求

# ========== 激进策略：扩大捕获范围 ==========
# 适用场景：挖掘更多潜在机会，容忍更高假阳性
LONG_TERM_CORR_THRESHOLD = 0.5    # 放宽协整关系要求
SHORT_TERM_CORR_THRESHOLD = 0.5   # 接受较高的短期相关性
CORR_DIFF_THRESHOLD = 0.30        # 更小的差值即可触发
AVG_BETA_THRESHOLD = 0.8          # 接受较低波动性
```

**调优原则**：
- **协整强度** (LONG_TERM)：值越高，均值回归假设越可靠，但机会越少
- **破裂程度** (SHORT_TERM)：值越低，相关性破裂越明显，信号越强
- **套利窗口** (CORR_DIFF)：值越大，套利空间越大，但触发频率越低
- **波动要求** (BETA)：值越高，收益潜力越大，但风险也越高

### K线组合选择

可以根据交易频率调整 K线组合:

```python
# 高频交易
combinations = [("1m", "1d"), ("5m", "3d")]

# 中长线交易
combinations = [("15m", "14d"), ("1h", "30d")]

# 超短线交易
combinations = [("1m", "6h"), ("5m", "1d")]
```

## 性能优化

### 缓存机制

- **BTC 数据缓存**: 避免重复下载相同周期的 BTC 数据
- **山寨币数据缓存**: 缓存已下载的山寨币数据

### 限流控制

- **请求间隔**: 1.5秒 (避免触发 Hyperliquid 限流)
- **币种间隔**: 2秒
- **启用速率限制**: `enableRateLimit=True`

### 数据验证

- **最小数据点**: 50 个数据点才进行分析
- **相关系数计算**: 至少需要 10 个数据点
- **Beta 系数计算**: 至少需要 10 个数据点

## 风险提示

1. **回测局限性**: 历史相关性不代表未来表现
2. **市场变化**: 市场结构变化可能导致相关性失效
3. **滑点成本**: 实际交易中需考虑滑点和手续费
4. **流动性风险**: 小市值币种可能存在流动性不足
5. **技术风险**: API 限流、网络延迟等技术问题

## 故障排查

### 常见问题

1. **飞书通知未发送**
   - 检查 `LARKBOT_ID` 环境变量是否正确
   - 验证 Webhook URL 是否有效
   - 查看日志中的错误信息

2. **数据下载失败**
   - 检查网络连接
   - 验证交易所 API 可用性
   - 查看是否触发限流 (增加请求间隔)

3. **分析结果为空**
   - 检查数据量是否足够 (>50 个数据点)
   - 验证交易对是否存在
   - 查看日志中的跳过原因

### 日志分析

日志文件位于 `hyperliquid.log`,采用轮转策略:
- 单文件最大 10MB
- 保留 5 个备份文件
- UTF-8 编码

日志级别说明:
- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息 (进度、统计)
- **WARNING**: 警告信息 (数据不足、跳过币种)
- **ERROR**: 错误信息 (下载失败、计算异常)

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

本项目基于原项目开发,请遵守相关开源许可。

## 致谢

- 原项目: [related_corrcoef_abnormal_alert](https://github.com/zhajingwen/related_corrcoef_abnormal_alert)
- CCXT 库: 提供统一的交易所 API 接口
- Hyperliquid 交易所: 数据来源

## 联系方式

如有问题或建议,请通过以下方式联系:

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**免责声明**: 本工具仅供学习和研究使用,不构成任何投资建议。使用本工具进行交易产生的风险和损失由使用者自行承担。
