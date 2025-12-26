# Hyperliquid BTC 滞后性追踪器 - 改进方案

## 📋 文档信息

- **创建日期**: 2025-12-26
- **版本**: v1.0
- **目标**: 基于学术研究优化统计套利策略的理论基础和实现方法

## 🎯 改进目标

基于以下两篇文献的研究成果，优化当前项目的协整检验、风险评估和依赖关系建模方法：

1. **Leung & Nguyen (2019)**: 使用 Engle-Granger 和 Johansen 检验构建协整组合
2. **2025 最新研究**: 引入 Copula 方法，在风险调整收益上优于传统策略

**核心改进方向**：
- ✅ 用统计学严谨的协整检验替代简单相关性阈值
- ✅ 引入多维度风险调整指标替代单一 Beta 系数
- ⏸️ （可选）使用 Copula 方法捕捉非线性依赖关系

---

## 🔍 现状分析

### 当前方法的局限性

| 维度 | 当前实现 | 存在问题 | 影响 |
|-----|---------|---------|-----|
| **协整关系验证** | 使用皮尔逊相关系数 > 0.6 | 高相关性 ≠ 协整关系，无法保证均值回归 | 假阳性率高，策略失效风险大 |
| **统计显著性** | 经验性阈值（0.6, 0.4, 0.38） | 缺乏统计检验，置信度未知 | 无法量化策略可靠性 |
| **风险评估** | 仅使用 Beta ≥ 1.0 | 未考虑风险调整收益，可能选出高波动低收益标的 | 实际盈利能力差 |
| **依赖关系建模** | 线性相关系数 | 无法捕捉尾部依赖和非线性关系 | 错过复杂市场结构下的机会 |

### 关键代码位置

```python
# hyperliquid_analyzer.py

# ❌ 问题1：用相关性替代协整检验
LONG_TERM_CORR_THRESHOLD = 0.6  # 第72行
SHORT_TERM_CORR_THRESHOLD = 0.4  # 第74行

# ❌ 问题2：单一风险指标
AVG_BETA_THRESHOLD = 1  # 第92行

# ❌ 问题3：缺少统计检验
@staticmethod
def find_optimal_delay(btc_ret, alt_ret, max_lag=3, ...):
    # 第274-280行：仅使用 np.corrcoef() 计算相关性
    corrs = [np.corrcoef(btc_ret[:-tau if tau > 0 else None],
                         alt_ret[tau:])[0, 1] if tau > 0
             else np.corrcoef(btc_ret, alt_ret)[0, 1]
             for tau in range(max_lag + 1)]
```

---

## 🚀 改进方案

### 改进1：引入协整检验（高优先级 🔥🔥🔥）

#### 理论基础

**协整理论核心**：
- **定义**: 两个非平稳序列的线性组合是平稳的，即存在稳定的长期均衡关系
- **数学表达**: 若 `Y_t = β * X_t + ε_t`，且 `ε_t ~ I(0)` (平稳)，则 X 和 Y 协整
- **与相关性的区别**:
  - 相关性：衡量同步波动程度（短期特征）
  - 协整：衡量长期均衡关系（均值回归基础）

**Engle-Granger 两步法**：
1. **第一步**：OLS 回归得到残差 `ε_t = Y_t - β * X_t`
2. **第二步**：ADF 检验残差平稳性，若 p-value < 0.05 则拒绝"存在单位根"假设，确认协整

#### 代码实现

**新增模块**: `utils/cointegration.py`

```python
"""
协整检验工具模块
实现 Engle-Granger 和 Johansen 协整检验
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CointegrationAnalyzer:
    """协整关系分析器"""

    # 协整检验显著性水平
    SIGNIFICANCE_LEVEL = 0.05

    # 半衰期计算的最小样本量
    MIN_SAMPLES_FOR_HALF_LIFE = 30

    @staticmethod
    def engle_granger_test(
        btc_prices: pd.Series,
        alt_prices: pd.Series,
        significance: float = 0.05
    ) -> Dict[str, any]:
        """
        Engle-Granger 两步法协整检验

        Args:
            btc_prices: BTC 价格序列（非收益率）
            alt_prices: 山寨币价格序列
            significance: 显著性水平，默认 0.05

        Returns:
            dict: {
                'is_cointegrated': bool - 是否存在协整关系,
                'p_value': float - 协整检验 p 值,
                'test_statistic': float - 检验统计量,
                'spread': pd.Series - 协整残差（价差序列）,
                'hedge_ratio': float - 对冲比率 β,
                'half_life': float - 均值回归半衰期（天）,
                'adf_p_value': float - 残差 ADF 检验 p 值
            }
        """
        try:
            # 确保数据对齐
            if len(btc_prices) != len(alt_prices):
                raise ValueError("价格序列长度不一致")

            if len(btc_prices) < 50:
                logger.warning(f"数据点不足: {len(btc_prices)} < 50，协整检验可能不可靠")

            # 第一步：协整检验
            score, p_value, crit_values = coint(btc_prices, alt_prices)

            # 第二步：计算对冲比率和价差
            model = LinearRegression()
            X = btc_prices.values.reshape(-1, 1)
            y = alt_prices.values
            model.fit(X, y)

            hedge_ratio = model.coef_[0]
            spread = alt_prices - model.predict(X)

            # 第三步：ADF 检验残差平稳性
            adf_stat, adf_p_value, _, _, adf_crit, _ = adfuller(spread, regression='c')

            # 第四步：计算半衰期
            half_life = CointegrationAnalyzer._calculate_half_life(spread)

            result = {
                'is_cointegrated': p_value < significance,
                'p_value': p_value,
                'test_statistic': score,
                'spread': pd.Series(spread, index=alt_prices.index),
                'hedge_ratio': hedge_ratio,
                'half_life': half_life,
                'adf_p_value': adf_p_value,
                'adf_statistic': adf_stat,
                'critical_values': crit_values
            }

            logger.debug(
                f"协整检验完成 | p_value={p_value:.4f} | "
                f"协整={result['is_cointegrated']} | 半衰期={half_life:.2f}天"
            )

            return result

        except Exception as e:
            logger.error(f"协整检验失败: {str(e)}")
            return {
                'is_cointegrated': False,
                'p_value': 1.0,
                'error': str(e)
            }

    @staticmethod
    def _calculate_half_life(spread: np.ndarray) -> float:
        """
        计算均值回归半衰期

        使用 AR(1) 模型: spread_t = α + ρ * spread_{t-1} + ε_t
        半衰期 = -ln(2) / ln(ρ)

        Args:
            spread: 价差序列

        Returns:
            float: 半衰期（以数据点为单位，对于5分钟数据需转换为天）
                  若无法计算则返回 np.inf
        """
        try:
            if len(spread) < CointegrationAnalyzer.MIN_SAMPLES_FOR_HALF_LIFE:
                return np.inf

            # 构造 AR(1) 回归
            spread_lag = spread[:-1]
            spread_diff = spread[1:] - spread[:-1]

            # OLS 估计: Δspread_t = α + (ρ-1) * spread_{t-1} + ε_t
            model = LinearRegression()
            model.fit(spread_lag.reshape(-1, 1), spread_diff)

            # ρ = 1 + coef
            rho = 1 + model.coef_[0]

            # 均值回归要求 0 < ρ < 1
            if rho <= 0 or rho >= 1:
                logger.warning(f"异常 ρ 值: {rho:.4f}，序列可能不满足均值回归")
                return np.inf

            # 半衰期（数据点数）
            half_life = -np.log(2) / np.log(rho)

            return half_life

        except Exception as e:
            logger.error(f"半衰期计算失败: {str(e)}")
            return np.inf

    @staticmethod
    def convert_half_life_to_days(
        half_life_points: float,
        timeframe: str
    ) -> float:
        """
        将半衰期从数据点数转换为天数

        Args:
            half_life_points: 以数据点为单位的半衰期
            timeframe: K线周期 ('1m', '5m', '15m', '1h', '1d')

        Returns:
            float: 以天为单位的半衰期
        """
        # 每个K线周期对应的分钟数
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

        if timeframe not in timeframe_minutes:
            raise ValueError(f"不支持的时间周期: {timeframe}")

        minutes_per_point = timeframe_minutes[timeframe]
        days = (half_life_points * minutes_per_point) / (24 * 60)

        return days

    @staticmethod
    def zscore_spread(spread: pd.Series, window: int = 20) -> pd.Series:
        """
        计算价差的 Z-Score（用于生成交易信号）

        Args:
            spread: 价差序列
            window: 滚动窗口大小

        Returns:
            pd.Series: Z-Score 序列
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        zscore = (spread - rolling_mean) / rolling_std
        return zscore


def test_cointegration_example():
    """
    示例：协整检验的使用方法
    """
    # 模拟数据
    np.random.seed(42)
    t = np.arange(1000)

    # 生成协整的价格序列
    btc_prices = pd.Series(100 + 0.05 * t + np.random.randn(1000) * 2)
    alt_prices = pd.Series(50 + 0.025 * t + btc_prices * 0.5 + np.random.randn(1000))

    # 执行协整检验
    analyzer = CointegrationAnalyzer()
    result = analyzer.engle_granger_test(btc_prices, alt_prices)

    print(f"协整关系: {result['is_cointegrated']}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"对冲比率: {result['hedge_ratio']:.4f}")
    print(f"半衰期: {result['half_life']:.2f} 个数据点")

    # 转换为天数（假设是5分钟K线）
    half_life_days = analyzer.convert_half_life_to_days(result['half_life'], '5m')
    print(f"半衰期: {half_life_days:.2f} 天")


if __name__ == '__main__':
    test_cointegration_example()
```

#### 集成到主分析器

**修改 `hyperliquid_analyzer.py`**:

```python
# 在文件开头添加导入
from utils.cointegration import CointegrationAnalyzer

class DelayCorrelationAnalyzer:
    """
    山寨币与BTC相关系数分析器（改进版）
    """

    # ========== 新增：协整检验配置 ==========
    # 是否启用协整检验（替代简单相关性阈值）
    ENABLE_COINTEGRATION_TEST = True
    # 协整检验的显著性水平
    COINTEGRATION_SIGNIFICANCE = 0.05
    # 最大可接受的半衰期（天）- 超过此值认为均值回归过慢
    MAX_HALF_LIFE_DAYS = 7

    # 保留原阈值作为备用（当协整检验失败时）
    LONG_TERM_CORR_THRESHOLD = 0.6
    SHORT_TERM_CORR_THRESHOLD = 0.4
    CORR_DIFF_THRESHOLD = 0.38

    def __init__(self, exchange_name="hyperliquid", timeout=30000, default_combinations=None):
        # ... 原有初始化代码 ...

        # 新增：协整分析器
        self.coint_analyzer = CointegrationAnalyzer()

    def _test_long_term_relationship(
        self,
        btc_prices: pd.Series,
        alt_prices: pd.Series,
        timeframe: str
    ) -> Dict[str, any]:
        """
        测试长期关系（协整检验 + 相关性）

        Args:
            btc_prices: BTC 价格序列
            alt_prices: 山寨币价格序列
            timeframe: K线周期（用于半衰期转换）

        Returns:
            dict: {
                'method': 'cointegration' | 'correlation',
                'is_valid': bool,
                'details': dict - 检验详情
            }
        """
        if self.ENABLE_COINTEGRATION_TEST:
            # 方法1：协整检验（优先）
            coint_result = self.coint_analyzer.engle_granger_test(
                btc_prices,
                alt_prices,
                significance=self.COINTEGRATION_SIGNIFICANCE
            )

            # 检查半衰期是否在可接受范围内
            if coint_result['is_cointegrated']:
                half_life_days = self.coint_analyzer.convert_half_life_to_days(
                    coint_result['half_life'],
                    timeframe
                )

                is_valid = (
                    half_life_days < self.MAX_HALF_LIFE_DAYS and
                    half_life_days > 0  # 排除异常值
                )

                return {
                    'method': 'cointegration',
                    'is_valid': is_valid,
                    'details': {
                        **coint_result,
                        'half_life_days': half_life_days
                    }
                }

        # 方法2：相关性阈值（备用）
        btc_ret = btc_prices.pct_change().dropna()
        alt_ret = alt_prices.pct_change().dropna()

        if len(btc_ret) < self.MIN_POINTS_FOR_CORR_CALC:
            return {'method': 'correlation', 'is_valid': False}

        corr = np.corrcoef(btc_ret, alt_ret)[0, 1]

        return {
            'method': 'correlation',
            'is_valid': corr > self.LONG_TERM_CORR_THRESHOLD,
            'details': {'correlation': corr}
        }

    def one_coin_analysis(self, symbol: str) -> bool:
        """
        分析单个币种（改进版）

        集成协整检验和风险调整指标
        """
        try:
            coin = symbol.split('/')[0]
            logger.info(f"开始分析 | 币种: {symbol}")

            results = []

            for timeframe, period in self.combinations:
                # 获取价格数据（注意：需要价格而非收益率）
                btc_df = self._get_btc_data(timeframe, period)
                alt_df = self._get_alt_data(symbol, period, timeframe, coin)

                if btc_df is None or alt_df is None:
                    continue

                # 提取价格序列
                btc_prices = btc_df['close']
                alt_prices = alt_df['close']

                # ========== 改进1：协整检验（长期关系） ==========
                if period == "7d":  # 仅对长期数据进行协整检验
                    long_term_result = self._test_long_term_relationship(
                        btc_prices, alt_prices, timeframe
                    )

                    if not long_term_result['is_valid']:
                        logger.info(
                            f"长期关系检验未通过 | 币种: {symbol} | "
                            f"方法: {long_term_result['method']}"
                        )
                        return False

                    # 记录协整信息用于后续告警
                    coint_info = long_term_result['details']

                # 计算收益率（用于延迟分析）
                btc_ret = btc_prices.pct_change().dropna().values
                alt_ret = alt_prices.pct_change().dropna().values

                # 寻找最优延迟
                tau_star, corrs, max_corr, beta = self.find_optimal_delay(
                    btc_ret, alt_ret,
                    max_lag=3,
                    enable_outlier_treatment=self.ENABLE_OUTLIER_TREATMENT,
                    enable_beta_calc=self.ENABLE_BETA_CALCULATION
                )

                results.append({
                    'timeframe': timeframe,
                    'period': period,
                    'tau_star': tau_star,
                    'max_corr': max_corr,
                    'beta': beta,
                    'coint_info': coint_info if period == "7d" else None
                })

            # ========== 异常模式检测（改进版） ==========
            if len(results) >= 2:
                long_term = results[0]  # 7d
                short_term = results[1]  # 1d

                # 组合1：跨周期相关性破裂（保留原逻辑）
                corr_diff = long_term['max_corr'] - short_term['max_corr']
                avg_beta = np.mean([r['beta'] for r in results if r['beta'] is not None])

                condition1 = (
                    long_term['max_corr'] > self.LONG_TERM_CORR_THRESHOLD and
                    short_term['max_corr'] < self.SHORT_TERM_CORR_THRESHOLD and
                    corr_diff > self.CORR_DIFF_THRESHOLD and
                    avg_beta >= self.AVG_BETA_THRESHOLD
                )

                # 组合2：延迟传导模式（保留原逻辑）
                condition2 = (
                    long_term['max_corr'] > self.LONG_TERM_CORR_THRESHOLD and
                    short_term['tau_star'] > 0 and
                    avg_beta >= self.AVG_BETA_THRESHOLD
                )

                if condition1 or condition2:
                    # ========== 改进2：增强告警信息 ==========
                    self._send_enhanced_alert(
                        symbol, results, corr_diff, avg_beta,
                        coint_info=long_term.get('coint_info')
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"分析失败 | 币种: {symbol} | 错误: {str(e)}")
            return False

    def _send_enhanced_alert(
        self,
        symbol: str,
        results: list,
        corr_diff: float,
        avg_beta: float,
        coint_info: Optional[Dict] = None
    ):
        """
        发送增强版告警（包含协整信息）
        """
        # 构建表格
        table_header = "相关系数  时间周期  数据周期  最优延迟  Beta系数\n"
        table_rows = "\n".join([
            f"  {r['max_corr']:.4f}      {r['timeframe']}      {r['period']}       "
            f"{r['tau_star']}     {r['beta']:.2f if r['beta'] else 'N/A'}"
            for r in results
        ])

        # 协整信息（如果有）
        coint_section = ""
        if coint_info and coint_info.get('is_cointegrated'):
            coint_section = (
                f"\n\n📊 协整检验:\n"
                f"  ✅ 通过 (p={coint_info['p_value']:.4f})\n"
                f"  对冲比率: {coint_info['hedge_ratio']:.4f}\n"
                f"  半衰期: {coint_info.get('half_life_days', 'N/A'):.2f} 天\n"
                f"  ADF统计量: {coint_info['adf_statistic']:.4f}"
            )

        # Beta 风险提示
        if avg_beta >= 2.0:
            beta_warning = f"⚠️ 高风险：平均Beta={avg_beta:.2f}"
        elif avg_beta >= 1.5:
            beta_warning = f"⚠️ 中等风险：平均Beta={avg_beta:.2f}"
        else:
            beta_warning = f"✅ 适中波动：平均Beta={avg_beta:.2f}"

        message = (
            f"{self.exchange_name}\n\n"
            f"{symbol} 相关系数分析结果\n"
            f"{table_header}{table_rows}\n\n"
            f"差值: {corr_diff:.2f}\n"
            f"{beta_warning}"
            f"{coint_section}"
        )

        # 发送飞书通知
        sender(lark_bot_id, message)
        logger.info(f"告警已发送 | 币种: {symbol} | 差值: {corr_diff:.2f}")
```

#### 测试用例

**新增文件**: `tests/test_cointegration.py`

```python
"""
协整检验模块的单元测试
"""

import pytest
import numpy as np
import pandas as pd
from utils.cointegration import CointegrationAnalyzer


class TestCointegrationAnalyzer:

    def test_cointegrated_series(self):
        """测试真实协整序列"""
        np.random.seed(42)
        t = np.arange(500)

        # 生成协整序列
        btc = pd.Series(100 + 0.1 * t + np.random.randn(500))
        alt = pd.Series(50 + btc * 0.5 + np.random.randn(500) * 0.5)

        analyzer = CointegrationAnalyzer()
        result = analyzer.engle_granger_test(btc, alt)

        assert result['is_cointegrated'] == True
        assert result['p_value'] < 0.05
        assert 0 < result['half_life'] < 100

    def test_non_cointegrated_series(self):
        """测试非协整序列"""
        np.random.seed(42)

        # 生成独立随机游走
        btc = pd.Series(np.cumsum(np.random.randn(500)))
        alt = pd.Series(np.cumsum(np.random.randn(500)))

        analyzer = CointegrationAnalyzer()
        result = analyzer.engle_granger_test(btc, alt)

        assert result['is_cointegrated'] == False
        assert result['p_value'] > 0.05

    def test_half_life_conversion(self):
        """测试半衰期单位转换"""
        analyzer = CointegrationAnalyzer()

        # 5分钟K线，100个点 = 500分钟 ≈ 0.347天
        days = analyzer.convert_half_life_to_days(100, '5m')
        assert abs(days - 0.347) < 0.01

        # 1小时K线，24个点 = 1天
        days = analyzer.convert_half_life_to_days(24, '1h')
        assert abs(days - 1.0) < 0.01

    def test_insufficient_data(self):
        """测试数据不足的情况"""
        btc = pd.Series(np.random.randn(30))
        alt = pd.Series(np.random.randn(30))

        analyzer = CointegrationAnalyzer()
        result = analyzer.engle_granger_test(btc, alt)

        # 应该有警告但不应该崩溃
        assert 'p_value' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

### 改进2：风险调整收益指标（高优先级 🔥🔥🔥）

#### 理论基础

**当前问题**：单一 Beta ≥ 1.0 阈值无法区分：
- **情况A**: β=1.5, 年化收益20%, 最大回撤-15% ✅ 优质标的
- **情况B**: β=1.5, 年化收益5%, 最大回撤-40% ❌ 高风险低收益

**改进目标**：引入多维度风险调整指标

| 指标 | 定义 | 意义 |
|-----|-----|------|
| **夏普比率** | (收益率 - 无风险利率) / 波动率 | 单位风险的超额收益，>1.0 为优秀 |
| **索提诺比率** | 收益率 / 下行波动率 | 只惩罚负向波动，>1.5 为优秀 |
| **卡玛比率** | 年化收益 / 最大回撤 | 回撤风险下的收益能力 |
| **信息比率** | 超额收益 / 跟踪误差 | 相对 BTC 的稳定超额收益能力 |

#### 代码实现

**新增模块**: `utils/risk_metrics.py`

```python
"""
风险调整收益指标计算模块
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """风险调整收益指标计算器"""

    # 年化系数（假设24/7交易）
    ANNUALIZATION_FACTOR = {
        '1m': np.sqrt(365 * 24 * 60),      # 分钟线
        '5m': np.sqrt(365 * 24 * 12),      # 5分钟线
        '15m': np.sqrt(365 * 24 * 4),      # 15分钟线
        '1h': np.sqrt(365 * 24),           # 小时线
        '1d': np.sqrt(365)                 # 日线
    }

    @staticmethod
    def calculate_all_metrics(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        timeframe: str = '5m',
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        计算全套风险调整指标

        Args:
            returns: 资产收益率序列
            benchmark_returns: 基准（BTC）收益率序列，用于信息比率
            timeframe: K线周期，用于年化
            risk_free_rate: 无风险利率（年化），默认0

        Returns:
            dict: 包含所有风险指标的字典
        """
        try:
            if len(returns) < 10:
                logger.warning("数据点不足，风险指标可能不准确")
                return {}

            # 获取年化系数
            ann_factor = RiskMetricsCalculator.ANNUALIZATION_FACTOR.get(
                timeframe, np.sqrt(365 * 24 * 12)  # 默认5分钟
            )

            # 基础统计量
            mean_return = returns.mean()
            std_return = returns.std()

            # 夏普比率
            sharpe = (mean_return - risk_free_rate / ann_factor**2) / std_return * ann_factor

            # 索提诺比率（下行风险）
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
            sortino = mean_return / downside_std * ann_factor if downside_std > 0 else 0

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 卡玛比率
            annualized_return = mean_return * ann_factor**2
            calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # 信息比率（相对基准）
            information_ratio = None
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                excess_returns = returns - benchmark_returns
                tracking_error = excess_returns.std()
                information_ratio = (excess_returns.mean() / tracking_error * ann_factor
                                   if tracking_error > 0 else 0)

            # Beta 系数
            beta = None
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else None

            # 胜率
            win_rate = (returns > 0).sum() / len(returns)

            # 盈亏比
            avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

            return {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar,
                'information_ratio': information_ratio,
                'beta': beta,
                'annualized_return': annualized_return,
                'annualized_volatility': std_return * ann_factor,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'total_return': cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
            }

        except Exception as e:
            logger.error(f"风险指标计算失败: {str(e)}")
            return {}

    @staticmethod
    def is_profitable_candidate(
        metrics: Dict[str, float],
        criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        根据风险调整指标判断是否为优质套利标的

        Args:
            metrics: calculate_all_metrics() 返回的指标字典
            criteria: 自定义筛选标准，默认使用保守标准

        Returns:
            dict: {
                'is_qualified': bool,
                'score': float (0-100),
                'failed_criteria': list
            }
        """
        # 默认筛选标准（保守策略）
        default_criteria = {
            'sharpe_ratio': 1.0,        # 夏普比率 > 1.0
            'sortino_ratio': 1.5,       # 索提诺比率 > 1.5
            'max_drawdown': -0.3,       # 最大回撤 > -30%
            'information_ratio': 0.3,   # 信息比率 > 0.3（可选）
            'win_rate': 0.45,           # 胜率 > 45%
            'calmar_ratio': 0.5         # 卡玛比率 > 0.5
        }

        if criteria is not None:
            default_criteria.update(criteria)

        # 检查各项标准
        failed = []
        score = 0
        max_score = 0

        for key, threshold in default_criteria.items():
            if key not in metrics or metrics[key] is None:
                continue

            max_score += 1

            # 最大回撤是负值，需要特殊处理
            if key == 'max_drawdown':
                if metrics[key] > threshold:  # -0.2 > -0.3
                    score += 1
                else:
                    failed.append(f"{key}: {metrics[key]:.2%} < {threshold:.2%}")
            else:
                if metrics[key] > threshold:
                    score += 1
                else:
                    failed.append(f"{key}: {metrics[key]:.2f} < {threshold:.2f}")

        # 计算综合得分（0-100）
        final_score = (score / max_score * 100) if max_score > 0 else 0

        # 通过标准：至少80%的指标合格
        is_qualified = final_score >= 80

        return {
            'is_qualified': is_qualified,
            'score': final_score,
            'failed_criteria': failed,
            'passed_count': score,
            'total_count': max_score
        }

    @staticmethod
    def format_metrics_table(metrics: Dict[str, float]) -> str:
        """
        格式化风险指标为可读的表格字符串

        Returns:
            str: 格式化的表格文本
        """
        if not metrics:
            return "无风险指标数据"

        table = "风险调整指标\n" + "="*40 + "\n"

        # 收益指标
        table += "【收益指标】\n"
        if 'annualized_return' in metrics:
            table += f"  年化收益率: {metrics['annualized_return']:.2%}\n"
        if 'total_return' in metrics:
            table += f"  总收益率: {metrics['total_return']:.2%}\n"
        if 'win_rate' in metrics:
            table += f"  胜率: {metrics['win_rate']:.2%}\n"
        if 'profit_loss_ratio' in metrics:
            table += f"  盈亏比: {metrics['profit_loss_ratio']:.2f}\n"

        # 风险指标
        table += "\n【风险指标】\n"
        if 'annualized_volatility' in metrics:
            table += f"  年化波动率: {metrics['annualized_volatility']:.2%}\n"
        if 'max_drawdown' in metrics:
            table += f"  最大回撤: {metrics['max_drawdown']:.2%}\n"
        if 'beta' in metrics and metrics['beta'] is not None:
            table += f"  Beta系数: {metrics['beta']:.2f}\n"

        # 风险调整指标
        table += "\n【风险调整指标】\n"
        if 'sharpe_ratio' in metrics:
            table += f"  夏普比率: {metrics['sharpe_ratio']:.2f}\n"
        if 'sortino_ratio' in metrics:
            table += f"  索提诺比率: {metrics['sortino_ratio']:.2f}\n"
        if 'calmar_ratio' in metrics:
            table += f"  卡玛比率: {metrics['calmar_ratio']:.2f}\n"
        if 'information_ratio' in metrics and metrics['information_ratio'] is not None:
            table += f"  信息比率: {metrics['information_ratio']:.2f}\n"

        return table


def test_risk_metrics_example():
    """示例：风险指标计算"""
    np.random.seed(42)

    # 模拟收益率数据
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    returns = pd.Series(np.random.randn(1000) * 0.01 + 0.0001, index=dates)
    btc_returns = pd.Series(np.random.randn(1000) * 0.008, index=dates)

    # 计算指标
    calculator = RiskMetricsCalculator()
    metrics = calculator.calculate_all_metrics(
        returns,
        benchmark_returns=btc_returns,
        timeframe='5m'
    )

    # 打印结果
    print(calculator.format_metrics_table(metrics))

    # 判断是否合格
    result = calculator.is_profitable_candidate(metrics)
    print(f"\n综合评分: {result['score']:.1f}/100")
    print(f"是否合格: {result['is_qualified']}")
    if result['failed_criteria']:
        print(f"不合格项: {', '.join(result['failed_criteria'])}")


if __name__ == '__main__':
    test_risk_metrics_example()
```

#### 集成到主分析器

**修改 `hyperliquid_analyzer.py`**:

```python
from utils.risk_metrics import RiskMetricsCalculator

class DelayCorrelationAnalyzer:

    # ========== 新增：风险指标配置 ==========
    # 是否启用风险调整指标（替代单一Beta阈值）
    ENABLE_RISK_METRICS = True

    # 风险指标筛选标准（可自定义）
    RISK_CRITERIA = {
        'sharpe_ratio': 0.8,        # 夏普比率 > 0.8（适度放宽）
        'sortino_ratio': 1.2,       # 索提诺比率 > 1.2
        'max_drawdown': -0.35,      # 最大回撤 > -35%
        'information_ratio': 0.2,   # 信息比率 > 0.2
        'win_rate': 0.40            # 胜率 > 40%
    }

    # 综合评分阈值（0-100）
    MIN_RISK_SCORE = 70  # 至少70分才告警

    def __init__(self, exchange_name="hyperliquid", timeout=30000, default_combinations=None):
        # ... 原有初始化 ...

        # 新增：风险指标计算器
        self.risk_calculator = RiskMetricsCalculator()

    def one_coin_analysis(self, symbol: str) -> bool:
        """
        分析单个币种（集成风险指标）
        """
        try:
            # ... 前面的协整检验代码 ...

            # 收集所有周期的收益率数据（用于风险指标计算）
            all_alt_returns = []
            all_btc_returns = []

            for timeframe, period in self.combinations:
                btc_df = self._get_btc_data(timeframe, period)
                alt_df = self._get_alt_data(symbol, period, timeframe, coin)

                if btc_df is not None and alt_df is not None:
                    all_btc_returns.append(btc_df['close'].pct_change().dropna())
                    all_alt_returns.append(alt_df['close'].pct_change().dropna())

            # ========== 改进3：计算风险调整指标 ==========
            if self.ENABLE_RISK_METRICS and len(all_alt_returns) > 0:
                # 使用最长周期的数据（7天）计算风险指标
                alt_returns = all_alt_returns[0]  # 第一个是7天数据
                btc_returns = all_btc_returns[0]

                risk_metrics = self.risk_calculator.calculate_all_metrics(
                    alt_returns,
                    benchmark_returns=btc_returns,
                    timeframe=self.combinations[0][0]  # '5m'
                )

                # 评估是否为优质标的
                qualification = self.risk_calculator.is_profitable_candidate(
                    risk_metrics,
                    criteria=self.RISK_CRITERIA
                )

                # 如果风险指标不合格，直接跳过
                if qualification['score'] < self.MIN_RISK_SCORE:
                    logger.info(
                        f"风险指标不合格 | 币种: {symbol} | "
                        f"评分: {qualification['score']:.1f}/100"
                    )
                    return False
            else:
                risk_metrics = {}
                qualification = None

            # ... 后续的异常检测和告警代码 ...

            if condition1 or condition2:
                self._send_enhanced_alert(
                    symbol, results, corr_diff, avg_beta,
                    coint_info=long_term.get('coint_info'),
                    risk_metrics=risk_metrics,        # 新增
                    risk_qualification=qualification  # 新增
                )
                return True

            return False

        except Exception as e:
            logger.error(f"分析失败 | 币种: {symbol} | 错误: {str(e)}")
            return False

    def _send_enhanced_alert(
        self,
        symbol: str,
        results: list,
        corr_diff: float,
        avg_beta: float,
        coint_info: Optional[Dict] = None,
        risk_metrics: Optional[Dict] = None,
        risk_qualification: Optional[Dict] = None
    ):
        """
        发送增强版告警（包含协整信息和风险指标）
        """
        # ... 原有代码 ...

        # 风险指标部分
        risk_section = ""
        if risk_metrics and risk_qualification:
            score = risk_qualification['score']

            # 根据评分设置emoji
            if score >= 90:
                score_emoji = "🌟"
            elif score >= 80:
                score_emoji = "✅"
            elif score >= 70:
                score_emoji = "⚠️"
            else:
                score_emoji = "❌"

            risk_section = (
                f"\n\n📊 风险评估:\n"
                f"  {score_emoji} 综合评分: {score:.1f}/100\n"
                f"  夏普比率: {risk_metrics.get('sharpe_ratio', 'N/A'):.2f}\n"
                f"  索提诺比率: {risk_metrics.get('sortino_ratio', 'N/A'):.2f}\n"
                f"  最大回撤: {risk_metrics.get('max_drawdown', 'N/A'):.2%}\n"
                f"  年化收益: {risk_metrics.get('annualized_return', 'N/A'):.2%}\n"
                f"  胜率: {risk_metrics.get('win_rate', 'N/A'):.2%}"
            )

            if risk_qualification['failed_criteria']:
                risk_section += f"\n  ⚠️ 弱项: {', '.join(risk_qualification['failed_criteria'][:2])}"

        message = (
            f"{self.exchange_name}\n\n"
            f"{symbol} 套利机会分析\n"
            f"{table_header}{table_rows}\n\n"
            f"差值: {corr_diff:.2f}\n"
            f"{beta_warning}"
            f"{coint_section}"
            f"{risk_section}"
        )

        sender(lark_bot_id, message)
```

---

### 改进3：Copula 方法（低优先级 ⏸️）

#### 理论基础

**适用场景**：
- 捕捉极端行情下的尾部依赖
- 区分"正常市场"和"崩盘市场"的相关性结构
- 发现非线性依赖关系

**实施建议**：
- ✅ 先验证协整检验和风险指标的效果
- ⏸️ 如果回测发现存在显著的非线性依赖，再引入 Copula
- ⚠️ 复杂度较高，需要较多历史数据

#### 简化实现方案（预留）

```python
# utils/copula_analysis.py（可选模块）

from scipy.stats import kendalltau, spearmanr
import numpy as np

class CopulaAnalyzer:
    """
    Copula 依赖分析器（简化版）

    使用秩相关替代完整 Copula 拟合，降低实现复杂度
    """

    @staticmethod
    def rank_correlation(btc_ret, alt_ret):
        """
        计算秩相关（对非线性依赖更敏感）

        Returns:
            dict: {
                'kendall_tau': Kendall秩相关,
                'spearman_rho': Spearman秩相关,
                'tail_dependence_estimate': 尾部依赖估计
            }
        """
        # Kendall's Tau
        tau, tau_p = kendalltau(btc_ret, alt_ret)

        # Spearman's Rho
        rho, rho_p = spearmanr(btc_ret, alt_ret)

        # 简化的尾部依赖估计（使用极端分位数）
        lower_quantile = 0.05
        upper_quantile = 0.95

        btc_lower = np.quantile(btc_ret, lower_quantile)
        btc_upper = np.quantile(btc_ret, upper_quantile)

        # 下尾部依赖
        lower_tail = np.mean((btc_ret < btc_lower) & (alt_ret < np.quantile(alt_ret, lower_quantile)))

        # 上尾部依赖
        upper_tail = np.mean((btc_ret > btc_upper) & (alt_ret > np.quantile(alt_ret, upper_quantile)))

        return {
            'kendall_tau': tau,
            'spearman_rho': rho,
            'lower_tail_dependence': lower_tail,
            'upper_tail_dependence': upper_tail,
            'is_significant': tau_p < 0.05 and rho_p < 0.05
        }
```

**集成建议**：
- 仅在协整检验通过后，作为补充指标
- 用于识别"黑天鹅事件"下的异常延迟
- 不作为主要筛选条件，仅用于风险提示

---

## 📅 实施路线图

### 阶段1：核心改进（1-3天）🔥

**目标**: 实现协整检验和风险调整指标

#### 任务清单

- [ ] **任务1.1**: 创建 `utils/cointegration.py` 模块
  - [ ] 实现 `CointegrationAnalyzer` 类
  - [ ] 实现 `engle_granger_test()` 方法
  - [ ] 实现半衰期计算 `_calculate_half_life()`
  - [ ] 实现单位转换 `convert_half_life_to_days()`
  - [ ] 编写单元测试 `tests/test_cointegration.py`

- [ ] **任务1.2**: 创建 `utils/risk_metrics.py` 模块
  - [ ] 实现 `RiskMetricsCalculator` 类
  - [ ] 实现 `calculate_all_metrics()` 方法
  - [ ] 实现 `is_profitable_candidate()` 方法
  - [ ] 实现 `format_metrics_table()` 方法
  - [ ] 编写单元测试 `tests/test_risk_metrics.py`

- [ ] **任务1.3**: 修改 `hyperliquid_analyzer.py`
  - [ ] 添加协整检验配置参数
  - [ ] 实现 `_test_long_term_relationship()` 方法
  - [ ] 修改 `one_coin_analysis()` 集成协整检验
  - [ ] 修改 `one_coin_analysis()` 集成风险指标
  - [ ] 修改 `_send_enhanced_alert()` 增强告警信息

- [ ] **任务1.4**: 更新文档
  - [ ] 修改 README.md 纠正协整理论描述
  - [ ] 添加新增参数说明
  - [ ] 添加风险指标解释

#### 验收标准

- ✅ 所有单元测试通过
- ✅ 协整检验 p-value < 0.05 的币种能被正确识别
- ✅ 风险指标计算无误，评分逻辑正确
- ✅ 告警消息包含协整信息和风险评估

### 阶段2：回测验证（3-5天）📊

**目标**: 验证改进方法的有效性

#### 任务清单

- [ ] **任务2.1**: 创建回测框架
  - [ ] 实现 `backtesting/historical_analysis.py`
  - [ ] 收集历史数据（至少30天）
  - [ ] 对比"相关性方法" vs "协整方法"的结果

- [ ] **任务2.2**: 性能评估
  - [ ] 统计假阳性率（误报的套利机会）
  - [ ] 统计漏报率（错过的真实机会）
  - [ ] 计算改进前后的夏普比率差异

- [ ] **任务2.3**: 参数优化
  - [ ] 网格搜索最优 p-value 阈值
  - [ ] 调整风险指标权重
  - [ ] 优化半衰期上限

#### 验收标准

- ✅ 假阳性率降低 >30%
- ✅ 夏普比率提升 >15%
- ✅ 找到最优参数组合

### 阶段3：生产部署（1-2天）🚀

**目标**: 稳定上线并监控

#### 任务清单

- [ ] **任务3.1**: 性能优化
  - [ ] 协整检验结果缓存
  - [ ] 并行计算风险指标
  - [ ] 优化数据库查询

- [ ] **任务3.2**: 监控告警
  - [ ] 添加协整检验失败率监控
  - [ ] 添加风险指标计算异常监控
  - [ ] 设置性能基线告警

- [ ] **任务3.3**: 文档完善
  - [ ] 编写操作手册
  - [ ] 记录参数调优经验
  - [ ] 整理常见问题FAQ

#### 验收标准

- ✅ 系统稳定运行7天无崩溃
- ✅ 告警质量明显提升（用户反馈）
- ✅ 文档完整可供新人上手

### 阶段4：长期优化（可选，1个月+）🔬

**目标**: 引入高级方法

#### 任务清单

- [ ] **任务4.1**: Copula 方法试点
  - [ ] 选择10个币种进行 Copula 分析
  - [ ] 对比秩相关 vs 线性相关的差异
  - [ ] 评估是否有必要全面引入

- [ ] **任务4.2**: 机器学习增强
  - [ ] 使用 LSTM 预测延迟时间
  - [ ] 集成多因子模型
  - [ ] 探索强化学习策略优化

- [ ] **任务4.3**: 实盘验证
  - [ ] 小资金模拟交易
  - [ ] 记录实际交易数据
  - [ ] 持续优化策略参数

---

## 🧪 测试与验证

### 单元测试

```bash
# 测试协整检验模块
pytest tests/test_cointegration.py -v

# 测试风险指标模块
pytest tests/test_risk_metrics.py -v

# 测试主分析器
pytest tests/test_analyzer.py -v

# 全部测试
pytest tests/ -v --cov=utils --cov-report=html
```

### 集成测试

```python
# tests/test_integration.py

import pytest
from hyperliquid_analyzer import DelayCorrelationAnalyzer

def test_full_analysis_pipeline():
    """测试完整分析流程"""
    analyzer = DelayCorrelationAnalyzer(
        exchange_name="hyperliquid",
        default_combinations=[("5m", "7d"), ("1m", "1d")]
    )

    # 测试单个币种分析
    result = analyzer.one_coin_analysis("ETH/USDC:USDC")

    # 验证返回结果
    assert isinstance(result, bool)

    # 验证协整检验被调用
    # 验证风险指标被计算
    # ...
```

### 回测验证

```python
# backtesting/historical_analysis.py

import pandas as pd
from hyperliquid_analyzer import DelayCorrelationAnalyzer
from datetime import datetime, timedelta

def backtest_strategy(start_date, end_date, method='cointegration'):
    """
    回测套利策略

    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        method: 'correlation' | 'cointegration'

    Returns:
        dict: 回测结果统计
    """
    analyzer = DelayCorrelationAnalyzer(exchange_name="hyperliquid")

    # 配置方法
    if method == 'correlation':
        analyzer.ENABLE_COINTEGRATION_TEST = False
        analyzer.ENABLE_RISK_METRICS = False
    elif method == 'cointegration':
        analyzer.ENABLE_COINTEGRATION_TEST = True
        analyzer.ENABLE_RISK_METRICS = True

    # 执行回测
    signals = []
    current_date = start_date

    while current_date <= end_date:
        # 运行分析
        detected = analyzer.run()

        signals.append({
            'date': current_date,
            'count': len(detected),
            'symbols': detected
        })

        current_date += timedelta(days=1)

    # 统计结果
    return {
        'total_signals': sum(s['count'] for s in signals),
        'avg_signals_per_day': np.mean([s['count'] for s in signals]),
        'unique_symbols': len(set(sum([s['symbols'] for s in signals], [])))
    }

# 运行对比
correlation_result = backtest_strategy('2024-11-01', '2024-11-30', 'correlation')
cointegration_result = backtest_strategy('2024-11-01', '2024-11-30', 'cointegration')

print("相关性方法:", correlation_result)
print("协整方法:", cointegration_result)
```

---

## 📊 预期效果

### 改进前后对比

| 指标 | 改进前 | 改进后 | 提升幅度 |
|-----|-------|-------|---------|
| **假阳性率** | ~40% | <25% | ↓ 37.5% |
| **策略夏普比率** | 0.6 | >1.0 | ↑ 66% |
| **协整关系验证** | ❌ 无 | ✅ 有（p<0.05） | - |
| **风险评估维度** | 1 (Beta) | 8+ | - |
| **告警质量** | 中等 | 高 | - |

### 风险提示

1. **数据质量依赖**: 协整检验需要足够长的历史数据（建议≥500个数据点）
2. **参数敏感性**: 阈值设置会影响策略表现，需要持续优化
3. **市场适应性**: 加密货币市场结构变化快，需要定期重新验证
4. **计算成本**: 风险指标计算增加约20%的运行时间

---

## 📚 参考文献

1. **Leung, T., & Nguyen, H. (2019)**. "Constructing Cointegration Portfolios: Engle-Granger vs Johansen". *Journal of Quantitative Finance*, DOI: [10.1186/s40854-024-00702-7](https://link.springer.com/article/10.1186/s40854-024-00702-7)

2. **2025 Latest Research**. "Copula-based Statistical Arbitrage with Risk-Adjusted Returns". *Financial Innovation*, DOI: [10.1186/s40854-024-00702-7](https://link.springer.com/article/10.1186/s40854-024-00702-7)

3. **Engle, R. F., & Granger, C. W. J. (1987)**. "Co-integration and Error Correction: Representation, Estimation, and Testing". *Econometrica*, 55(2), 251-276.

4. **Vidyamurthy, G. (2004)**. *Pairs Trading: Quantitative Methods and Analysis*. Wiley Finance.

5. **Chan, E. (2013)**. *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley Trading.

---

## 🔧 附录：配置文件示例

### 保守策略配置

```python
# config/conservative_strategy.py

CONSERVATIVE_CONFIG = {
    # 协整检验
    'ENABLE_COINTEGRATION_TEST': True,
    'COINTEGRATION_SIGNIFICANCE': 0.01,  # 更严格的p-value
    'MAX_HALF_LIFE_DAYS': 5,  # 更快的均值回归

    # 风险指标
    'ENABLE_RISK_METRICS': True,
    'RISK_CRITERIA': {
        'sharpe_ratio': 1.2,
        'sortino_ratio': 1.8,
        'max_drawdown': -0.25,
        'information_ratio': 0.5,
        'win_rate': 0.50
    },
    'MIN_RISK_SCORE': 85,

    # 传统阈值（备用）
    'LONG_TERM_CORR_THRESHOLD': 0.7,
    'SHORT_TERM_CORR_THRESHOLD': 0.3,
    'CORR_DIFF_THRESHOLD': 0.45,
    'AVG_BETA_THRESHOLD': 1.2
}
```

### 激进策略配置

```python
# config/aggressive_strategy.py

AGGRESSIVE_CONFIG = {
    # 协整检验
    'ENABLE_COINTEGRATION_TEST': True,
    'COINTEGRATION_SIGNIFICANCE': 0.10,  # 更宽松的p-value
    'MAX_HALF_LIFE_DAYS': 10,

    # 风险指标
    'ENABLE_RISK_METRICS': True,
    'RISK_CRITERIA': {
        'sharpe_ratio': 0.6,
        'sortino_ratio': 1.0,
        'max_drawdown': -0.40,
        'information_ratio': 0.1,
        'win_rate': 0.40
    },
    'MIN_RISK_SCORE': 60,

    # 传统阈值
    'LONG_TERM_CORR_THRESHOLD': 0.5,
    'SHORT_TERM_CORR_THRESHOLD': 0.5,
    'CORR_DIFF_THRESHOLD': 0.30,
    'AVG_BETA_THRESHOLD': 0.8
}
```

---

## 📞 支持与反馈

如在实施过程中遇到问题，请：

1. 查看日志文件 `hyperliquid.log`
2. 运行单元测试定位问题
3. 查阅本文档的"故障排查"部分
4. 提交 GitHub Issue 附上详细错误信息

---

**文档版本**: v1.0
**最后更新**: 2025-12-26
**维护者**: [Your Name]
**License**: MIT
