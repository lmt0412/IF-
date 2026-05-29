# IF 多因子回测模型

本项目以 IF（沪深300股指期货）为研究标的，构建量价因子、基本面因子和交叉因子，通过因子有效性检验筛选候选因子，并使用多个机器学习模型生成交易信号，最终完成样本外回测和策略报告输出。

项目主入口是 `main.py`。运行后会自动完成：

1. 数据读取与因子构建
2. 因子有效性检验
3. 自动筛选入模因子
4. 多模型训练与预测评估
5. 策略回测
6. PDF 回测报告生成

## 项目结构

```text
.
├── main.py                  # 主流程入口
├── factors.py               # 因子构建
├── factor_analysis.py       # 因子有效性检验与因子筛选
├── model.py                 # 模型训练与预测评估
├── backtest.py              # 策略信号生成与回测
├── optimize.py              # 杠杆与做空参数优化
├── report.py                # PDF 报告生成
├── data/
│   ├── cleaned_data.csv     # 当前主数据源
│   ├── raw_data.csv
│   └── test_data.csv        # 当前未使用，因与主数据日期高度重叠
└── Results/
    ├── factor/              # 因子分析结果
    ├── model/               # 模型预测评估结果
    ├── backtest/            # 回测结果
    ├── optimize_results.csv # 参数优化结果
    └── report.pdf           # 最终策略报告
```

## 数据说明

当前模型使用 `data/cleaned_data.csv`。数据为 IF 日频数据，主要字段包括：

| 字段 | 含义 |
|---|---|
| 日期 | 交易日期 |
| 收盘价 | IF 收盘价 |
| 成交占总市值占比 | 成交活跃度指标 |
| 流通市值换手率 | 现货市场换手指标 |
| pe | 沪深300估值指标 |
| macd | 技术指标 |
| 基差 | 期货价格与现货指数之间的价差 |

`test_data.csv` 当前没有用于主流程。此前检查发现它与 `cleaned_data.csv` 日期高度重叠，作为独立测试集会造成数据泄露。因此当前采用 `cleaned_data.csv` 内部按时间顺序切分：前 70% 用于训练，后 30% 用于样本外测试和回测。

## 因子构建

因子在 `factors.py` 中构建，主要分为三类。

### 量价因子

包括动量、波动率、均线偏离、MACD、基差、RSI、布林带位置、短期反转、价量背离等。

示例：

- `mom_5`、`mom_20`：短期和中期动量
- `vol_20`：20日波动率
- `bias_20`：价格相对20日均线偏离
- `basis_change`：基差日变化
- `rsi_14`：14日 RSI
- `bb_pct`：布林带位置
- `ret_1d_rev`：1日反转因子
- `price_vol_diverge`：价量背离

### 基本面因子

主要围绕 PE 构建：

- `pe_factor`
- `pe_z20`
- `pe_change`
- `pe_change_rate`
- `pe_rank_60`

### 交叉因子

用于捕捉非线性组合关系：

- `pe_x_mom20`：估值与20日动量交叉
- `vol_x_bias`：波动率与均线偏离交叉
- `rsi_x_mom5`：RSI 与5日动量交叉

## 预测目标

`factors.py` 中同时生成三个目标变量：

| 目标 | 含义 |
|---|---|
| `target_1d` | 下一交易日收益率 |
| `target_3d` | 未来3日收益率 |
| `target_5d` | 未来5日收益率 |

当前主流程在 `main.py` 中设置：

```python
TARGET = "target_1d"
```

即当前策略预测的是 IF 次日收益。

## 因子有效性检验

`factor_analysis.py` 负责因子检验，主要输出：

- IC 均值
- IC 标准差
- ICIR
- IC > 0 占比
- t 统计量
- p 值
- 因子 Spearman 相关性矩阵
- 五分组平均收益

自动筛选规则：

```text
|ICIR| >= 0.3
p < 0.05
与已选因子的相关性 < 0.7
```

筛选结果保存到：

```text
Results/factor/selected_features.csv
```

## 模型训练

`model.py` 中包含四类模型：

| 模型 | 作用 |
|---|---|
| Ridge | 线性收益预测模型 |
| Logistic Regression | 涨跌方向分类模型 |
| XGBoost | 非线性树模型 |
| LSTM | 时序神经网络模型 |

模型评估指标包括：

- RMSE
- MAE
- 方向准确率
- IC

结果保存到：

```text
Results/model/model_comparison.csv
```

## 策略回测

`backtest.py` 将模型预测值转换为交易信号，并计算策略净值。

当前信号规则：

```text
预测值 > 噪声门槛                  -> 做多
预测值 < -噪声门槛 × SHORT_MULTIPLIER -> 做空
其余                                -> 空仓
```

其中噪声门槛为：

```text
0.5 × 模型预测值标准差
```

当前 `main.py` 中的策略参数为：

```python
SIGNAL_FILTER = True
SHORT_MULTIPLIER = 3.0
LONG_LEVERAGE = 3.0
SHORT_LEVERAGE = 2.0
```

含义：

- 开启弱信号过滤
- 做空信号需要比做多信号更强
- 做多使用3倍杠杆
- 做空使用2倍杠杆

回测指标包括：

- 累计收益
- 年化收益
- 年化波动
- Sharpe
- 最大回撤
- 胜率

回测结果保存到：

```text
Results/backtest/backtest_performance.csv
Results/backtest/backtest_curve.png
Results/backtest/*_daily_returns.csv
```

## 参数优化

`optimize.py` 用于扫描交易参数组合，寻找更优的回测表现。

当前扫描参数包括：

```text
SHORT_MULTIPLIER: 1.0 到 3.0，每 0.5 递增
LONG_LEVERAGE:   1.0 到 3.0，每 0.5 递增
SHORT_LEVERAGE:  1.0 到 3.0，每 0.5 递增
```

优化结果保存到：

```text
Results/optimize_results.csv
```

## PDF 报告

`report.py` 会读取因子分析、模型评估和回测结果，自动生成 PDF 报告：

```text
Results/report.pdf
```

报告只展示当前回测中 Sharpe 最高的模型，并包含：

- 策略概述
- 选中因子
- 因子有效性结果
- 最优模型表现
- 回测绩效
- 净值曲线
- 策略结论

## 环境依赖

建议使用 Python 3.9+。主要依赖：

```text
pandas
numpy
matplotlib
scipy
scikit-learn
xgboost
torch
fpdf2
```

如果本地没有安装依赖，可以在虚拟环境中安装：

```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost torch fpdf2
```

Mac 上使用 XGBoost 可能需要安装 `libomp`：

```bash
brew install libomp
```

## 运行方式

进入项目目录：

```bash
cd ~/Desktop/IF多因子回测模型
```

激活虚拟环境：

```bash
source .venv/bin/activate
```

运行完整流程：

```bash
python main.py
```

## 主程序参数配置

`main.py` 顶部提供了几个核心参数，可以直接修改后重新运行：

```python
TARGET = "target_1d"
SIGNAL_FILTER = True      # True = 预测强度 < 0.5×std 时空仓，False = 每天都持仓
SHORT_MULTIPLIER = 3.0    # 做空门槛 = 做多门槛 × 此倍数，越大越难做空
LONG_LEVERAGE = 3.0       # 做多时的杠杆倍数，1.0 = 无杠杆
SHORT_LEVERAGE = 2.0      # 做空时的杠杆倍数，1.0 = 无杠杆
```

各参数含义：

| 参数 | 作用 |
|---|---|
| `TARGET` | 预测目标，可选 `target_1d`、`target_3d`、`target_5d` |
| `SIGNAL_FILTER` | 是否过滤弱信号；开启后预测强度不足时空仓 |
| `SHORT_MULTIPLIER` | 做空信号门槛倍数，数值越大越难做空 |
| `LONG_LEVERAGE` | 做多杠杆倍数 |
| `SHORT_LEVERAGE` | 做空杠杆倍数 |

运行参数优化：

```bash
caffeinate -i python optimize.py
```

`caffeinate -i` 用于防止 Mac 在优化过程中休眠。

## 当前策略模型

当前项目输出的策略模型可以概括为：

```text
基于量价、基本面和交叉因子的 IF 日频多因子择时策略。
通过 IC/ICIR 筛选有效因子，使用机器学习模型预测次日收益，
并根据预测强度生成做多、做空或空仓信号。
```

在当前参数配置下，策略偏向做多：

- 做多门槛较低
- 做空门槛更高
- 做多杠杆高于做空杠杆

## 注意事项

1. 当前回测未计入手续费、滑点和融资成本，实际交易收益会低于回测结果。
2. 当前测试集来自 `cleaned_data.csv` 的后30%，属于时间顺序样本外测试。
3. `test_data.csv` 当前未使用，因为与主数据日期高度重叠，会引入数据泄露。
4. 杠杆会同时放大收益和亏损，最大回撤也可能随之放大。
5. 本项目用于策略研究和课程/项目展示，不构成投资建议。
