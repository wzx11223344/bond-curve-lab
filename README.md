# bond-curve-lab — 中国国债收益率曲线分析实验室

基于 **Nelson-Siegel / Nelson-Siegel-Svensson** 模型的中国国债收益率曲线建模、分析与可视化工具。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 核心功能

- **数据获取** — 通过 `akshare` 自动获取中国国债（CGB）收益率曲线数据，覆盖 3M/6M/1Y/2Y/3Y/5Y/7Y/10Y/20Y/30Y 共 10 个标准期限
- **收益率曲线拟合** — Nelson-Siegel（4 参数）与 Nelson-Siegel-Svensson（6 参数）模型拟合
- **衍生曲线计算** — 瞬时远期利率曲线、贴现因子
- **指标计算** — 期限溢价（10Y-2Y）、斜率、曲率（蝴蝶价差）、实际收益率估计、信用利差
- **交互式可视化** — 基于 Plotly 的专业级交互图表
- **完整 HTML 报告** — 一键生成包含所有图表和分析的综合报告

---

## 快速开始

### 安装

```bash
git clone <repo-url>
cd bond-curve-lab
pip install -r requirements.txt
```

### 命令行使用

```bash
# 最新收益率曲线图
python bond_lab.py curve

# Nelson-Siegel 拟合曲线
python bond_lab.py fit

# 10Y-2Y 期限溢价历史走势
python bond_lab.py history

# 3D 收益率曲面
python bond_lab.py 3d

# 完整 HTML 报告（包含所有图表和分析）
python bond_lab.py report
```

### 运行完整演示

```bash
python examples/demo.py
```

---

## Nelson-Siegel 模型简介

Nelson-Siegel 模型将收益率曲线分解为三个潜在因子：

### 模型公式

$$y(\tau) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} + \beta_2 \cdot \Big(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\Big)$$

### 参数含义

| 参数 | 名称 | 经济含义 |
|------|------|----------|
| $\beta_0$ | 水平因子 (Level) | 长期均衡利率，$\tau \to \infty$ 时的渐进值 |
| $\beta_1$ | 斜率因子 (Slope) | 负值 = 正常向上倾斜曲线；正值 = 倒挂 |
| $\beta_2$ | 曲率因子 (Curvature) | 中期"驼峰"幅度，反映中期利率的相对高低 |
| $\lambda$ | 衰减参数 (Decay) | 控制曲率峰值的出现位置，数值越大峰值越靠长期端 |

### 因子载荷

- **水平载荷**: 恒为 1（对所有期限等权影响）
- **斜率载荷**: $\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}$，从 1 单调衰减到 0
- **曲率载荷**: $\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}$，从 0 开始，在中期达到峰值，再衰减到 0

### 即期利率与远期利率关系

瞬时远期利率曲线可从 NS 参数直接推导：

$$f(\tau) = \beta_0 + \beta_1 \cdot e^{-\tau/\lambda} + \beta_2 \cdot \frac{\tau}{\lambda} \cdot e^{-\tau/\lambda}$$

### Nelson-Siegel-Svensson 扩展

Svensson (1994) 增加了第二个曲率因子，更灵活地拟合长期端的复杂形状：

$$y(\tau) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-\tau/\lambda_1}}{\tau/\lambda_1} + \beta_2 \cdot \Big(\frac{1 - e^{-\tau/\lambda_1}}{\tau/\lambda_1} - e^{-\tau/\lambda_1}\Big) + \beta_3 \cdot \Big(\frac{1 - e^{-\tau/\lambda_2}}{\tau/\lambda_2} - e^{-\tau/\lambda_2}\Big)$$

---

## 项目结构

```
bond-curve-lab/
├── README.md                      # 项目文档
├── requirements.txt               # Python 依赖
├── bond_lab.py                    # CLI 入口
├── bond_curve_lab/
│   ├── __init__.py                # 包初始化，公开 API
│   ├── fetcher.py                 # 数据获取（akshare/中国债券信息网）
│   ├── nelson_siegel.py           # NS/NSS 模型拟合、远期利率、贴现因子
│   ├── indicators.py              # 债券市场指标计算
│   └── viz.py                     # Plotly 交互式图表与 HTML 报告
├── examples/
│   └── demo.py                    # 完整演示脚本
└── output/                        # 输出目录
    └── .gitkeep
```

---

## API 参考

### 数据获取 (`fetcher.py`)

```python
from bond_curve_lab import fetch_spot_yields, fetch_historical_curves

# 获取最新即期收益率
maturities, yields = fetch_spot_yields()

# 获取历史收益率曲线
dates, yields_matrix = fetch_historical_curves(start="2024-01-01", end="2025-12-31")
```

### 模型拟合 (`nelson_siegel.py`)

```python
from bond_curve_lab import fit_nelson_siegel, forward_rate_curve, interpret_params

# 拟合 NS 模型
result = fit_nelson_siegel(maturities, yields)
print(result["formula"])       # 完整公式
print(result["r_squared"])     # R-squared
print(result["params"])        # [beta0, beta1, beta2, lambda]

# 参数解读
interp = interpret_params(result["params"])

# 远期利率曲线
fwd = forward_rate_curve(result["params"], tenors=maturities)
```

### 指标计算 (`indicators.py`)

```python
from bond_curve_lab import term_premium, curvature

# 期限溢价 (10Y - 2Y)
spread = term_premium(yield_10y, yield_2y)  # 单位：百分点

# 蝴蝶价差 (2Y / 5Y / 10Y)
butterfly = curvature(yields_5y=y5, yields_10y=y10, yields_2y=y2)
```

### 可视化 (`viz.py`)

```python
from bond_curve_lab import plot_yield_curve, generate_html_report, plot_3d_surface

# 收益率曲线（含 NS 拟合叠加）
plot_yield_curve(maturities, yields, ns_fitted=fitted_yields)

# 3D 曲面图
plot_3d_surface(dates, maturities, yields_matrix)

# 生成完整 HTML 报告
report_path = generate_html_report(maturities, yields, ns_result, date_str, ...)
```

---

## 数据来源

通过 [akshare](https://github.com/akfamily/akshare) 获取中国债券信息网（ChinaBond）发布的国债收益率数据：

- `akshare.bond_china_close_return` — 国债收益率曲线日频数据
- 覆盖 2019 年至今的完整历史数据
- 自动缓存机制（6 小时 TTL），减少重复网络请求

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 数据获取 | akshare, pandas |
| 模型拟合 | scipy.optimize.curve_fit |
| 可视化 | Plotly (interactive) |
| CLI | argparse + rich |
| 科学计算 | numpy, scipy |

---

## 理论参考文献

- Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves." *Journal of Business*, 60(4), 473-489.
- Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates: Sweden 1992-1994." NBER Working Paper No. 4871.
- Diebold, F.X. and Li, C. (2006). "Forecasting the Term Structure of Government Bond Yields." *Journal of Econometrics*, 130(2), 337-364.

---

## License

MIT License
