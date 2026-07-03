"""
Visualization Module — Plotly Charts
=====================================

Professional quant-style interactive charts for China bond yield curve analysis.

Color palette:
- Blue (#1f77b4)  — spot / actual yields
- Orange (#ff7f0e) — forward rates
- Green (#2ca02c)  — fitted / model
- Red (#d62728)    — warnings / recession
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Default template and aesthetics
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Quant-like professional template
_QUANT_TEMPLATE = go.layout.Template()
_QUANT_TEMPLATE.layout = go.Layout(
    font=dict(family="Arial, sans-serif", size=13, color="#2c3e50"),
    title=dict(font=dict(size=20, color="#1a1a2e"), x=0.5),
    xaxis=dict(
        showgrid=True, gridcolor="#e8e8e8", gridwidth=0.5,
        zeroline=False, linecolor="#b0b0b0",
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e8e8e8", gridwidth=0.5,
        zeroline=True, zerolinecolor="#cccccc", zerolinewidth=1,
        linecolor="#b0b0b0",
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=60, r=40, t=60, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)",
    ),
    hovermode="x unified",
)

# Colorway
_QUANT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_QUANT_TEMPLATE.layout.colorway = _QUANT_COLORS

pio.templates["quant"] = _QUANT_TEMPLATE
pio.templates.default = "quant"


def _save(fig: go.Figure, name: str) -> str:
    """Save figure as HTML and return path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return str(path)


# ---------------------------------------------------------------------------
# 1. Yield Curve — actual vs fitted
# ---------------------------------------------------------------------------

def plot_yield_curve(
    maturities: np.ndarray,
    yields: np.ndarray,
    ns_fitted: Optional[np.ndarray] = None,
    nss_fitted: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    date_str: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Plot actual yield curve with optional Nelson-Siegel fitted overlay.

    Parameters
    ----------
    maturities : ndarray
        Tenor in years.
    yields : ndarray
        Observed yields.
    ns_fitted : ndarray, optional
        Nelson-Siegel fitted yields.
    nss_fitted : ndarray, optional
        NSS fitted yields.
    title : str, optional
        Chart title.
    date_str : str, optional
        Date label.
    save : bool
        Save to HTML.

    Returns
    -------
    go.Figure
    """
    title = title or "China Government Bond Yield Curve"
    if date_str:
        title += f" — {date_str}"

    maturities = np.asarray(maturities, dtype=float)

    fig = go.Figure()

    # Actual yields
    fig.add_trace(go.Scatter(
        x=maturities, y=yields,
        mode="markers+lines",
        name="Actual CGB Yields",
        marker=dict(size=10, symbol="circle", color="#1f77b4", line=dict(width=1, color="white")),
        line=dict(width=2, color="#1f77b4", dash="dot"),
        hovertemplate="Maturity: %{x:.2f}Y<br>Yield: %{y:.3f}%<extra></extra>",
    ))

    # NS fitted
    if ns_fitted is not None:
        tau_dense = np.linspace(0.1, 30, 200)
        from bond_curve_lab.nelson_siegel import _ns_yield  # type: ignore

        fig.add_trace(go.Scatter(
            x=maturities, y=ns_fitted,
            mode="lines",
            name="Nelson-Siegel Fit",
            line=dict(width=2.5, color="#2ca02c"),
            hovertemplate="Maturity: %{x:.2f}Y<br>NS Yield: %{y:.3f}%<extra></extra>",
        ))

    # NSS fitted
    if nss_fitted is not None:
        fig.add_trace(go.Scatter(
            x=maturities, y=nss_fitted,
            mode="lines",
            name="NSS Fit",
            line=dict(width=2.5, color="#9467bd", dash="dash"),
            hovertemplate="Maturity: %{x:.2f}Y<br>NSS Yield: %{y:.3f}%<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Maturity (Years)",
        yaxis_title="Yield (%)",
        xaxis=dict(tickvals=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                    ticktext=["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]),
    )

    if save:
        _save(fig, "yield_curve")
    return fig


# ---------------------------------------------------------------------------
# 2. Curve animation over time
# ---------------------------------------------------------------------------

def plot_curve_animation(
    dates_list: List[str],
    yield_curves_list: List[np.ndarray],
    maturities: np.ndarray,
    title: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Slider animation of yield curve over time.

    Parameters
    ----------
    dates_list : list of str
        Dates for each curve.
    yield_curves_list : list of ndarray
        Yield curves corresponding to each date.
    maturities : ndarray
        Common maturity vector.
    title : str, optional
    save : bool
    """
    title = title or "CGB Yield Curve Evolution"

    fig = go.Figure()

    # Add all traces, one perdate
    for i, (date, curve) in enumerate(zip(dates_list, yield_curves_list)):
        visible = (i == len(dates_list) - 1)  # show latest by default
        fig.add_trace(go.Scatter(
            x=maturities, y=curve,
            mode="lines+markers",
            name=date,
            visible=visible,
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate="Maturity: %{x:.2f}Y<br>Yield: %{y:.3f}%<extra></extra>",
        ))

    # Slider
    steps = []
    for i in range(len(dates_list)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"{title} — {dates_list[i]}"},
            ],
            label=dates_list[i],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=len(dates_list) - 1,
        currentvalue={"prefix": "Date: "},
        pad={"t": 50},
        steps=steps,
        len=0.9,
        x=0.05,
    )]

    fig.update_layout(
        title=f"{title} — {dates_list[-1]}",
        xaxis_title="Maturity (Years)",
        yaxis_title="Yield (%)",
        sliders=sliders,
    )

    if save:
        _save(fig, "curve_animation")
    return fig


# ---------------------------------------------------------------------------
# 3. 3D surface
# ---------------------------------------------------------------------------

def plot_3d_surface(
    dates: List[str],
    maturities: np.ndarray,
    yields_matrix: np.ndarray,
    title: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    3D surface: Time x Maturity x Yield.

    Shows the full yield surface over time, revealing shifts, twists,
    and butterfly movements in the curve.
    """
    title = title or "CGB Yield Surface"

    maturities = np.asarray(maturities, dtype=float)
    date_nums = np.arange(len(dates))

    fig = go.Figure(data=[go.Surface(
        z=yields_matrix,
        x=maturities,
        y=dates,
        colorscale="Viridis",
        colorbar=dict(title="Yield (%)", tickformat=".2f"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True)),
        ),
        hovertemplate="Date: %{y}<br>Maturity: %{x:.1f}Y<br>Yield: %{z:.3f}%<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Maturity (Years)",
            yaxis_title="Date",
            zaxis_title="Yield (%)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectratio=dict(x=2, y=1.5, z=1),
        ),
    )

    if save:
        _save(fig, "3d_surface")
    return fig


# ---------------------------------------------------------------------------
# 4. Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    dates: List[str],
    maturities: np.ndarray,
    yields_matrix: np.ndarray,
    title: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Heatmap of yield curve changes over time.

    Rows = dates (newest top), columns = maturities.
    Color intensity = yield level.
    """
    title = title or "CGB Yield Curve Heatmap"

    maturities = np.asarray(maturities, dtype=float)

    fig = go.Figure(data=[go.Heatmap(
        z=yields_matrix[::-1],  # newest first
        x=[f"{m:.1f}Y" for m in maturities],
        y=dates[::-1],
        colorscale="RdYlBu_r",
        colorbar=dict(title="Yield (%)", tickformat=".2f"),
        hovertemplate="Date: %{y}<br>Maturity: %{x}<br>Yield: %{z:.3f}%<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Maturity",
        yaxis_title="Date",
        xaxis=dict(side="bottom"),
    )

    if save:
        _save(fig, "heatmap")
    return fig


# ---------------------------------------------------------------------------
# 5. Term premium history (10Y-2Y)
# ---------------------------------------------------------------------------

def plot_term_premium_history(
    dates: List[str],
    premiums: List[float],
    title: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    10Y-2Y term spread over time with zero line and recession shading.

    The 10Y-2Y spread is one of the most reliable recession indicators.
    """
    title = title or "China 10Y-2Y Term Premium History"
    premiums = np.asarray(premiums)

    fig = go.Figure()

    # Term premium line
    fig.add_trace(go.Scatter(
        x=dates, y=premiums,
        mode="lines",
        name="10Y-2Y Spread",
        line=dict(width=2.5, color="#1f77b4"),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.08)",
        hovertemplate="Date: %{x}<br>10Y-2Y: %{y:.2f} bp<extra></extra>",
    ))

    # Zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color="#d62728",
        annotation_text="Inversion threshold", annotation_position="bottom right",
    )

    # Shade inversion regions
    inversion = premiums < 0
    if inversion.any():
        # Find contiguous inversion blocks
        changes = np.diff(np.concatenate([[False], inversion, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for s, e in zip(starts, ends):
            fig.add_vrect(
                x0=dates[s], x1=dates[min(e - 1, len(dates) - 1)],
                fillcolor="rgba(214,39,40,0.12)", layer="below", line_width=0,
                annotation_text="Inverted", annotation_position="top left",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Spread (bp)",
        yaxis=dict(ticksuffix=" bp"),
    )

    if save:
        _save(fig, "term_premium_history")
    return fig


# ---------------------------------------------------------------------------
# 6. Curve components evolution (beta0, beta1, beta2)
# ---------------------------------------------------------------------------

def plot_curve_components(
    ns_params_over_time: List[List[float]],
    dates: List[str],
    title: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Evolution of Nelson-Siegel parameters: level, slope, curvature.

    Reveals structural shifts in the yield curve over time.
    """
    title = title or "Nelson-Siegel Factor Evolution"

    betas = np.array(ns_params_over_time)  # shape (n_dates, 4)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Level (β₀) — Long-term Rate", "Slope (β₁) — Negative = Upward Slope", "Curvature (β₂) — Hump Magnitude"),
    )

    component_names = ["Level (β₀)", "Slope (β₁)", "Curvature (β₂)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(3):
        vals = betas[:, i]
        mask = ~np.isnan(vals)
        fig.add_trace(go.Scatter(
            x=np.array(dates)[mask], y=vals[mask],
            mode="lines",
            name=component_names[i],
            line=dict(width=2, color=colors[i]),
            hovertemplate="Date: %{x}<br>%{y:.4f}<extra></extra>",
        ), row=i + 1, col=1)

        # Add zero line for slope/curvature
        if i >= 1:
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=i + 1, col=1)

    fig.update_layout(title=title, height=800)

    if save:
        _save(fig, "curve_components")
    return fig


# ---------------------------------------------------------------------------
# 7. Forward vs Spot curve
# ---------------------------------------------------------------------------

def plot_forward_vs_spot(
    maturities: np.ndarray,
    spot: np.ndarray,
    forward: np.ndarray,
    title: Optional[str] = None,
    save: bool = True,
) -> go.Figure:
    """
    Compare spot (zero-coupon) yield curve to instantaneous forward curve.

    Blue = spot, Orange = forward.
    """
    title = title or "Spot vs Forward Rate Curve"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=maturities, y=spot,
        mode="lines+markers",
        name="Spot (Zero-Coupon)",
        line=dict(width=2.5, color="#1f77b4"),
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="Maturity: %{x:.2f}Y<br>Spot: %{y:.3f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=maturities, y=forward,
        mode="lines+markers",
        name="Forward Rate",
        line=dict(width=2.5, color="#ff7f0e", dash="dashdot"),
        marker=dict(size=8, color="#ff7f0e"),
        hovertemplate="Maturity: %{x:.2f}Y<br>Forward: %{y:.3f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Maturity (Years)",
        yaxis_title="Rate (%)",
    )

    if save:
        _save(fig, "forward_vs_spot")
    return fig


# ---------------------------------------------------------------------------
# 8. Full HTML report
# ---------------------------------------------------------------------------

def generate_html_report(
    maturities: np.ndarray,
    yields: np.ndarray,
    ns_result: dict,
    date_str: str,
    historical_dates: Optional[List[str]] = None,
    hist_yields: Optional[np.ndarray] = None,
    ns_betas_over_time: Optional[List[List[float]]] = None,
    indicators: Optional[Dict] = None,
    hist_indicators: Optional[Dict] = None,
) -> str:
    """
    Generate a comprehensive self-contained HTML report with all charts
    and analysis.

    Returns path to the saved report.
    """
    from bond_curve_lab.nelson_siegel import forward_rate_curve

    ns_params = ns_result.get("params", [])

    # Build individual chart figures
    # 1. Yield curve with NS fit
    fig1 = plot_yield_curve(
        maturities, yields, ns_fitted=np.array(ns_result.get("fitted_yields", [])),
        date_str=date_str, save=False,
    )

    # 2. Forward vs Spot
    if ns_params:
        dense_tenors = np.linspace(maturities[0], maturities[-1], 200)
        fwd = forward_rate_curve(ns_params, dense_tenors, model="ns")
        from bond_curve_lab.nelson_siegel import _ns_yield
        spot_dense = _ns_yield(dense_tenors, *ns_params)
        fig2 = plot_forward_vs_spot(dense_tenors, spot_dense, fwd, save=False)
    else:
        fig2 = go.Figure()

    # 3. Term premium history
    if hist_indicators:
        fig3 = plot_term_premium_history(
            hist_indicators["dates"],
            hist_indicators["term_premium_10y2y_bp"],
            save=False,
        )
    else:
        fig3 = go.Figure()

    # 4. NS components
    if ns_betas_over_time and historical_dates:
        fig4 = plot_curve_components(ns_betas_over_time, historical_dates, save=False)
    else:
        fig4 = go.Figure()

    # 5. Heatmap
    if historical_dates is not None and hist_yields is not None:
        fig5 = plot_heatmap(historical_dates, maturities, hist_yields, save=False)
    else:
        fig5 = go.Figure()

    # Build HTML
    def _chart_html(fig, chart_id):
        return f'<div id="{chart_id}" style="width:100%;min-height:500px;"></div>\n' + \
               f'<script>\n{fig.to_json()}\nPlotly.newPlot("{chart_id}", JSON.parse(document.currentScript.textContent).data, JSON.parse(document.currentScript.textContent).layout, {{responsive: true}});\n</script>'

    html_parts = [
        '<!DOCTYPE html><html lang="zh-CN"><head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>CGB Yield Curve Report — {date_str}</title>',
        '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>',
        '<style>',
        'body{font-family:"Segoe UI",Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f5f6fa;color:#2c3e50;}',
        'h1{color:#1a1a2e;border-bottom:3px solid #1f77b4;padding-bottom:10px;}',
        'h2{color:#1a1a2e;margin-top:40px;border-left:4px solid #2ca02c;padding-left:12px;}',
        '.card{background:#fff;border-radius:8px;padding:20px;margin:20px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08);}',
        '.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;}',
        '.stat-item{background:#f8f9fc;border-radius:6px;padding:16px;text-align:center;border:1px solid #e8e8e8;}',
        '.stat-value{font-size:1.8em;font-weight:700;color:#1f77b4;}',
        '.stat-label{font-size:0.85em;color:#7f8c8d;margin-top:4px;}',
        '.warning{color:#d62728;font-weight:600;}',
        '.param-table{width:100%;border-collapse:collapse;margin:10px 0;}',
        '.param-table th,.param-table td{border:1px solid #ddd;padding:8px 12px;text-align:left;}',
        '.param-table th{background:#1f77b4;color:#fff;}',
        '.param-table tr:nth-child(even){background:#f8f9fc;}',
        '.footer{text-align:center;color:#95a5a6;margin-top:40px;font-size:0.85em;}',
        '</style></head><body>',
        f'<h1>China Government Bond Yield Curve Analysis</h1>',
        f'<p style="color:#7f8c8d;">Report Date: {date_str} | Generated by bond-curve-lab v1.0.0</p>',
    ]

    # Key metrics card
    if indicators:
        html_parts.append('<div class="card"><h2>Key Indicators</h2><div class="stat-grid">')
        for key, label in [
            ("yield_10y", "10Y Yield"),
            ("yield_2y", "2Y Yield"),
            ("term_premium_10y2y_bp", "10Y-2Y Spread"),
            ("slope_30y3m_bp", "30Y-3M Slope"),
            ("curvature_2y5y10y_bp", "Butterfly (2Y-5Y-10Y)"),
        ]:
            val = indicators.get(key)
            if val is not None:
                warning_class = ""
                if key == "term_premium_10y2y_bp" and val < 0:
                    warning_class = " warning"
                html_parts.append(
                    f'<div class="stat-item"><div class="stat-value{warning_class}">{val:.1f}</div>'
                    f'<div class="stat-label">{label} (bp if spread)</div></div>'
                )
        if "real_yield_10y" in indicators:
            html_parts.append(
                f'<div class="stat-item"><div class="stat-value">{indicators["real_yield_10y"]:.2f}%</div>'
                f'<div class="stat-label">10Y Real Yield (est.)</div></div>'
            )
        if "signal" in indicators:
            html_parts.append(
                f'<div class="stat-item"><div class="stat-value" style="font-size:1em;color:#d62728;">'
                f'{indicators["signal"]}</div><div class="stat-label">Curve Signal</div></div>'
            )
        html_parts.append('</div></div>')

    # NS Model parameters
    if ns_params:
        html_parts.append('<div class="card"><h2>Nelson-Siegel Model Parameters</h2>')
        html_parts.append(f'<p><strong>Formula:</strong> <code>{ns_result.get("formula", "")}</code></p>')
        html_parts.append(f'<p>R-squared: <strong>{ns_result.get("r_squared", 0):.4f}</strong> | RMSE: <strong>{ns_result.get("rmse", 0):.4f}</strong></p>')
        html_parts.append('<table class="param-table"><tr><th>Parameter</th><th>Name</th><th>Value</th><th>Interpretation</th></tr>')
        param_names = ns_result.get("param_names", [])
        for i, (name, val) in enumerate(zip(param_names, ns_params)):
            cls = "ordinary"
            if name == "beta1" and val < -1:
                cls = "Steep upward slope"
            elif name == "beta1" and val > 0:
                cls = "Inverted curve signal"
            elif name == "beta0":
                cls = "Long-term equilibrium rate"
            elif name == "beta2":
                cls = "Medium-term curvature"
            else:
                cls = "Decay factor"
            html_parts.append(f'<tr><td>β<sub>{i}</sub></td><td>{name}</td><td>{val:.6f}</td><td>{cls}</td></tr>')
        html_parts.append('</table></div>')

    # Charts
    html_parts.append('<div class="card"><h2>Yield Curve with Nelson-Siegel Fit</h2>')
    html_parts.append(_chart_html(fig1, "chart_curve"))
    html_parts.append('</div>')

    if ns_params:
        html_parts.append('<div class="card"><h2>Spot vs Forward Rate Curve</h2>')
        html_parts.append(_chart_html(fig2, "chart_fwd"))
        html_parts.append('</div>')

    if hist_indicators:
        html_parts.append('<div class="card"><h2>10Y-2Y Term Premium History</h2>')
        html_parts.append(_chart_html(fig3, "chart_tp"))
        html_parts.append('</div>')

    if ns_betas_over_time:
        html_parts.append('<div class="card"><h2>Nelson-Siegel Factor Evolution</h2>')
        html_parts.append(_chart_html(fig4, "chart_components"))
        html_parts.append('</div>')

    if historical_dates is not None and hist_yields is not None:
        html_parts.append('<div class="card"><h2>Yield Curve Heatmap</h2>')
        html_parts.append(_chart_html(fig5, "chart_heatmap"))
        html_parts.append('</div>')

    html_parts.append(
        '<div class="footer">bond-curve-lab v1.0.0 | '
        f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
        'Nelson-Siegel Yield Curve Analysis</div>'
    )
    html_parts.append('</body></html>')

    html_content = "\n".join(html_parts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "bond_report.html"
    report_path.write_text(html_content, encoding="utf-8")

    return str(report_path)
