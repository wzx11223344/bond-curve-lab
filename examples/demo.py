#!/usr/bin/env python3
"""
bond-curve-lab Demo
====================

Full pipeline demonstration:
  1. Fetch latest CGB spot yields
  2. Fit Nelson-Siegel + NSS models
  3. Compute key indicators
  4. Fetch historical time-series data
  5. Generate all charts
  6. Produce comprehensive HTML report

Run: python examples/demo.py
"""

import sys
import os

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import numpy as np
from rich.console import Console
from rich.table import Table

from bond_curve_lab import (
    ChinaBondFetcher,
    fetch_spot_yields,
    fetch_historical_curves,
    fit_nelson_siegel,
    fit_nelson_siegel_svensson,
    interpret_params,
    forward_rate_curve,
    compute_all_indicators,
    compute_historical_indicators,
    plot_yield_curve,
    plot_curve_animation,
    plot_3d_surface,
    plot_heatmap,
    plot_term_premium_history,
    plot_curve_components,
    plot_forward_vs_spot,
    generate_html_report,
)
from bond_curve_lab.nelson_siegel import fit_ns_over_time

console = Console()
TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]


def main():
    console.rule("[bold blue]bond-curve-lab — China Bond Yield Curve Analysis Lab[/bold blue]")
    console.print("[dim]Nelson-Siegel Yield Curve Modeling & Visualization[/dim]\n")

    # ------------------------------------------------------------------
    # Step 1: Fetch spot yields
    # ------------------------------------------------------------------
    console.print("[bold]Step 1/6: Fetching latest CGB spot yields...[/bold]")
    maturities, yields = fetch_spot_yields()
    date_str = datetime.now().strftime("%Y-%m-%d")

    table = Table(title=f"CGB Spot Yields — {date_str}")
    table.add_column("Tenor", style="cyan")
    table.add_column("Yield (%)", style="green", justify="right")
    for t, y in zip([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], yields):
        table.add_row(f"{t:.2f}Y", f"{y:.4f}")
    console.print(table)

    # ------------------------------------------------------------------
    # Step 2: Fit NS and NSS models
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 2/6: Fitting Nelson-Siegel & NSS models...[/bold]")

    ns_result = fit_nelson_siegel(maturities, yields)
    console.print(f"  NS:  R-squared={ns_result['r_squared']:.4f}, "
                  f"RMSE={ns_result['rmse']:.4f}")

    try:
        nss_result = fit_nelson_siegel_svensson(maturities, yields)
        console.print(f"  NSS: R-squared={nss_result['r_squared']:.4f}, "
                      f"RMSE={nss_result['rmse']:.4f}")
    except Exception as e:
        console.print(f"  [yellow]NSS fit failed: {e}[/yellow]")
        nss_result = None

    # Parameter interpretation
    interp = interpret_params(ns_result["params"], model="ns")
    for k, v in interp.items():
        console.print(f"  [cyan]{k}:[/cyan] {v}")

    # ------------------------------------------------------------------
    # Step 3: Compute indicators
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 3/6: Computing bond market indicators...[/bold]")

    fetcher = ChinaBondFetcher()
    try:
        cpi = fetcher.fetch_cpi()
        console.print(f"  CPI YoY: {cpi:.2f}%")
    except Exception:
        cpi = None

    indicators = compute_all_indicators(maturities, yields, cpi=cpi)
    for k, v in indicators.items():
        if k == "signal":
            console.print(f"  [bold red]Signal: {v}[/bold red]")
        else:
            console.print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Step 4: Fetch historical data
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 4/6: Fetching historical yield curves...[/bold]")
    dates, matrix = fetch_historical_curves()
    console.print(f"  Fetched {len(dates)} daily curves ({dates[0]} to {dates[-1]})")

    hist_indicators = compute_historical_indicators(dates, matrix, maturities)

    # Fit NS over time
    console.print("  Fitting NS model over time...")
    ns_time = fit_ns_over_time(dates, matrix, maturities)
    valid_fits = sum(1 for b in ns_time["betas"] if not np.isnan(b[0]))
    console.print(f"  Successfully fitted {valid_fits}/{len(dates)} curves")

    # ------------------------------------------------------------------
    # Step 5: Generate all charts
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 5/6: Generating interactive charts...[/bold]")

    # 5a. Yield curve with NS/NSS fit
    ns_fitted = np.array(ns_result.get("fitted_yields", []))
    nss_fitted = np.array(nss_result["fitted_yields"]) if nss_result else None
    p1 = plot_yield_curve(maturities, yields, ns_fitted=ns_fitted, nss_fitted=nss_fitted, date_str=date_str)
    console.print(f"  [green]Yield curve chart saved[/green]")

    # 5b. Forward vs Spot
    if ns_result["params"]:
        dense = np.linspace(0.1, 30, 200)
        from bond_curve_lab.nelson_siegel import _ns_yield
        spot_dense = _ns_yield(dense, *ns_result["params"])
        fwd = forward_rate_curve(ns_result["params"], dense)
        p2 = plot_forward_vs_spot(dense, spot_dense, fwd)
        console.print(f"  [green]Forward vs Spot chart saved[/green]")

    # 5c. Term premium history
    tp_bp = np.array(hist_indicators["term_premium_10y2y_bp"])
    p3 = plot_term_premium_history(dates, tp_bp)
    console.print(f"  [green]Term premium history chart saved[/green]")

    # 5d. NS component evolution
    ns_betas = ns_time["betas"]
    if any(not np.isnan(b[0]) for b in ns_betas):
        p4 = plot_curve_components(ns_betas, dates)
        console.print(f"  [green]NS component evolution chart saved[/green]")

    # 5e. 3D surface
    if len(dates) > 120:
        step = len(dates) // 60
        plot_dates = dates[::step]
        plot_matrix = matrix[::step]
    else:
        plot_dates = dates
        plot_matrix = matrix
    p5 = plot_3d_surface(plot_dates, maturities, plot_matrix)
    console.print(f"  [green]3D yield surface chart saved[/green]")

    # 5f. Heatmap
    p6 = plot_heatmap(plot_dates, maturities, plot_matrix)
    console.print(f"  [green]Heatmap chart saved[/green]")

    # 5g. Animation
    sample_dates = dates[-24:]  # last 24 days
    sample_curves = [matrix[i] for i in range(len(dates) - 24, len(dates))]
    p7 = plot_curve_animation(sample_dates, sample_curves, maturities)
    console.print(f"  [green]Curve animation chart saved[/green]")

    # ------------------------------------------------------------------
    # Step 6: Full HTML report
    # ------------------------------------------------------------------
    console.print("\n[bold]Step 6/6: Generating comprehensive HTML report...[/bold]")
    report_path = generate_html_report(
        maturities=maturities,
        yields=yields,
        ns_result=ns_result,
        date_str=date_str,
        historical_dates=dates,
        hist_yields=matrix,
        ns_betas_over_time=ns_betas,
        indicators=indicators,
        hist_indicators=hist_indicators,
    )
    console.print(f"  [green]HTML report saved: {report_path}[/green]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    console.rule("[bold green]Demo Complete[/bold green]")
    console.print(f"\n[bold]Output files in output/:[/bold]")
    from bond_curve_lab.viz import OUTPUT_DIR
    for f in sorted(OUTPUT_DIR.glob("*.html")):
        console.print(f"  [cyan]{f.name}[/cyan]")


if __name__ == "__main__":
    main()
