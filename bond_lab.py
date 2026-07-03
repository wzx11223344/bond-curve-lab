#!/usr/bin/env python3
"""
bond_lab.py — CLI entry for bond-curve-lab
===========================================

China Bond Yield Curve Analysis Lab.

Usage:
    python bond_lab.py curve      # Latest yield curve chart
    python bond_lab.py fit         # Nelson-Siegel fitted curve
    python bond_lab.py history     # 10Y-2Y spread history
    python bond_lab.py 3d          # 3D yield surface
    python bond_lab.py report      # Full HTML report
"""

import argparse
import sys
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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


def _print_ns_params(result: dict) -> None:
    """Pretty-print Nelson-Siegel results."""
    table = Table(title=result.get("model", "Model") + " Results")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for name, val in zip(result.get("param_names", []), result.get("params", [])):
        table.add_row(name, f"{val:.6f}")

    table.add_row("R-squared", f"{result.get('r_squared', 0):.6f}")
    table.add_row("RMSE", f"{result.get('rmse', 0):.6f}")
    console.print(table)


def cmd_curve(args):
    """Show latest yield curve chart."""
    with console.status("[bold green]Fetching latest CGB yield curve..."):
        maturities, yields = fetch_spot_yields()
        date_str = datetime.now().strftime("%Y-%m-%d")

    console.print(Panel.fit(f"[bold blue]CGB Yield Curve — {date_str}[/bold blue]"))
    plot_yield_curve(maturities, yields, date_str=date_str)
    console.print("[green]Chart saved to output/yield_curve.html[/green]")


def cmd_fit(args):
    """Fit Nelson-Siegel model and show curve."""
    with console.status("[bold green]Fetching data and fitting Nelson-Siegel model..."):
        maturities, yields = fetch_spot_yields()
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Fit NS
        ns_result = fit_nelson_siegel(maturities, yields)
        _print_ns_params(ns_result)

        # Interpret
        interp = interpret_params(ns_result["params"], model="ns")
        console.print(Panel.fit("\n".join(f"[bold]{k}:[/bold] {v}" for k, v in interp.items()),
                                 title="Parameter Interpretation"))

        # Also try NSS
        try:
            nss_result = fit_nelson_siegel_svensson(maturities, yields)
            console.print(f"\n[bold]NSS:[/bold] R-squared={nss_result.get('r_squared',0):.6f}, "
                          f"RMSE={nss_result.get('rmse',0):.6f}")
        except Exception as e:
            console.print(f"[yellow]NSS fit skipped: {e}[/yellow]")
            nss_result = None

        # Plot
        ns_fitted = np.array(ns_result.get("fitted_yields", []))
        nss_fitted = np.array(nss_result["fitted_yields"]) if nss_result else None
        plot_yield_curve(maturities, yields, ns_fitted=ns_fitted, nss_fitted=nss_fitted,
                         date_str=date_str)

        # Forward vs spot
        if ns_result["params"]:
            dense = np.linspace(maturities[0], maturities[-1], 200)
            from bond_curve_lab.nelson_siegel import _ns_yield
            spot_dense = _ns_yield(dense, *ns_result["params"])
            fwd = forward_rate_curve(ns_result["params"], dense)
            plot_forward_vs_spot(dense, spot_dense, fwd)
            console.print("[green]Charts saved to output/[/green]")


def cmd_history(args):
    """Show 10Y-2Y spread history."""
    with console.status("[bold green]Fetching historical data..."):
        dates, matrix = fetch_historical_curves()
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        indicators = compute_historical_indicators(dates, matrix, maturities)

    tp_bp = np.array(indicators["term_premium_10y2y_bp"])
    latest = tp_bp[-1]
    min_val = tp_bp.min()
    max_val = tp_bp.max()

    console.print(Panel.fit(
        f"[bold]10Y-2Y Term Premium[/bold]\n"
        f"Latest: [bold]{'red' if latest < 0 else 'green'}]{latest:.2f} bp[/]\n"
        f"Range: {min_val:.2f} to {max_val:.2f} bp\n"
        f"Period: {dates[0]} to {dates[-1]}"
    ))

    plot_term_premium_history(dates, tp_bp)
    console.print("[green]Chart saved to output/term_premium_history.html[/green]")


def cmd_3d(args):
    """Show 3D yield surface."""
    with console.status("[bold green]Fetching historical data for 3D surface..."):
        dates, matrix = fetch_historical_curves()
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

    console.print(f"Data: {len(dates)} days, 10 maturities")

    # Sample if too many dates
    if len(dates) > 120:
        step = len(dates) // 60
        dates = dates[::step]
        matrix = matrix[::step]
        console.print(f"[yellow]Downsampled to {len(dates)} dates for readability[/yellow]")

    plot_3d_surface(dates, maturities, matrix)
    plot_heatmap(dates, maturities, matrix)
    console.print("[green]Charts saved to output/3d_surface.html and output/heatmap.html[/green]")


def cmd_report(args):
    """Generate full HTML report."""
    console.print("[bold]Generating comprehensive bond yield curve report...[/bold]\n")

    with console.status("[bold green]Step 1/4: Fetching spot data..."):
        maturities, yields = fetch_spot_yields()
        date_str = datetime.now().strftime("%Y-%m-%d")

    console.print(f"  Spot yields fetched: {date_str}")

    with console.status("[bold green]Step 2/4: Fitting Nelson-Siegel model..."):
        ns_result = fit_nelson_siegel(maturities, yields)
        _print_ns_params(ns_result)

    with console.status("[bold green]Step 3/4: Fetching historical data..."):
        fetcher = ChinaBondFetcher(use_cache=True)
        dates, matrix = fetcher.fetch_historical_curves()
        hist_indicators = compute_historical_indicators(dates, matrix, maturities)

    console.print(f"  Historical: {len(dates)} days")

    with console.status("[bold green]Step 4/4: Fitting NS over time..."):
        ns_time = fit_ns_over_time(dates, matrix, maturities)
        ns_betas = ns_time["betas"]

    # Current indicators
    try:
        cpi = fetcher.fetch_cpi()
    except Exception:
        cpi = None

    indicators = compute_all_indicators(maturities, yields, cpi=cpi)
    indicators["date"] = date_str

    # Generate report
    with console.status("[bold green]Generating HTML report..."):
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

    console.print(f"\n[bold green]Report saved:[/bold green] {report_path}")
    console.print(Panel.fit(f"[bold]Report Summary[/bold]\n"
                            f"Date: {date_str}\n"
                            f"10Y-2Y Spread: {indicators.get('term_premium_10y2y_bp', 'N/A'):.1f} bp\n"
                            f"10Y Yield: {indicators.get('yield_10y', 'N/A'):.3f}%\n"
                            f"NS R-squared: {ns_result.get('r_squared', 0):.4f}\n"
                            f"Signal: {indicators.get('signal', 'N/A')}"))


def main():
    parser = argparse.ArgumentParser(
        description="China Bond Yield Curve Analysis Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bond_lab.py curve       Latest yield curve chart
  python bond_lab.py fit          Nelson-Siegel fitted curve
  python bond_lab.py history      10Y-2Y spread history
  python bond_lab.py 3d           3D yield surface
  python bond_lab.py report       Full HTML report
        """,
    )
    parser.add_argument("command", nargs="?", default="curve",
                        choices=["curve", "fit", "history", "3d", "report"],
                        help="Command to run")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable cache, force fresh data fetch")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for charts")

    args = parser.parse_args()

    # Update output dir if specified
    if args.output_dir:
        from bond_curve_lab.viz import OUTPUT_DIR
        import pathlib
        setattr(sys.modules["bond_curve_lab.viz"], "OUTPUT_DIR", pathlib.Path(args.output_dir))

    try:
        if args.command == "curve":
            cmd_curve(args)
        elif args.command == "fit":
            cmd_fit(args)
        elif args.command == "history":
            cmd_history(args)
        elif args.command == "3d":
            cmd_3d(args)
        elif args.command == "report":
            cmd_report(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
