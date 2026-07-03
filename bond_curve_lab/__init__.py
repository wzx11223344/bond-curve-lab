"""
bond-curve-lab: China Bond Yield Curve Analysis Lab
====================================================

A quantitative toolkit for fetching, modeling, and visualizing
Chinese government bond yield curves using the Nelson-Siegel model.
"""

__version__ = "1.0.0"
__author__ = "bond-curve-lab"

from bond_curve_lab.fetcher import (
    ChinaBondFetcher,
    fetch_spot_yields,
    fetch_historical_curves,
)
from bond_curve_lab.nelson_siegel import (
    fit_nelson_siegel,
    fit_nelson_siegel_svensson,
    interpret_params,
    forward_rate_curve,
    discount_factors,
)
from bond_curve_lab.indicators import (
    term_premium,
    slope_steepness,
    curvature,
    real_yield,
    credit_spread,
    compute_all_indicators,
    compute_historical_indicators,
)
from bond_curve_lab.viz import (
    plot_yield_curve,
    plot_curve_animation,
    plot_3d_surface,
    plot_heatmap,
    plot_term_premium_history,
    plot_curve_components,
    plot_forward_vs_spot,
    generate_html_report,
)
