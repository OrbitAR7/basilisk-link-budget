#!/usr/bin/env python3
"""Run the Basilisk comm scenario and report the downlink link budget.

Examples
--------
    python run_link_budget.py                       # Basilisk if available, else synthetic
    python run_link_budget.py --synthetic           # force the analytic model
    python run_link_budget.py --band s --rate 2e6   # S-band cubesat downlink
    python run_link_budget.py --plots out/          # write figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from linkbudget import analysis, rf, scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true",
                        help="use the analytic truth model instead of Basilisk")
    parser.add_argument("--band", choices=["x", "s"], default="x",
                        help="reference radio configuration (default: x)")
    parser.add_argument("--rate", type=float, default=None,
                        help="downlink data rate in bits/s (overrides the band default)")
    parser.add_argument("--modcod", choices=sorted(rf.MODCODS), default=None,
                        help="modulation and coding scheme")
    parser.add_argument("--margin", type=float, default=3.0,
                        help="required link margin in dB (default: 3)")
    parser.add_argument("--mask", type=float, default=10.0,
                        help="ground-station elevation mask in degrees (default: 10)")
    parser.add_argument("--duration", type=float, default=12.0,
                        help="simulation duration in hours (default: 12)")
    parser.add_argument("--step", type=float, default=1.0,
                        help="integration step in seconds (default: 1)")
    parser.add_argument("--sample", type=float, default=5.0,
                        help="telemetry sampling step in seconds (default: 5)")
    parser.add_argument("--tx-power", type=float, default=None,
                        help="spacecraft transmit power in watts")
    parser.add_argument("--dish", type=float, default=None,
                        help="ground station dish diameter in metres")
    parser.add_argument("--plots", type=Path, default=None,
                        help="directory to write figures into")
    return parser


def make_budget(args) -> rf.LinkBudget:
    budget = rf.x_band_leo_downlink() if args.band == "x" else rf.s_band_leo_downlink()
    if args.rate is not None:
        budget.data_rate_bps = args.rate
    if args.modcod is not None:
        budget.modcod = rf.MODCODS[args.modcod]
    if args.tx_power is not None:
        budget.transmitter.power_w = args.tx_power
    if args.dish is not None:
        budget.station.dish_diameter_m = args.dish
    budget.required_margin_db = args.margin
    return budget


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    budget = make_budget(args)

    cfg = scenario.CommScenarioConfig(
        min_elevation_deg=args.mask,
        duration_s=args.duration * 3600.0,
        time_step_s=args.step,
        sample_step_s=args.sample,
        downlink_baud_bps=budget.data_rate_bps,
    )

    use_synthetic = args.synthetic or not scenario.HAVE_BASILISK
    if use_synthetic and not args.synthetic:
        print("! Basilisk is not importable; falling back to the analytic model.")
        print(f"!   import error: {scenario.BASILISK_IMPORT_ERROR!r}\n")

    telemetry = (
        scenario.run_synthetic_scenario(cfg) if use_synthetic else scenario.run_scenario(cfg)
    )
    series = analysis.evaluate_link(telemetry, budget)
    passes = analysis.summarize_passes(series, cfg.sample_step_s)

    print(f"Telemetry source : {telemetry.source}")
    print(f"Samples          : {len(telemetry)}  over {cfg.duration_s / 3600:.1f} h\n")

    # Worst-case geometry: the mask elevation, where the link is hardest to close.
    worst_range = rf.slant_range_from_elevation(
        np.radians(cfg.min_elevation_deg), cfg.altitude_m, cfg.planet_radius_m
    )
    print(analysis.format_budget_table(
        budget, worst_range, np.radians(cfg.min_elevation_deg),
        title=f"WORST-CASE LINK BUDGET  (elevation = {cfg.min_elevation_deg:.0f} deg)",
    ))
    print()
    zenith_range = rf.slant_range_from_elevation(np.radians(90.0), cfg.altitude_m,
                                                 cfg.planet_radius_m)
    print(analysis.format_budget_table(
        budget, zenith_range, np.radians(90.0), title="BEST-CASE LINK BUDGET  (zenith)"
    ))
    print()
    print(analysis.format_pass_report(passes, budget))
    print()

    closing_mask = analysis.recommended_elevation_mask_deg(
        budget, cfg.altitude_m, cfg.planet_radius_m
    )
    if np.isnan(closing_mask):
        print("The link does not close at ANY elevation with this configuration.")
    else:
        print(f"Elevation mask required for link closure : {closing_mask:.2f} deg")
        print(f"Elevation mask configured in Basilisk    : {cfg.min_elevation_deg:.2f} deg")
        if closing_mask > cfg.min_elevation_deg:
            print("  -> Basilisk grants access below the RF-viable elevation; the "
                  "simulated data volume is optimistic.")
        else:
            print("  -> The configured mask is conservative; the link closes across "
                  "every granted window.")

    if args.plots is not None:
        args.plots.mkdir(parents=True, exist_ok=True)
        analysis.plot_overview(telemetry, series, budget, args.plots / "overview.png")
        if passes:
            analysis.plot_pass_detail(series, budget, 0, args.plots / "pass_0.png")
        print(f"\nFigures written to {args.plots}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
