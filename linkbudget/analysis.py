"""Couple the RF link budget to Basilisk comm telemetry.

Basilisk's ``groundLocation`` decides access on geometry alone -- elevation
above a fixed mask -- and ``spaceToGroundTransmitter`` then downlinks at its
full configured baud rate for the whole window.  Nothing in the simulator asks
whether the radio link actually closes at 10 degrees elevation and 1900 km of
slant range.  This module adds that question, and quantifies the gap between
the *geometry-limited* data volume Basilisk reports and the *link-limited*
volume the RF budget will actually support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rf import LinkBudget


@dataclass
class LinkTimeSeries:
    """Per-sample link budget evaluated along a telemetry run."""

    time_s: np.ndarray
    elevation_rad: np.ndarray
    slant_range_m: np.ndarray
    has_access: np.ndarray
    cn0_dbhz: np.ndarray
    ebn0_db: np.ndarray
    margin_db: np.ndarray
    max_rate_bps: np.ndarray
    shannon_bps: np.ndarray
    link_closes: np.ndarray
    #: Rate Basilisk actually applied (positive bits/s; 0 when not downlinking).
    bsk_rate_bps: np.ndarray
    #: Same, but zeroed wherever the RF budget says the link does not close.
    effective_rate_bps: np.ndarray
    #: Rate an adaptive-modcod radio could have run at, capped by the hardware.
    adaptive_rate_bps: np.ndarray


def evaluate_link(telemetry, budget: LinkBudget, hardware_max_rate_bps=np.inf) -> LinkTimeSeries:
    """Apply ``budget`` to every sample of ``telemetry``.

    Off-access samples are still evaluated geometrically (the numbers are
    meaningless there and are masked out downstream), which keeps every array
    the same length as the telemetry.

    ``hardware_max_rate_bps`` caps the adaptive-modcod column; leave it at
    infinity to see what the channel alone would support, or set it to the
    modem's real ceiling.
    """
    elevation = np.asarray(telemetry.elevation_rad, dtype=float)
    slant_range = np.asarray(telemetry.slant_range_m, dtype=float)
    access = np.asarray(telemetry.has_access).astype(bool)

    # Guard against the degenerate zero range some recorders emit at t=0.
    safe_range = np.where(slant_range > 1.0, slant_range, 1.0)

    cn0 = budget.cn0_dbhz(safe_range, elevation)
    ebn0 = cn0 - 10.0 * np.log10(budget.data_rate_bps)
    margin = ebn0 - budget.threshold_ebn0_db
    closes = access & (margin >= budget.required_margin_db)

    max_rate = budget.max_data_rate_bps(safe_range, elevation)
    adaptive = np.where(access, np.minimum(max_rate, hardware_max_rate_bps), 0.0)

    bsk_rate = np.maximum(-np.asarray(telemetry.tx_baud_bps, dtype=float), 0.0)
    effective = np.where(closes, bsk_rate, 0.0)

    return LinkTimeSeries(
        time_s=np.asarray(telemetry.time_s, dtype=float),
        elevation_rad=elevation,
        slant_range_m=slant_range,
        has_access=access.astype(int),
        cn0_dbhz=cn0,
        ebn0_db=ebn0,
        margin_db=margin,
        max_rate_bps=max_rate,
        shannon_bps=budget.shannon_capacity_bps(safe_range, elevation),
        link_closes=closes,
        bsk_rate_bps=bsk_rate,
        effective_rate_bps=effective,
        adaptive_rate_bps=adaptive,
    )


# ---------------------------------------------------------------------------
# Pass extraction
# ---------------------------------------------------------------------------


def find_access_windows(has_access):
    """Return ``[(start_index, stop_index_exclusive), ...]`` for each window."""
    flags = np.asarray(has_access).astype(bool).astype(np.int8)
    if flags.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


@dataclass
class PassSummary:
    """Everything worth knowing about one ground-station contact."""

    index: int
    start_s: float
    stop_s: float
    duration_s: float
    max_elevation_deg: float
    min_slant_range_km: float
    max_slant_range_km: float
    peak_cn0_dbhz: float
    worst_cn0_dbhz: float
    worst_margin_db: float
    best_margin_db: float
    closure_fraction: float
    bsk_volume_bits: float
    link_limited_volume_bits: float
    adaptive_volume_bits: float

    @property
    def volume_shortfall_bits(self) -> float:
        """Bits Basilisk claims to have downlinked that the RF link cannot support."""
        return self.bsk_volume_bits - self.link_limited_volume_bits

    @property
    def shortfall_percent(self) -> float:
        if self.bsk_volume_bits <= 0:
            return 0.0
        return 100.0 * self.volume_shortfall_bits / self.bsk_volume_bits


def summarize_passes(series: LinkTimeSeries, sample_dt_s=None) -> list[PassSummary]:
    """Reduce a :class:`LinkTimeSeries` to one record per access window."""
    if sample_dt_s is None:
        sample_dt_s = float(np.median(np.diff(series.time_s))) if series.time_s.size > 1 else 1.0

    summaries = []
    for idx, (start, stop) in enumerate(find_access_windows(series.has_access)):
        sl = slice(start, stop)
        margin = series.margin_db[sl]
        summaries.append(
            PassSummary(
                index=idx,
                start_s=float(series.time_s[start]),
                stop_s=float(series.time_s[stop - 1]),
                duration_s=float(stop - start) * sample_dt_s,
                max_elevation_deg=float(np.degrees(series.elevation_rad[sl].max())),
                min_slant_range_km=float(series.slant_range_m[sl].min() / 1e3),
                max_slant_range_km=float(series.slant_range_m[sl].max() / 1e3),
                peak_cn0_dbhz=float(series.cn0_dbhz[sl].max()),
                worst_cn0_dbhz=float(series.cn0_dbhz[sl].min()),
                worst_margin_db=float(margin.min()),
                best_margin_db=float(margin.max()),
                closure_fraction=float(series.link_closes[sl].mean()),
                bsk_volume_bits=float(series.bsk_rate_bps[sl].sum() * sample_dt_s),
                link_limited_volume_bits=float(series.effective_rate_bps[sl].sum() * sample_dt_s),
                adaptive_volume_bits=float(series.adaptive_rate_bps[sl].sum() * sample_dt_s),
            )
        )
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_budget_table(budget: LinkBudget, slant_range_m, elevation_rad, title=None) -> str:
    """Render a conventional line-item link budget."""
    rows = budget.breakdown(slant_range_m, elevation_rad)
    width = max(len(k) for k in rows)
    header = title or "LINK BUDGET"
    lines = [header, "=" * max(len(header), width + 16)]
    for key, value in rows.items():
        lines.append(f"{key:<{width}}  {value:>12.3f}")
    return "\n".join(lines)


def format_pass_report(passes: list[PassSummary], budget: LinkBudget) -> str:
    """Human-readable per-pass table plus totals."""
    if not passes:
        return "No access windows found in this run.\n"

    lines = [
        f"ACCESS PASSES  (downlink configured at {budget.data_rate_bps / 1e6:.1f} Mbps, "
        f"{budget.modcod.name}, required margin {budget.required_margin_db:.1f} dB)",
        "  BSK = volume Basilisk reports   RF = same, minus samples where the link "
        "cannot close   ACM = rate-adaptive potential",
        "=" * 118,
        f"{'#':>2}  {'start [s]':>10}  {'dur [s]':>8}  {'max el':>7}  {'min rng':>9}  "
        f"{'peak C/N0':>10}  {'worst mg':>9}  {'closed':>7}  {'BSK [Gb]':>9}  {'RF [Gb]':>8}  {'ACM [Gb]':>9}",
        "-" * 118,
    ]
    for p in passes:
        lines.append(
            f"{p.index:>2}  {p.start_s:>10.0f}  {p.duration_s:>8.0f}  "
            f"{p.max_elevation_deg:>6.1f}d  {p.min_slant_range_km:>8.0f}k  "
            f"{p.peak_cn0_dbhz:>10.1f}  {p.worst_margin_db:>9.1f}  "
            f"{100 * p.closure_fraction:>6.0f}%  "
            f"{p.bsk_volume_bits / 1e9:>9.2f}  {p.link_limited_volume_bits / 1e9:>8.2f}  "
            f"{p.adaptive_volume_bits / 1e9:>9.2f}"
        )

    total_bsk = sum(p.bsk_volume_bits for p in passes)
    total_rf = sum(p.link_limited_volume_bits for p in passes)
    total_acm = sum(p.adaptive_volume_bits for p in passes)
    shortfall = 100.0 * (total_bsk - total_rf) / total_bsk if total_bsk > 0 else 0.0

    lines += [
        "-" * 118,
        f"{'':>2}  {'TOTAL':>10}  {sum(p.duration_s for p in passes):>8.0f}"
        f"{'':>48}{total_bsk / 1e9:>19.2f}  {total_rf / 1e9:>8.2f}  {total_acm / 1e9:>9.2f}",
        "",
        f"Basilisk (geometry-limited) volume : {total_bsk / 1e9:8.2f} Gbit",
        f"Link-budget-limited volume         : {total_rf / 1e9:8.2f} Gbit",
        f"Optimistic by                      : {shortfall:8.1f} %",
        f"Adaptive-modcod potential          : {total_acm / 1e9:8.2f} Gbit",
    ]
    return "\n".join(lines)


def recommended_elevation_mask_deg(budget: LinkBudget, altitude_m, planet_radius_m=6378136.6):
    """Elevation mask that would make every granted access window RF-viable."""
    return float(
        np.degrees(budget.min_closing_elevation_rad(altitude_m, planet_radius_m))
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_overview(telemetry, series: LinkTimeSeries, budget: LinkBudget, path=None):
    """Four-panel summary figure. Returns the matplotlib figure."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = series.time_s / 3600.0
    access = series.has_access.astype(bool)
    masked = lambda arr: np.where(access, arr, np.nan)  # noqa: E731

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(hours, np.degrees(series.elevation_rad), lw=0.8, color="0.7",
                 label="elevation (all)")
    axes[0].plot(hours, masked(np.degrees(series.elevation_rad)), lw=1.6, color="C0",
                 label="elevation (access)")
    mask_deg = telemetry.config.min_elevation_deg if telemetry.config else 10.0
    axes[0].axhline(mask_deg, ls="--", color="C3", lw=1.0, label="access mask")
    axes[0].set_ylabel("elevation [deg]")
    axes[0].set_ylim(0, 90)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Basilisk ground-station access geometry")

    axes[1].plot(hours, masked(series.slant_range_m / 1e3), lw=1.4, color="C1")
    axes[1].set_ylabel("slant range [km]")

    axes[2].plot(hours, masked(series.cn0_dbhz), lw=1.4, color="C2", label="C/N0")
    axes[2].set_ylabel("C/N0 [dB-Hz]")
    twin = axes[2].twinx()
    twin.plot(hours, masked(series.margin_db), lw=1.2, color="C3", label="margin")
    twin.axhline(budget.required_margin_db, ls="--", color="C3", lw=0.9)
    twin.axhline(0.0, ls=":", color="k", lw=0.9)
    twin.set_ylabel("link margin [dB]")
    axes[2].legend(loc="upper left", fontsize=8)
    twin.legend(loc="lower right", fontsize=8)
    axes[2].set_title(
        f"Link budget @ {budget.data_rate_bps / 1e6:.0f} Mbps, {budget.modcod.name} "
        f"(dashed = {budget.required_margin_db:.0f} dB required margin)"
    )

    axes[3].plot(hours, telemetry.stored_bits / 1e9, lw=1.4, color="C4", label="buffer [Gbit]")
    axes[3].set_ylabel("stored data [Gbit]")
    rate_axis = axes[3].twinx()
    rate_axis.plot(hours, series.bsk_rate_bps / 1e6, lw=1.0, color="C0",
                   label="Basilisk downlink")
    rate_axis.plot(hours, series.effective_rate_bps / 1e6, lw=1.6, ls="--", color="C3",
                   label="RF-supported")
    rate_axis.set_ylabel("downlink rate [Mbps]")
    rate_axis.legend(loc="upper right", fontsize=8)
    axes[3].set_xlabel("time [hours]")
    axes[3].set_title("Onboard buffer and downlink rate")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_pass_detail(series: LinkTimeSeries, budget: LinkBudget, pass_index=0, path=None):
    """Zoom into a single pass: elevation, C/N0, margin, supportable rate."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = find_access_windows(series.has_access)
    if not windows:
        raise ValueError("no access windows to plot")
    start, stop = windows[pass_index]
    sl = slice(start, stop)
    minutes = (series.time_s[sl] - series.time_s[start]) / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(minutes, np.degrees(series.elevation_rad[sl]), color="C0")
    axes[0].set_ylabel("elevation [deg]")
    axes[0].set_title(f"Pass {pass_index} detail")

    axes[1].plot(minutes, series.cn0_dbhz[sl], color="C2", label="C/N0")
    axes[1].set_ylabel("C/N0 [dB-Hz]")
    twin = axes[1].twinx()
    twin.plot(minutes, series.margin_db[sl], color="C3", label="margin")
    twin.axhline(budget.required_margin_db, ls="--", color="C3", lw=0.9)
    twin.axhline(0.0, ls=":", color="k", lw=0.9)
    twin.set_ylabel("margin [dB]")

    axes[2].plot(minutes, series.max_rate_bps[sl] / 1e6, color="C1",
                 label="max supportable (with margin)")
    axes[2].plot(minutes, series.shannon_bps[sl] / 1e6, color="0.5", ls=":",
                 label="Shannon capacity")
    axes[2].axhline(budget.data_rate_bps / 1e6, ls="--", color="C0", label="configured rate")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("rate [Mbps]")
    axes[2].set_xlabel("time from AOS [min]")
    axes[2].legend(loc="lower center", fontsize=8)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
