"""Validate Basilisk's downlink chain and couple it to the RF link budget.

Two groups of tests:

1. *Simulator behaviour* -- does ``spaceToGroundTransmitter`` gate on access,
   does ``simpleStorageUnit`` conserve bits, does the buffer respect its bounds.
2. *Link-budget coupling* -- Basilisk's access is geometric only, so these tests
   pin down when the simulated data volume is trustworthy and when it is
   optimistic, and prove the coupling responds in both directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_basilisk
from linkbudget import analysis, rf, scenario

pytestmark = [requires_basilisk, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Transmitter gating
# ---------------------------------------------------------------------------


def test_no_downlink_outside_access_windows(bsk_telemetry, access_mask):
    """The whole point of wiring the access message into the transmitter."""
    off_access = bsk_telemetry.tx_baud_bps[~access_mask]
    assert np.all(off_access == 0.0), (
        f"{np.count_nonzero(off_access)} samples downlinked without ground-station access"
    )


def test_downlink_rate_is_quantized_to_the_configured_baud(bsk_telemetry, comm_config):
    """Basilisk's transmitter is fixed-rate: it emits either 0 or -nodeBaudRate."""
    unique = np.unique(bsk_telemetry.tx_baud_bps)
    allowed = {0.0, -comm_config.downlink_baud_bps}
    assert set(unique).issubset(allowed), f"unexpected downlink rates: {unique}"


def test_downlink_actually_happens_during_passes(bsk_telemetry, access_mask, comm_config):
    active = bsk_telemetry.tx_baud_bps[access_mask] == -comm_config.downlink_baud_bps
    assert active.mean() > 0.5, (
        f"only {100 * active.mean():.0f}% of access time was spent transmitting"
    )


def test_transmitter_holds_off_until_a_full_packet_is_buffered():
    """With less than ``packetSize`` buffered, the transmitter must stay silent."""
    cfg = scenario.CommScenarioConfig(
        duration_s=6 * 3600.0,
        time_step_s=2.0,
        sample_step_s=2.0,
        instrument_baud_bps=0.0,          # nothing accumulates
        initial_buffer_bits=1e5,          # well below the packet size
        packet_size_bits=1e6,
    )
    telemetry = scenario.run_scenario(cfg)
    assert telemetry.has_access.any(), "test needs at least one access window"
    assert np.all(telemetry.tx_baud_bps == 0.0), (
        "transmitter downlinked despite an under-filled packet"
    )


def test_transmitter_drains_but_never_overdraws_the_buffer():
    cfg = scenario.CommScenarioConfig(
        duration_s=6 * 3600.0,
        time_step_s=1.0,
        sample_step_s=1.0,
        instrument_baud_bps=0.0,
        initial_buffer_bits=5e8,
        downlink_baud_bps=50e6,
        packet_size_bits=1e6,
    )
    telemetry = scenario.run_scenario(cfg)
    assert telemetry.stored_bits.min() >= -1e-6, "buffer went negative"
    assert telemetry.stored_bits[-1] < telemetry.stored_bits[0], "buffer never drained"


# ---------------------------------------------------------------------------
# Storage-unit bookkeeping
# ---------------------------------------------------------------------------


def test_net_baud_is_the_sum_of_the_attached_data_nodes(bsk_telemetry):
    """``simpleStorageUnit`` sums instrument supply and transmitter demand."""
    expected = bsk_telemetry.instrument_baud_bps + bsk_telemetry.tx_baud_bps
    # Skip the first sample: recorders latch before the first integration step.
    assert np.allclose(bsk_telemetry.net_baud_bps[1:], expected[1:], rtol=1e-9, atol=1e-6)


def test_stored_data_stays_within_its_bounds(bsk_telemetry, comm_config):
    assert bsk_telemetry.stored_bits.min() >= -1e-6
    assert bsk_telemetry.stored_bits.max() <= comm_config.storage_capacity_bits + 1e-6


def test_bit_conservation_across_the_whole_run(bsk_telemetry, comm_config):
    """Buffer change must equal generated bits minus downlinked bits."""
    dt = comm_config.sample_step_s
    generated = bsk_telemetry.instrument_baud_bps[1:].sum() * dt
    downlinked = -bsk_telemetry.tx_baud_bps[1:].sum() * dt
    measured = bsk_telemetry.stored_bits[-1] - bsk_telemetry.stored_bits[0]

    tolerance = 2.0 * comm_config.downlink_baud_bps * dt
    assert measured == pytest.approx(generated - downlinked, abs=tolerance), (
        f"buffer moved {measured:.3e} bits but the nodes account for "
        f"{generated - downlinked:.3e} bits"
    )


def test_buffer_integrates_the_net_baud_step_by_step(bsk_telemetry, comm_config):
    """Sample-to-sample: delta(storage) == netBaud * dt, away from the clamps."""
    dt = comm_config.sample_step_s
    delta = np.diff(bsk_telemetry.stored_bits)
    predicted = bsk_telemetry.net_baud_bps[1:] * dt

    # Ignore samples at the empty/full clamps, where the module saturates.
    interior = (bsk_telemetry.stored_bits[1:] > 1e6) & (
        bsk_telemetry.stored_bits[1:] < comm_config.storage_capacity_bits - 1e6
    )
    assert interior.any()
    error = np.abs(delta[interior] - predicted[interior])
    assert np.median(error) < 1.0, f"median integration error {np.median(error):.3e} bits"


def test_instrument_runs_at_its_configured_rate(bsk_telemetry, comm_config):
    assert np.allclose(
        bsk_telemetry.instrument_baud_bps[1:], comm_config.instrument_baud_bps, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Link-budget coupling
# ---------------------------------------------------------------------------


def test_nominal_link_closes_across_every_granted_pass(bsk_series, budget, access_mask):
    """Design verification for the reference 50 Mbps X-band configuration."""
    margins = bsk_series.margin_db[access_mask]
    assert margins.min() >= budget.required_margin_db, (
        f"worst in-pass margin {margins.min():.2f} dB is below the "
        f"{budget.required_margin_db:.1f} dB requirement"
    )


def test_cn0_is_highest_at_closest_approach(bsk_series, bsk_telemetry):
    windows = analysis.find_access_windows(bsk_telemetry.has_access)
    assert windows
    for start, stop in windows:
        cn0 = bsk_series.cn0_dbhz[start:stop]
        ranges = bsk_series.slant_range_m[start:stop]
        # Allow a sample of slack: C/N0 also carries the elevation-dependent
        # noise term, so the two extrema can land on adjacent samples.
        assert abs(int(np.argmax(cn0)) - int(np.argmin(ranges))) <= 2, (
            "peak C/N0 should coincide with minimum slant range"
        )


def test_healthy_link_reproduces_the_basilisk_data_volume(bsk_passes):
    """When the budget closes everywhere, RF-limited volume == simulated volume."""
    for p in bsk_passes:
        assert p.link_limited_volume_bits == pytest.approx(p.bsk_volume_bits, rel=1e-12)
        assert p.closure_fraction == pytest.approx(1.0, abs=1e-12)


def test_starved_link_cuts_the_deliverable_volume(bsk_telemetry):
    """Drop the transmit power and the RF layer must reject volume Basilisk reports.

    This is the headline result: Basilisk's downlink is geometry-limited, so its
    reported data volume does not move when the radio gets worse.  Only the link
    budget catches it.
    """
    nominal = rf.x_band_leo_downlink()
    starved = nominal.degraded(
        transmitter=rf.Transmitter(power_w=0.05, antenna_gain_dbi=0.0, line_loss_db=2.0)
    )

    nominal_series = analysis.evaluate_link(bsk_telemetry, nominal)
    starved_series = analysis.evaluate_link(bsk_telemetry, starved)

    bsk_volume = nominal_series.bsk_rate_bps.sum()
    assert starved_series.bsk_rate_bps.sum() == pytest.approx(bsk_volume), (
        "Basilisk's own downlink must be unaffected by the radio configuration"
    )
    assert nominal_series.effective_rate_bps.sum() == pytest.approx(bsk_volume)
    assert starved_series.effective_rate_bps.sum() < 0.5 * bsk_volume, (
        "the starved link should reject most of the simulated data volume"
    )


def test_marginal_link_is_rejected_only_at_low_elevation(bsk_telemetry, access_mask):
    """A deliberately marginal budget should close near zenith and fail near the mask."""
    marginal = rf.x_band_leo_downlink().degraded(data_rate_bps=50e6)
    # Trim the power until the link sits right on the edge at mid elevation.
    marginal = marginal.degraded(
        transmitter=rf.Transmitter(power_w=0.35, antenna_gain_dbi=6.0, line_loss_db=1.0,
                                   pointing_loss_db=1.0)
    )
    series = analysis.evaluate_link(bsk_telemetry, marginal)

    closed = series.link_closes[access_mask]
    if not (closed.any() and (~closed).any()):
        pytest.skip("chosen configuration is not marginal for this geometry")

    elevations = np.degrees(series.elevation_rad[access_mask])
    assert elevations[closed].min() > elevations[~closed].min(), (
        "link closure should favour higher elevations"
    )


def test_basilisk_grants_access_below_the_rf_viable_elevation(bsk_telemetry, comm_config):
    """Document the modelling gap: the access mask is not a link-closure test."""
    aggressive = rf.x_band_leo_downlink().degraded(data_rate_bps=600e6)
    closing_deg = analysis.recommended_elevation_mask_deg(
        aggressive, comm_config.altitude_m, comm_config.planet_radius_m
    )
    assert np.isfinite(closing_deg)
    assert closing_deg > comm_config.min_elevation_deg, (
        "test configuration should require a stricter mask than Basilisk enforces"
    )

    series = analysis.evaluate_link(bsk_telemetry, aggressive)
    granted = series.has_access.astype(bool)
    unusable = granted & ~series.link_closes
    assert unusable.any(), (
        "Basilisk should be granting access on samples where the link cannot close"
    )


def test_adaptive_modcod_beats_the_fixed_rate_over_a_pass(bsk_passes):
    """A rate-adaptive radio should extract more bits than the fixed 50 Mbps link.

    Quantifies what Basilisk's fixed-baud transmitter leaves on the table: near
    zenith the channel supports roughly two orders of magnitude more than the
    10-degree-elevation worst case the rate was sized for.
    """
    for p in bsk_passes:
        assert p.adaptive_volume_bits > p.bsk_volume_bits, (
            f"pass {p.index}: adaptive volume {p.adaptive_volume_bits:.3e} did not exceed "
            f"the fixed-rate volume {p.bsk_volume_bits:.3e}"
        )


# ---------------------------------------------------------------------------
# Cross-check against the analytic model
# ---------------------------------------------------------------------------


def test_basilisk_and_analytic_models_find_the_same_passes(bsk_telemetry, truth_telemetry):
    bsk_windows = analysis.find_access_windows(bsk_telemetry.has_access)
    truth_windows = analysis.find_access_windows(truth_telemetry.has_access)
    assert len(bsk_windows) == len(truth_windows), (
        f"Basilisk found {len(bsk_windows)} passes, the analytic model found "
        f"{len(truth_windows)}"
    )

    dt = bsk_telemetry.config.sample_step_s
    for (b_start, b_stop), (t_start, t_stop) in zip(bsk_windows, truth_windows):
        assert abs(b_start - t_start) * dt < 30.0, "acquisition-of-signal times disagree"
        assert abs(b_stop - t_stop) * dt < 30.0, "loss-of-signal times disagree"


def test_basilisk_and_analytic_models_agree_on_downlinked_volume(bsk_telemetry,
                                                                 truth_telemetry):
    dt = bsk_telemetry.config.sample_step_s
    bsk_bits = -bsk_telemetry.tx_baud_bps.sum() * dt
    truth_bits = -truth_telemetry.tx_baud_bps.sum() * dt
    assert bsk_bits == pytest.approx(truth_bits, rel=0.02), (
        f"Basilisk downlinked {bsk_bits / 1e9:.2f} Gbit, the analytic model "
        f"{truth_bits / 1e9:.2f} Gbit"
    )
