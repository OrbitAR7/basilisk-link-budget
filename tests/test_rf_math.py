"""Validate the RF link-budget layer against closed-form textbook results.

Nothing here touches Basilisk -- this is the harness testing itself, so that a
failure in the simulator tests can be attributed to the simulator rather than
to the radio math grading it.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkbudget import rf


# ---------------------------------------------------------------------------
# Constants and basic conversions
# ---------------------------------------------------------------------------


def test_boltzmann_term_is_the_classic_228_6():
    assert rf.BOLTZMANN_DB == pytest.approx(228.5991, abs=1e-3)


def test_db_round_trip():
    values = np.array([1e-12, 1.0, 3.5, 1e9])
    assert np.allclose(rf.from_db(rf.to_db(values)), values, rtol=1e-12)


def test_to_db_handles_zero_without_warning():
    assert rf.to_db(0.0) < -300.0


# ---------------------------------------------------------------------------
# Free-space path loss
# ---------------------------------------------------------------------------


def test_fspl_matches_engineering_shorthand():
    """FSPL[dB] = 32.44 + 20log10(d_km) + 20log10(f_MHz)."""
    budget = rf.LinkBudget(frequency_hz=2.2e9)
    for range_km in (500.0, 1000.0, 2000.0, 35786.0):
        expected = 32.44 + 20 * np.log10(range_km) + 20 * np.log10(2200.0)
        assert budget.fspl_db(range_km * 1e3) == pytest.approx(expected, abs=0.01)


def test_fspl_doubling_range_costs_6_02_db():
    budget = rf.x_band_leo_downlink()
    assert budget.fspl_db(2e6) - budget.fspl_db(1e6) == pytest.approx(6.0206, abs=1e-4)


def test_fspl_doubling_frequency_costs_6_02_db():
    low = rf.LinkBudget(frequency_hz=4.1e9).fspl_db(1e6)
    high = rf.LinkBudget(frequency_hz=8.2e9).fspl_db(1e6)
    assert high - low == pytest.approx(6.0206, abs=1e-4)


def test_fspl_is_vectorized():
    budget = rf.x_band_leo_downlink()
    ranges = np.array([5e5, 1e6, 2e6])
    assert budget.fspl_db(ranges).shape == ranges.shape


# ---------------------------------------------------------------------------
# Antenna and noise
# ---------------------------------------------------------------------------


def test_dish_gain_matches_aperture_formula():
    station = rf.GroundStation(dish_diameter_m=5.4, aperture_efficiency=0.6)
    frequency = 8.2e9
    wavelength = rf.SPEED_OF_LIGHT_MPS / frequency
    expected = 10 * np.log10(0.6 * (np.pi * 5.4 / wavelength) ** 2)
    assert station.antenna_gain_dbi(frequency) == pytest.approx(expected, abs=1e-9)


def test_doubling_dish_diameter_gains_6_02_db():
    small = rf.GroundStation(dish_diameter_m=3.0).antenna_gain_dbi(8.2e9)
    large = rf.GroundStation(dish_diameter_m=6.0).antenna_gain_dbi(8.2e9)
    assert large - small == pytest.approx(6.0206, abs=1e-4)


def test_eirp_is_power_plus_gain_minus_losses():
    tx = rf.Transmitter(power_w=10.0, antenna_gain_dbi=6.0, line_loss_db=1.0,
                        pointing_loss_db=0.5)
    assert tx.eirp_dbw == pytest.approx(10.0 + 6.0 - 1.0 - 0.5, abs=1e-9)


def test_atmospheric_loss_follows_cosecant_law():
    station = rf.GroundStation(zenith_atmos_loss_db=0.25, min_mapping_elevation_deg=5.0)
    assert station.atmos_loss_db(np.radians(90.0)) == pytest.approx(0.25, abs=1e-9)
    assert station.atmos_loss_db(np.radians(30.0)) == pytest.approx(0.50, abs=1e-9)
    assert station.atmos_loss_db(np.radians(10.0)) == pytest.approx(
        0.25 / np.sin(np.radians(10.0)), abs=1e-9
    )


def test_atmospheric_mapping_is_clamped_near_the_horizon():
    station = rf.GroundStation(min_mapping_elevation_deg=5.0)
    at_floor = station.atmos_loss_db(np.radians(5.0))
    below = station.atmos_loss_db(np.radians(0.5))
    assert below == pytest.approx(at_floor, abs=1e-12)
    assert np.isfinite(station.atmos_loss_db(0.0))


def test_system_noise_temperature_rises_toward_the_horizon():
    station = rf.GroundStation()
    elevations = np.radians([90.0, 45.0, 20.0, 10.0, 5.0])
    temps = station.system_noise_temp_k(elevations)
    assert np.all(np.diff(temps) > 0.0), "T_sys must increase as elevation drops"


def test_g_over_t_is_worse_at_low_elevation():
    station = rf.GroundStation()
    assert station.g_over_t_dbk(8.2e9, np.radians(10.0)) < station.g_over_t_dbk(
        8.2e9, np.radians(90.0)
    )


def test_zero_loss_atmosphere_gives_cold_sky():
    station = rf.GroundStation(zenith_atmos_loss_db=0.0, ground_spillover_temp_k=0.0)
    assert station.antenna_noise_temp_k(np.radians(90.0)) == pytest.approx(2.73, abs=1e-9)


# ---------------------------------------------------------------------------
# C/N0, Eb/N0, margin
# ---------------------------------------------------------------------------


def test_cn0_equals_hand_computed_budget():
    """Reassemble the budget by hand from its own components."""
    budget = rf.x_band_leo_downlink()
    slant_range, elevation = 1.2e6, np.radians(25.0)

    expected = (
        budget.transmitter.eirp_dbw
        - budget.fspl_db(slant_range)
        - budget.station.atmos_loss_db(elevation)
        - budget.rain_allowance_db
        - budget.polarization_loss_db
        - budget.station.pointing_loss_db
        - budget.other_loss_db
        + budget.station.g_over_t_dbk(budget.frequency_hz, elevation)
        + rf.BOLTZMANN_DB
    )
    assert budget.cn0_dbhz(slant_range, elevation) == pytest.approx(expected, abs=1e-9)


def test_ebn0_drops_3_01_db_when_the_rate_doubles():
    budget = rf.x_band_leo_downlink()
    slow = budget.ebn0_db(1e6, np.radians(30.0), 25e6)
    fast = budget.ebn0_db(1e6, np.radians(30.0), 50e6)
    assert slow - fast == pytest.approx(3.0103, abs=1e-4)


def test_margin_is_ebn0_minus_threshold():
    budget = rf.x_band_leo_downlink()
    ebn0 = budget.ebn0_db(1e6, np.radians(30.0))
    assert budget.margin_db(1e6, np.radians(30.0)) == pytest.approx(
        ebn0 - budget.threshold_ebn0_db, abs=1e-12
    )


def test_threshold_includes_implementation_loss():
    budget = rf.LinkBudget(modcod=rf.MODCODS["QPSK-LDPC1/2"], implementation_loss_db=1.5)
    assert budget.threshold_ebn0_db == pytest.approx(1.0 + 1.5, abs=1e-12)


def test_max_data_rate_inverts_to_exactly_the_required_margin():
    """Running at max_data_rate must leave precisely required_margin_db of margin."""
    budget = rf.x_band_leo_downlink()
    for elevation_deg in (5.0, 10.0, 30.0, 90.0):
        elevation = np.radians(elevation_deg)
        slant_range = rf.slant_range_from_elevation(elevation, 500e3)
        rate = budget.max_data_rate_bps(slant_range, elevation)
        margin = budget.margin_db(slant_range, elevation, rate)
        assert margin == pytest.approx(budget.required_margin_db, abs=1e-9)


def test_max_rate_never_exceeds_shannon_capacity():
    """The supportable rate must sit below Shannon *at the bandwidth it would occupy*.

    Comparing against the bandwidth of the currently configured rate would be
    meaningless -- a higher rate buys itself proportionally more spectrum.
    """
    budget = rf.x_band_leo_downlink()
    elevations = np.radians(np.linspace(5.0, 90.0, 40))
    ranges = rf.slant_range_from_elevation(elevations, 500e3)

    supported = budget.max_data_rate_bps(ranges, elevations)
    occupied = supported / budget.modcod.spectral_efficiency_bps_per_hz
    capacity = budget.shannon_capacity_bps(ranges, elevations, bandwidth_hz=occupied)
    assert np.all(supported < capacity), "practical rate must sit below the Shannon bound"


def test_configured_rate_is_below_shannon_wherever_the_link_closes():
    budget = rf.x_band_leo_downlink()
    elevations = np.radians(np.linspace(5.0, 90.0, 40))
    ranges = rf.slant_range_from_elevation(elevations, 500e3)
    closes = budget.closes(ranges, elevations)
    capacity = budget.shannon_capacity_bps(ranges, elevations)
    assert np.all(budget.data_rate_bps < capacity[closes])


def test_margin_improves_monotonically_with_elevation():
    budget = rf.x_band_leo_downlink()
    elevations = np.radians(np.linspace(5.0, 90.0, 60))
    ranges = rf.slant_range_from_elevation(elevations, 500e3)
    margins = budget.margin_db(ranges, elevations)
    assert np.all(np.diff(margins) > 0.0)


def test_nominal_x_band_link_closes_at_the_10_degree_mask():
    """Design verification: the reference 50 Mbps X-band link must close at the mask."""
    budget = rf.x_band_leo_downlink()
    elevation = np.radians(10.0)
    slant_range = rf.slant_range_from_elevation(elevation, 500e3)
    margin = budget.margin_db(slant_range, elevation)
    assert margin >= budget.required_margin_db, (
        f"worst-case margin {margin:.2f} dB is below the {budget.required_margin_db} dB "
        "requirement"
    )


def test_starving_the_transmitter_breaks_the_link():
    budget = rf.x_band_leo_downlink()
    elevation = np.radians(10.0)
    slant_range = rf.slant_range_from_elevation(elevation, 500e3)
    starved = budget.degraded(transmitter=rf.Transmitter(power_w=0.005, antenna_gain_dbi=0.0))
    assert budget.closes(slant_range, elevation)
    assert not starved.closes(slant_range, elevation)


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------


def test_slant_range_at_zenith_equals_altitude():
    assert rf.slant_range_from_elevation(np.radians(90.0), 500e3) == pytest.approx(500e3, abs=1e-6)


def test_slant_range_at_horizon_equals_tangent_length():
    radius = 6378136.6
    expected = np.sqrt((radius + 500e3) ** 2 - radius**2)
    assert rf.slant_range_from_elevation(0.0, 500e3, radius) == pytest.approx(expected, rel=1e-12)


def test_slant_range_decreases_with_elevation():
    elevations = np.radians(np.linspace(0.0, 90.0, 50))
    ranges = rf.slant_range_from_elevation(elevations, 500e3)
    assert np.all(np.diff(ranges) < 0.0)


# ---------------------------------------------------------------------------
# Closure elevation and reporting
# ---------------------------------------------------------------------------


def test_min_closing_elevation_brackets_the_margin_crossing():
    budget = rf.x_band_leo_downlink().degraded(data_rate_bps=400e6)
    closing = budget.min_closing_elevation_rad(500e3)
    assert np.isfinite(closing)

    just_above = closing + np.radians(0.5)
    just_below = closing - np.radians(0.5)
    assert budget.closes(rf.slant_range_from_elevation(just_above, 500e3), just_above)
    assert not budget.closes(rf.slant_range_from_elevation(just_below, 500e3), just_below)


def test_min_closing_elevation_is_nan_when_the_link_never_closes():
    hopeless = rf.x_band_leo_downlink().degraded(data_rate_bps=1e12)
    assert np.isnan(hopeless.min_closing_elevation_rad(500e3))


def test_breakdown_is_internally_consistent():
    budget = rf.x_band_leo_downlink()
    rows = budget.breakdown(1.2e6, np.radians(25.0))
    assert rows["Eb/N0 [dB]"] == pytest.approx(
        rows["C/N0 [dB-Hz]"] - 10 * np.log10(budget.data_rate_bps), abs=1e-9
    )
    assert rows["Margin [dB]"] == pytest.approx(
        rows["Eb/N0 [dB]"] - rows["Required Eb/N0 [dB]"], abs=1e-9
    )
    assert rows["EIRP [dBW]"] == pytest.approx(budget.transmitter.eirp_dbw, abs=1e-9)


@pytest.mark.parametrize("name", sorted(rf.MODCODS))
def test_every_modcod_is_physically_sensible(name):
    modcod = rf.MODCODS[name]
    assert 0.0 < modcod.code_rate <= 1.0
    assert modcod.bits_per_symbol >= 1.0
    assert -2.0 < modcod.required_ebn0_db < 20.0
    assert modcod.spectral_efficiency_bps_per_hz > 0.0


def test_higher_order_modcods_need_more_ebn0_per_bit_per_hz():
    """Sanity: spectral efficiency is bought with Eb/N0."""
    ordered = sorted(rf.MODCODS.values(), key=lambda m: m.spectral_efficiency_bps_per_hz)
    efficiencies = [m.spectral_efficiency_bps_per_hz for m in ordered]
    assert efficiencies == sorted(efficiencies)
    assert ordered[-1].required_ebn0_db > rf.MODCODS["QPSK-LDPC1/2"].required_ebn0_db
