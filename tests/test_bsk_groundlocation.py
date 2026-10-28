"""Validate Basilisk's ``groundLocation`` access geometry.

Everything the link budget consumes -- slant range and elevation -- comes out of
this one Basilisk module, so it gets graded first.  Basilisk computes access by
forming the line-of-sight vector, rotating it through ``dcm_LP * dcm_PN``, and
dotting it against the site normal.  :mod:`linkbudget.geometry` instead solves
Kepler's equation and applies the law of cosines.  Where the two agree to
machine precision, the geometry is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_basilisk
from linkbudget import geometry as geo

pytestmark = [requires_basilisk, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Site placement
# ---------------------------------------------------------------------------


def test_site_sits_on_the_sphere_basilisk_was_told_to_use(bsk_telemetry, comm_config):
    """``LLA2PCPF`` takes a single radius, so the site radius must be exactly R+alt."""
    radii = np.linalg.norm(bsk_telemetry.r_LN_N, axis=1)
    expected = comm_config.planet_radius_m + comm_config.station_alt_m
    assert np.allclose(radii, expected, rtol=1e-12), (
        f"site radius drifted: {radii.min():.3f}..{radii.max():.3f} m vs {expected:.3f} m"
    )


def test_site_inertial_track_matches_the_commanded_earth_rotation(bsk_telemetry, comm_config):
    """The DCM chain inside Basilisk must reproduce the rotation we published."""
    expected = geo.site_inertial_position(
        np.radians(comm_config.station_lat_deg),
        np.radians(comm_config.station_lon_deg),
        comm_config.station_alt_m,
        bsk_telemetry.time_s,
        comm_config.earth_theta0_rad,
        comm_config.earth_omega_radps if comm_config.earth_rotation else 0.0,
        comm_config.planet_radius_m,
    )
    error = np.linalg.norm(bsk_telemetry.r_LN_N - expected, axis=1)
    assert error.max() < 1e-3, f"site position error up to {error.max():.6f} m"


def test_site_latitude_is_preserved_under_rotation(bsk_telemetry, comm_config):
    radii = np.linalg.norm(bsk_telemetry.r_LN_N, axis=1)
    latitudes = np.degrees(np.arcsin(bsk_telemetry.r_LN_N[:, 2] / radii))
    assert np.allclose(latitudes, comm_config.station_lat_deg, atol=1e-9)


# ---------------------------------------------------------------------------
# Orbit propagation
# ---------------------------------------------------------------------------


def test_circular_orbit_radius_is_conserved(bsk_telemetry, comm_config):
    """Point-mass gravity plus a circular initial state should hold the radius."""
    radii = np.linalg.norm(bsk_telemetry.r_BN_N, axis=1)
    expected = comm_config.planet_radius_m + comm_config.altitude_m
    drift = np.abs(radii - expected).max()
    assert drift < 1e3, f"orbit radius drifted {drift:.1f} m over the run"


def test_specific_orbital_energy_is_conserved(bsk_telemetry):
    radii = np.linalg.norm(bsk_telemetry.r_BN_N, axis=1)
    speeds = np.linalg.norm(bsk_telemetry.v_BN_N, axis=1)
    energy = 0.5 * speeds**2 - geo.EARTH_MU_M3S2 / radii
    assert np.abs(energy - energy[0]).max() / abs(energy[0]) < 1e-6


def test_integrated_trajectory_matches_analytic_two_body(bsk_telemetry, comm_config):
    """Basilisk integrates numerically; compare against the closed-form solution."""
    expected, _ = geo.propagate_two_body(comm_config.classical_elements(),
                                         bsk_telemetry.time_s)
    error = np.linalg.norm(bsk_telemetry.r_BN_N - expected, axis=1)
    assert error.max() < 5e3, (
        f"integrator diverged from the Kepler solution by {error.max() / 1e3:.2f} km "
        f"over {bsk_telemetry.time_s[-1] / 3600:.1f} h"
    )


# ---------------------------------------------------------------------------
# Access geometry -- the core comparison
# ---------------------------------------------------------------------------


def test_slant_range_matches_spherical_trigonometry(bsk_telemetry):
    """Compare against the law of cosines using Basilisk's own state vectors.

    Feeding Basilisk's ``r_BN_N``/``r_LN_N`` back in isolates the access math
    from any propagation difference, so the tolerance can be very tight.
    """
    expected, _, _ = geo.access_geometry(bsk_telemetry.r_BN_N, bsk_telemetry.r_LN_N)
    assert np.allclose(bsk_telemetry.slant_range_m, expected, rtol=1e-12, atol=1e-6)


def test_elevation_matches_spherical_trigonometry(bsk_telemetry):
    _, expected, _ = geo.access_geometry(bsk_telemetry.r_BN_N, bsk_telemetry.r_LN_N)
    error = np.abs(bsk_telemetry.elevation_rad - expected)
    assert error.max() < 1e-9, f"elevation differs by up to {np.degrees(error.max()):.3e} deg"


def test_range_and_elevation_satisfy_the_law_of_cosines(bsk_telemetry):
    """Independent identity: d^2 = r_sc^2 + r_gs^2 - 2*r_sc*r_gs*cos(gamma)."""
    r_sc = np.linalg.norm(bsk_telemetry.r_BN_N, axis=1)
    r_gs = np.linalg.norm(bsk_telemetry.r_LN_N, axis=1)
    cos_gamma = np.clip(
        np.sum(bsk_telemetry.r_BN_N * bsk_telemetry.r_LN_N, axis=1) / (r_sc * r_gs), -1.0, 1.0
    )
    expected = np.sqrt(r_sc**2 + r_gs**2 - 2.0 * r_sc * r_gs * cos_gamma)
    assert np.allclose(bsk_telemetry.slant_range_m, expected, rtol=1e-12, atol=1e-6)


def test_elevation_stays_within_physical_bounds(bsk_telemetry):
    degrees = np.degrees(bsk_telemetry.elevation_rad)
    assert degrees.max() <= 90.0 + 1e-9
    assert degrees.min() >= -90.0 - 1e-9


def test_azimuth_stays_within_pi(bsk_telemetry):
    assert np.abs(bsk_telemetry.azimuth_rad).max() <= np.pi + 1e-12


def test_range_rate_matches_the_derivative_of_slant_range(bsk_telemetry, access_mask):
    """``range_dot`` is reported in the rotating SEZ frame; |r_BL| is frame-invariant,
    so it must equal the numerical time derivative of the slant range."""
    numeric = np.gradient(bsk_telemetry.slant_range_m, bsk_telemetry.time_s)
    # Exclude window edges, where the central difference straddles a discontinuity
    # in which spacecraft is nearest -- and only compare where access is granted.
    interior = np.zeros_like(access_mask)
    interior[2:-2] = access_mask[2:-2]
    assert interior.any()
    error = np.abs(numeric[interior] - bsk_telemetry.range_rate_mps[interior])
    assert np.median(error) < 5.0, f"median range-rate error {np.median(error):.3f} m/s"


def test_range_rate_changes_sign_at_closest_approach(bsk_telemetry):
    """Every pass should show an approaching leg and a receding leg."""
    from linkbudget.analysis import find_access_windows

    windows = find_access_windows(bsk_telemetry.has_access)
    assert windows, "no access windows recorded"
    for start, stop in windows:
        segment = bsk_telemetry.range_rate_mps[start:stop]
        if segment.size < 5:
            continue
        assert segment[0] < 0.0 < segment[-1], "range rate should go from closing to opening"


# ---------------------------------------------------------------------------
# The access predicate itself
# ---------------------------------------------------------------------------


def test_access_flag_matches_the_elevation_mask(bsk_telemetry, comm_config):
    expected = geo.has_access(
        bsk_telemetry.elevation_rad,
        bsk_telemetry.slant_range_m,
        np.radians(comm_config.min_elevation_deg),
        comm_config.max_range_m,
    )
    mismatches = np.flatnonzero(expected.astype(int) != bsk_telemetry.has_access)
    assert mismatches.size == 0, (
        f"{mismatches.size} samples disagree with the mask predicate, "
        f"first at t={bsk_telemetry.time_s[mismatches[0]]:.1f} s"
    )


def test_access_is_never_granted_below_the_mask(bsk_telemetry, comm_config, access_mask):
    below = np.degrees(bsk_telemetry.elevation_rad[access_mask]) <= comm_config.min_elevation_deg
    assert not below.any()


def test_access_flag_is_binary(bsk_telemetry):
    assert set(np.unique(bsk_telemetry.has_access)).issubset({0, 1})


def test_maximum_range_constraint_is_honoured(comm_config):
    """A tight max-range clamp must cut the windows down, not be ignored."""
    from linkbudget import scenario

    tight = scenario.CommScenarioConfig(
        **{**comm_config.__dict__, "max_range_m": 1000e3, "duration_s": 6 * 3600.0,
           "sample_step_s": 5.0, "time_step_s": 2.0}
    )
    telemetry = scenario.run_scenario(tight)
    granted = telemetry.has_access.astype(bool)
    if granted.any():
        assert telemetry.slant_range_m[granted].max() <= 1000e3 + 1e-6


# ---------------------------------------------------------------------------
# Pass-level sanity
# ---------------------------------------------------------------------------


def test_run_contains_several_boulder_passes(bsk_passes):
    assert len(bsk_passes) >= 2, (
        f"expected multiple passes over 12 h, got {len(bsk_passes)}"
    )


def test_pass_durations_are_physically_reasonable(bsk_passes):
    for p in bsk_passes:
        assert 60.0 < p.duration_s < 900.0, (
            f"pass {p.index} lasted {p.duration_s:.0f} s, which is not a 500 km LEO pass"
        )


def test_closest_approach_never_beats_the_orbit_altitude(bsk_passes, comm_config):
    for p in bsk_passes:
        assert p.min_slant_range_km >= comm_config.altitude_m / 1e3 - 1.0


def test_horizon_range_bounds_every_pass(bsk_passes, comm_config):
    """No sample inside a window can be farther than the zero-elevation tangent range."""
    orbit_radius = comm_config.planet_radius_m + comm_config.altitude_m
    site_radius = comm_config.planet_radius_m + comm_config.station_alt_m
    limit_km = geo.horizon_slant_range(orbit_radius, site_radius) / 1e3
    for p in bsk_passes:
        assert p.max_slant_range_km <= limit_km + 1.0
