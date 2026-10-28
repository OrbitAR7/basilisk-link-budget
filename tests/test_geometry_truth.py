"""Self-tests for the independent geometry model.

The Basilisk tests grade the simulator against :mod:`linkbudget.geometry`, so
that reference has to be trustworthy on its own first.  Everything here checks
it against analytic identities that do not depend on either implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkbudget import geometry as geo


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def test_circular_orbit_keeps_constant_radius():
    oe = geo.ClassicalElements(a=geo.EARTH_RADIUS_M + 500e3, e=0.0, i=np.radians(97.4))
    times = np.linspace(0.0, oe.period_s, 200)
    positions, _ = geo.propagate_two_body(oe, times)
    radii = np.linalg.norm(positions, axis=1)
    assert np.allclose(radii, oe.a, rtol=1e-12)


def test_circular_orbit_keeps_constant_speed():
    oe = geo.ClassicalElements(a=geo.EARTH_RADIUS_M + 500e3, e=0.0, i=np.radians(51.6))
    times = np.linspace(0.0, oe.period_s, 200)
    _, velocities = geo.propagate_two_body(oe, times)
    speeds = np.linalg.norm(velocities, axis=1)
    assert np.allclose(speeds, np.sqrt(geo.EARTH_MU_M3S2 / oe.a), rtol=1e-12)


def test_orbit_closes_after_one_period():
    oe = geo.ClassicalElements(a=7000e3, e=0.15, i=np.radians(30.0), raan=np.radians(20.0),
                               arg_periapsis=np.radians(40.0), true_anomaly=np.radians(10.0))
    positions, velocities = geo.propagate_two_body(oe, [0.0, oe.period_s])
    assert np.allclose(positions[0], positions[1], atol=1e-3)
    assert np.allclose(velocities[0], velocities[1], atol=1e-6)


def test_specific_energy_and_angular_momentum_are_conserved():
    oe = geo.ClassicalElements(a=8000e3, e=0.25, i=np.radians(63.4),
                               arg_periapsis=np.radians(90.0))
    times = np.linspace(0.0, 2 * oe.period_s, 300)
    positions, velocities = geo.propagate_two_body(oe, times)

    radii = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    energy = 0.5 * speeds**2 - geo.EARTH_MU_M3S2 / radii
    momentum = np.cross(positions, velocities)

    assert np.allclose(energy, energy[0], rtol=1e-11)
    # Compare against the vector magnitude: the component along the node line is
    # numerically zero here, so a pure relative tolerance is meaningless for it.
    scale = np.linalg.norm(momentum[0])
    assert np.allclose(momentum, momentum[0], rtol=1e-11, atol=1e-11 * scale)


def test_kepler_solver_inverts_the_kepler_equation():
    for e in (0.0, 0.1, 0.5, 0.9):
        mean = np.linspace(-np.pi, np.pi, 101)
        ecc = geo._solve_kepler(mean, e)
        assert np.allclose(ecc - e * np.sin(ecc), mean, atol=1e-11)


def test_inclination_bounds_the_ground_track_latitude():
    inclination_deg = 51.6
    oe = geo.ClassicalElements(a=geo.EARTH_RADIUS_M + 400e3, i=np.radians(inclination_deg))
    positions, _ = geo.propagate_two_body(oe, np.linspace(0.0, oe.period_s, 500))
    latitudes = np.degrees(np.arcsin(positions[:, 2] / np.linalg.norm(positions, axis=1)))
    assert latitudes.max() == pytest.approx(inclination_deg, abs=0.5)
    assert latitudes.min() == pytest.approx(-inclination_deg, abs=0.5)


# ---------------------------------------------------------------------------
# Site placement and planet rotation
# ---------------------------------------------------------------------------


def test_site_radius_is_planet_radius_plus_altitude():
    site = geo.lla_to_pcpf(np.radians(40.0), np.radians(-105.0), 1624.0)
    assert np.linalg.norm(site) == pytest.approx(geo.EARTH_RADIUS_M + 1624.0, rel=1e-14)


def test_site_latitude_and_longitude_round_trip():
    lat, lon = np.radians(40.009971), np.radians(-105.243895)
    site = geo.lla_to_pcpf(lat, lon, 1624.0)
    assert np.arcsin(site[2] / np.linalg.norm(site)) == pytest.approx(lat, abs=1e-12)
    assert np.arctan2(site[1], site[0]) == pytest.approx(lon, abs=1e-12)


def test_planet_dcm_is_orthonormal():
    for theta in np.linspace(0.0, 2 * np.pi, 17):
        dcm = geo.planet_dcm(theta)
        assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(dcm) == pytest.approx(1.0, abs=1e-14)


def test_planet_dcm_dot_matches_a_finite_difference():
    theta, omega, h = 0.7, geo.EARTH_OMEGA_RADPS, 1e-4
    numeric = (geo.planet_dcm(theta + omega * h) - geo.planet_dcm(theta - omega * h)) / (2 * h)
    assert np.allclose(numeric, geo.planet_dcm_dot(theta, omega), atol=1e-12)


def test_angular_velocity_recovered_from_dcm_pair_is_about_z():
    """The same extraction Basilisk performs: w_tilde = -dcm_dot * dcm^T."""
    theta, omega = 1.3, geo.EARTH_OMEGA_RADPS
    w_tilde = -geo.planet_dcm_dot(theta, omega) @ geo.planet_dcm(theta).T
    w = np.array([w_tilde[2, 1], w_tilde[0, 2], w_tilde[1, 0]])
    assert np.allclose(w, [0.0, 0.0, omega], atol=1e-18)


def test_site_traces_a_circle_of_constant_radius():
    times = np.linspace(0.0, 86400.0, 500)
    site = geo.site_inertial_position(np.radians(40.0), np.radians(-105.0), 1624.0, times)
    radii = np.linalg.norm(site, axis=1)
    assert np.allclose(radii, geo.EARTH_RADIUS_M + 1624.0, rtol=1e-13)
    assert np.allclose(site[:, 2], site[0, 2], rtol=1e-13)


def test_site_returns_to_its_start_after_one_sidereal_day():
    sidereal_day = 2 * np.pi / geo.EARTH_OMEGA_RADPS
    site = geo.site_inertial_position(np.radians(40.0), np.radians(-105.0), 1624.0,
                                      [0.0, sidereal_day])
    assert np.allclose(site[0], site[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Access geometry
# ---------------------------------------------------------------------------


def test_zenith_pass_gives_90_degree_elevation_and_altitude_range():
    site = geo.lla_to_pcpf(0.0, 0.0, 0.0)
    overhead = site * (geo.EARTH_RADIUS_M + 500e3) / geo.EARTH_RADIUS_M
    slant, elevation, gamma = geo.access_geometry([overhead], [site])
    assert elevation[0] == pytest.approx(np.pi / 2, abs=1e-12)
    assert slant[0] == pytest.approx(500e3, rel=1e-12)
    assert gamma[0] == pytest.approx(0.0, abs=1e-7)


def test_horizon_geometry_gives_zero_elevation_and_tangent_range():
    """Place the spacecraft exactly on the local horizon plane."""
    radius_site = geo.EARTH_RADIUS_M
    radius_sc = geo.EARTH_RADIUS_M + 500e3
    gamma = np.arccos(radius_site / radius_sc)  # central angle to the horizon

    site = np.array([radius_site, 0.0, 0.0])
    spacecraft = radius_sc * np.array([np.cos(gamma), np.sin(gamma), 0.0])

    slant, elevation, _ = geo.access_geometry([spacecraft], [site])
    assert elevation[0] == pytest.approx(0.0, abs=1e-9)
    assert slant[0] == pytest.approx(
        geo.horizon_slant_range(radius_sc, radius_site), rel=1e-12
    )


def test_access_geometry_agrees_with_a_vector_formulation():
    """Cross-check spherical trig against the dot-product route on random geometry."""
    rng = np.random.default_rng(20260804)
    site = geo.lla_to_pcpf(np.radians(40.0), np.radians(-105.0), 1624.0)
    sites = np.tile(site, (400, 1))

    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    spacecraft = directions * (geo.EARTH_RADIUS_M + 500e3)

    slant, elevation, _ = geo.access_geometry(spacecraft, sites)

    line_of_sight = spacecraft - sites
    slant_vec = np.linalg.norm(line_of_sight, axis=1)
    site_hat = sites / np.linalg.norm(sites, axis=1, keepdims=True)
    elevation_vec = np.pi / 2 - np.arccos(
        np.clip(np.sum(site_hat * line_of_sight, axis=1) / slant_vec, -1.0, 1.0)
    )

    assert np.allclose(slant, slant_vec, rtol=1e-12)
    assert np.allclose(elevation, elevation_vec, atol=1e-9)


def test_elevation_is_negative_for_a_spacecraft_below_the_horizon():
    site = np.array([geo.EARTH_RADIUS_M, 0.0, 0.0])
    antipodal = np.array([-(geo.EARTH_RADIUS_M + 500e3), 0.0, 0.0])
    _, elevation, _ = geo.access_geometry([antipodal], [site])
    assert elevation[0] < 0.0


def test_access_predicate_uses_a_strict_mask_and_range_limit():
    mask = np.radians(10.0)
    elevation = np.array([np.radians(9.9), mask, np.radians(10.1), np.radians(45.0)])
    ranges = np.array([1e6, 1e6, 1e6, 3e6])

    unlimited = geo.has_access(elevation, ranges, mask)
    assert unlimited.tolist() == [False, False, True, True]

    limited = geo.has_access(elevation, ranges, mask, max_range_m=2e6)
    assert limited.tolist() == [False, False, True, False]


# ---------------------------------------------------------------------------
# Full synthetic run
# ---------------------------------------------------------------------------


def test_synthetic_run_produces_plausible_boulder_passes():
    oe = geo.ClassicalElements(
        a=geo.EARTH_RADIUS_M + 500e3, i=np.radians(97.4), raan=np.radians(45.0)
    )
    data = geo.synthetic_access_telemetry(
        oe, 40.009971, -105.243895, 1624.0, duration_s=12 * 3600.0, sample_dt_s=5.0
    )

    access = data["has_access"].astype(bool)
    assert access.any(), "a 12 h SSO run over Boulder should produce at least one pass"

    # Elevation never exceeds zenith, range never beats the orbit altitude.
    assert np.degrees(data["elevation_rad"]).max() <= 90.0 + 1e-9
    assert data["slant_range_m"][access].min() >= 500e3 - 1e3

    # Every access sample is above the mask, by construction of the predicate.
    assert np.all(np.degrees(data["elevation_rad"][access]) > 10.0)

    # Pass durations for a 500 km LEO at a 10 deg mask land in the 3-12 min band.
    edges = np.diff(np.concatenate(([0], access.astype(np.int8), [0])))
    durations = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * 5.0
    assert np.all(durations > 60.0)
    assert np.all(durations < 900.0)
