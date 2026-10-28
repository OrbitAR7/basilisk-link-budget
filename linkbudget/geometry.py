"""Independent orbit and ground-access truth model.

This is the reference implementation the Basilisk tests are graded against.
It is intentionally written a different way from Basilisk's ``GroundLocation``:
Basilisk builds the access geometry out of Eigen vectors and DCM products,
whereas this module solves Kepler's equation directly and uses spherical
trigonometry (law of cosines / law of sines) for range and elevation.  Two
independent formulations agreeing to machine precision is real evidence; two
copies of the same formulation agreeing is not.

No Basilisk import here, so these functions run whether or not the simulator
has been built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Equatorial radius Basilisk uses for Earth [m].  ``GroundLocation`` places
#: sites on a *sphere* of this radius (``LLA2PCPF`` takes a single radius),
#: so the truth model must do the same to be comparable.
EARTH_RADIUS_M = 6378136.6
#: Earth gravitational parameter [m^3/s^2], matching Basilisk's ``MU_EARTH``.
EARTH_MU_M3S2 = 0.3986004415e15
#: Earth rotation rate [rad/s].
EARTH_OMEGA_RADPS = 7.29211585275e-5


# ---------------------------------------------------------------------------
# Two-body propagation
# ---------------------------------------------------------------------------


@dataclass
class ClassicalElements:
    """Keplerian elements. Angles in radians, ``a`` in metres."""

    a: float
    e: float = 0.0
    i: float = 0.0
    raan: float = 0.0
    arg_periapsis: float = 0.0
    true_anomaly: float = 0.0

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.a**3 / EARTH_MU_M3S2)


def elements_to_rv(oe: ClassicalElements, mu=EARTH_MU_M3S2):
    """Classical elements -> inertial position/velocity vectors [m], [m/s]."""
    p = oe.a * (1.0 - oe.e**2)
    r = p / (1.0 + oe.e * np.cos(oe.true_anomaly))

    # Perifocal frame
    r_pqw = np.array([r * np.cos(oe.true_anomaly), r * np.sin(oe.true_anomaly), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array(
        [-np.sin(oe.true_anomaly), oe.e + np.cos(oe.true_anomaly), 0.0]
    )

    c_raan, s_raan = np.cos(oe.raan), np.sin(oe.raan)
    c_inc, s_inc = np.cos(oe.i), np.sin(oe.i)
    c_argp, s_argp = np.cos(oe.arg_periapsis), np.sin(oe.arg_periapsis)

    rot = np.array(
        [
            [
                c_raan * c_argp - s_raan * s_argp * c_inc,
                -c_raan * s_argp - s_raan * c_argp * c_inc,
                s_raan * s_inc,
            ],
            [
                s_raan * c_argp + c_raan * s_argp * c_inc,
                -s_raan * s_argp + c_raan * c_argp * c_inc,
                -c_raan * s_inc,
            ],
            [s_argp * s_inc, c_argp * s_inc, c_inc],
        ]
    )
    return rot @ r_pqw, rot @ v_pqw


def _solve_kepler(mean_anomaly, eccentricity, tol=1e-13, max_iter=100):
    """Newton-Raphson solution of M = E - e*sin(E)."""
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    eccentric = np.where(eccentricity < 0.8, mean_anomaly, np.pi * np.ones_like(mean_anomaly))
    for _ in range(max_iter):
        residual = eccentric - eccentricity * np.sin(eccentric) - mean_anomaly
        derivative = 1.0 - eccentricity * np.cos(eccentric)
        step = residual / derivative
        eccentric = eccentric - step
        if np.max(np.abs(step)) < tol:
            break
    return eccentric


def propagate_two_body(oe: ClassicalElements, times_s, mu=EARTH_MU_M3S2):
    """Analytic two-body propagation. Returns ``(r_N, v_N)`` of shape (N, 3)."""
    times_s = np.atleast_1d(np.asarray(times_s, dtype=float))
    mean_motion = np.sqrt(mu / oe.a**3)

    # Initial true anomaly -> eccentric -> mean anomaly
    e = oe.e
    ecc_anom0 = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(oe.true_anomaly / 2.0),
        np.sqrt(1.0 + e) * np.cos(oe.true_anomaly / 2.0),
    )
    mean_anom0 = ecc_anom0 - e * np.sin(ecc_anom0)

    mean_anom = mean_anom0 + mean_motion * times_s
    ecc_anom = _solve_kepler(mean_anom, e)
    true_anom = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(ecc_anom / 2.0),
        np.sqrt(1.0 - e) * np.cos(ecc_anom / 2.0),
    )

    positions = np.empty((times_s.size, 3))
    velocities = np.empty((times_s.size, 3))
    for idx, nu in enumerate(true_anom):
        step = ClassicalElements(oe.a, oe.e, oe.i, oe.raan, oe.arg_periapsis, float(nu))
        positions[idx], velocities[idx] = elements_to_rv(step, mu)
    return positions, velocities


# ---------------------------------------------------------------------------
# Ground site placement
# ---------------------------------------------------------------------------


def lla_to_pcpf(lat_rad, lon_rad, alt_m, planet_radius_m=EARTH_RADIUS_M):
    """Spherical geodetic -> planet-centred planet-fixed position [m].

    Matches Basilisk's ``LLA2PCPF`` when called with a scalar planet radius.
    """
    radius = planet_radius_m + alt_m
    return radius * np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )


def planet_dcm(theta_rad):
    """DCM mapping inertial N -> planet-fixed P for a z-axis rotation ``theta``.

    This is the ``J20002Pfix`` convention Basilisk's ``GroundLocation`` reads.
    """
    theta_rad = np.asarray(theta_rad, dtype=float)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array([[c, s, zero], [-s, c, zero], [zero, zero, one]])


def planet_dcm_dot(theta_rad, omega_radps):
    """Time derivative of :func:`planet_dcm`."""
    theta_rad = np.asarray(theta_rad, dtype=float)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    zero = np.zeros_like(c)
    return omega_radps * np.array([[-s, c, zero], [-c, -s, zero], [zero, zero, zero]])


def site_inertial_position(lat_rad, lon_rad, alt_m, times_s, theta0_rad=0.0,
                           omega_radps=EARTH_OMEGA_RADPS, planet_radius_m=EARTH_RADIUS_M):
    """Ground site position in the inertial frame over time. Shape (N, 3)."""
    times_s = np.atleast_1d(np.asarray(times_s, dtype=float))
    r_lp_p = lla_to_pcpf(lat_rad, lon_rad, alt_m, planet_radius_m)
    theta = theta0_rad + omega_radps * times_s
    # r_LP_N = dcm_PN^T * r_LP_P, done for every sample at once.
    c, s = np.cos(theta), np.sin(theta)
    return np.column_stack(
        [
            c * r_lp_p[0] - s * r_lp_p[1],
            s * r_lp_p[0] + c * r_lp_p[1],
            np.full_like(theta, r_lp_p[2]),
        ]
    )


# ---------------------------------------------------------------------------
# Access geometry -- spherical-trig formulation
# ---------------------------------------------------------------------------


def access_geometry(r_sc_n, r_site_n):
    """Range and elevation from site to spacecraft using spherical trig.

    Both arguments are (N, 3) inertial position arrays.  Rather than forming
    the line-of-sight vector and dotting it with the site normal (Basilisk's
    approach), this works from the two radii and the central angle between
    them, which is an algebraically distinct route to the same answer.

    Returns ``(slant_range_m, elevation_rad, central_angle_rad)``.
    """
    r_sc_n = np.atleast_2d(np.asarray(r_sc_n, dtype=float))
    r_site_n = np.atleast_2d(np.asarray(r_site_n, dtype=float))

    r_sc = np.linalg.norm(r_sc_n, axis=1)
    r_site = np.linalg.norm(r_site_n, axis=1)

    cos_gamma = np.clip(
        np.sum(r_sc_n * r_site_n, axis=1) / (r_sc * r_site), -1.0, 1.0
    )
    gamma = np.arccos(cos_gamma)

    # Law of cosines for the slant range.
    slant_range = np.sqrt(r_sc**2 + r_site**2 - 2.0 * r_sc * r_site * cos_gamma)

    # Elevation above the local horizon: the triangle site-centre-spacecraft
    # gives tan(el) = (r_sc*cos(gamma) - r_site) / (r_sc*sin(gamma)).
    elevation = np.arctan2(r_sc * cos_gamma - r_site, r_sc * np.sin(gamma))
    return slant_range, elevation, gamma


def has_access(elevation_rad, slant_range_m, min_elevation_rad, max_range_m=-1.0):
    """Basilisk's access predicate: strictly above the mask, within max range."""
    within_range = (
        np.ones_like(np.asarray(slant_range_m), dtype=bool)
        if max_range_m < 0
        else (np.asarray(slant_range_m) <= max_range_m)
    )
    return (np.asarray(elevation_rad) > min_elevation_rad) & within_range


def horizon_slant_range(orbit_radius_m, site_radius_m):
    """Slant range at exactly zero elevation -- the tangent-line length."""
    return np.sqrt(np.asarray(orbit_radius_m, dtype=float) ** 2 - site_radius_m**2)


# ---------------------------------------------------------------------------
# A complete synthetic telemetry run (used when Basilisk is unavailable)
# ---------------------------------------------------------------------------


def synthetic_access_telemetry(oe: ClassicalElements, lat_deg, lon_deg, alt_m,
                               duration_s, sample_dt_s, min_elevation_deg=10.0,
                               max_range_m=-1.0, theta0_rad=0.0,
                               planet_radius_m=EARTH_RADIUS_M,
                               omega_radps=EARTH_OMEGA_RADPS):
    """Produce the same fields Basilisk's access message carries, analytically."""
    times = np.arange(0.0, duration_s + sample_dt_s, sample_dt_s)
    r_sc, v_sc = propagate_two_body(oe, times)
    r_site = site_inertial_position(
        np.radians(lat_deg), np.radians(lon_deg), alt_m, times,
        theta0_rad, omega_radps, planet_radius_m,
    )
    slant_range, elevation, _ = access_geometry(r_sc, r_site)
    access = has_access(elevation, slant_range, np.radians(min_elevation_deg), max_range_m)
    return {
        "time_s": times,
        "r_BN_N": r_sc,
        "v_BN_N": v_sc,
        "r_LN_N": r_site,
        "slant_range_m": slant_range,
        "elevation_rad": elevation,
        "has_access": access.astype(int),
    }
