"""RF link-budget model for a space-to-ground downlink.

This module is deliberately free of any Basilisk dependency: it is pure
numpy so that the radio math can be unit-tested against closed-form
textbook results without building the simulator.  ``analysis.py`` is what
couples it to Basilisk telemetry.

Sign / unit conventions
-----------------------
* Everything internal is SI (metres, hertz, watts, kelvin, bits/second).
* Every quantity ending in ``_db``/``_dbi``/``_dbw``/``_dbhz`` is already
  logarithmic; every quantity ending in ``_w``/``_k``/``_hz``/``_bps`` is linear.
* Losses are stored as *positive* dB numbers and subtracted from the budget.
* All range/elevation inputs accept scalars or numpy arrays and broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SPEED_OF_LIGHT_MPS = 299792458.0
BOLTZMANN_W_PER_K_HZ = 1.380649e-23
#: -10*log10(k), the classic "228.6" term in the C/N0 equation [dB(W/K/Hz)]
BOLTZMANN_DB = -10.0 * np.log10(BOLTZMANN_W_PER_K_HZ)


def to_db(linear):
    """10*log10(x), with a guard so that zero maps to -inf rather than a warning."""
    linear = np.asarray(linear, dtype=float)
    return 10.0 * np.log10(np.where(linear > 0.0, linear, np.finfo(float).tiny))


def from_db(decibels):
    """Inverse of :func:`to_db`."""
    return np.power(10.0, np.asarray(decibels, dtype=float) / 10.0)


# ---------------------------------------------------------------------------
# Modulation / coding schemes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModCod:
    """A modulation-and-coding point.

    ``required_ebn0_db`` is the Eb/N0 needed at the demodulator input to hit
    ``target_ber``, *including* the coding gain of ``code_rate``.  The values
    below are representative textbook/CCSDS numbers -- replace them with your
    modem vendor's datasheet before flying anything.
    """

    name: str
    bits_per_symbol: float
    code_rate: float
    required_ebn0_db: float
    target_ber: float = 1e-6

    @property
    def spectral_efficiency_bps_per_hz(self) -> float:
        """Information bits per second per hertz of occupied bandwidth."""
        return self.bits_per_symbol * self.code_rate


#: Small library of common downlink modcods.
MODCODS = {
    "BPSK-uncoded": ModCod("BPSK-uncoded", 1.0, 1.0, 10.5),
    "QPSK-uncoded": ModCod("QPSK-uncoded", 2.0, 1.0, 10.5),
    "QPSK-conv1/2": ModCod("QPSK-conv1/2", 2.0, 0.5, 4.5),
    "QPSK-conv1/2-RS": ModCod("QPSK-conv1/2-RS", 2.0, 0.5 * 223 / 255, 2.5),
    "QPSK-LDPC1/2": ModCod("QPSK-LDPC1/2", 2.0, 0.5, 1.0),
    "QPSK-LDPC7/8": ModCod("QPSK-LDPC7/8", 2.0, 0.875, 3.4),
    "8PSK-LDPC2/3": ModCod("8PSK-LDPC2/3", 3.0, 2.0 / 3.0, 4.0),
    "16APSK-LDPC3/4": ModCod("16APSK-LDPC3/4", 4.0, 0.75, 7.4),
}


# ---------------------------------------------------------------------------
# Terminal definitions
# ---------------------------------------------------------------------------


@dataclass
class Transmitter:
    """Spacecraft-side radio."""

    power_w: float = 5.0
    antenna_gain_dbi: float = 6.0
    line_loss_db: float = 1.0
    pointing_loss_db: float = 1.0

    @property
    def power_dbw(self) -> float:
        return float(to_db(self.power_w))

    @property
    def eirp_dbw(self) -> float:
        """Effective isotropic radiated power, net of feed and pointing loss."""
        return self.power_dbw + self.antenna_gain_dbi - self.line_loss_db - self.pointing_loss_db


@dataclass
class GroundStation:
    """Ground-side radio.

    G/T is referred to the LNA input: the feed loss is taken off the antenna
    gain, and the antenna noise is attenuated by that same feed loss with the
    feed's own thermal contribution added back.
    """

    dish_diameter_m: float = 5.4
    aperture_efficiency: float = 0.60
    feed_loss_db: float = 0.5
    feed_physical_temp_k: float = 290.0
    lna_noise_temp_k: float = 75.0
    ground_spillover_temp_k: float = 15.0
    pointing_loss_db: float = 0.3
    #: Zenith clear-air (gaseous) attenuation at the operating band.
    zenith_atmos_loss_db: float = 0.25
    #: Mean physical temperature of the absorbing atmosphere.
    atmos_medium_temp_k: float = 275.0
    cosmic_background_k: float = 2.73
    #: Elevation below which the cosecant mapping function is frozen, because
    #: 1/sin(el) is unphysical near the horizon.
    min_mapping_elevation_deg: float = 5.0

    def antenna_gain_dbi(self, frequency_hz: float) -> float:
        wavelength_m = SPEED_OF_LIGHT_MPS / frequency_hz
        return float(
            to_db(self.aperture_efficiency * (np.pi * self.dish_diameter_m / wavelength_m) ** 2)
        )

    def atmos_mapping_factor(self, elevation_rad):
        """Slant-path air-mass relative to zenith (ITU-R P.618 cosecant law)."""
        elevation_rad = np.asarray(elevation_rad, dtype=float)
        floor_rad = np.radians(self.min_mapping_elevation_deg)
        return 1.0 / np.sin(np.maximum(elevation_rad, floor_rad))

    def atmos_loss_db(self, elevation_rad):
        return self.zenith_atmos_loss_db * self.atmos_mapping_factor(elevation_rad)

    def antenna_noise_temp_k(self, elevation_rad):
        """Sky brightness seen by the dish, in kelvin.

        Cold sky attenuated by the atmosphere, plus the atmosphere's own
        emission, plus a fixed ground-spillover term.
        """
        loss_lin = from_db(self.atmos_loss_db(elevation_rad))
        sky = self.cosmic_background_k / loss_lin
        atmosphere = self.atmos_medium_temp_k * (1.0 - 1.0 / loss_lin)
        return sky + atmosphere + self.ground_spillover_temp_k

    def system_noise_temp_k(self, elevation_rad):
        """System noise temperature referred to the LNA input."""
        feed_lin = from_db(self.feed_loss_db)
        antenna = self.antenna_noise_temp_k(elevation_rad) / feed_lin
        feed = self.feed_physical_temp_k * (1.0 - 1.0 / feed_lin)
        return antenna + feed + self.lna_noise_temp_k

    def g_over_t_dbk(self, frequency_hz: float, elevation_rad):
        """Figure of merit G/T at the LNA input [dB/K]."""
        gain_at_lna = self.antenna_gain_dbi(frequency_hz) - self.feed_loss_db
        return gain_at_lna - to_db(self.system_noise_temp_k(elevation_rad))


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------


@dataclass
class LinkBudget:
    """A full space-to-ground downlink budget."""

    frequency_hz: float = 8.2e9
    transmitter: Transmitter = field(default_factory=Transmitter)
    station: GroundStation = field(default_factory=GroundStation)
    modcod: ModCod = field(default_factory=lambda: MODCODS["QPSK-LDPC1/2"])
    data_rate_bps: float = 50e6
    #: Losses that do not depend on geometry.
    polarization_loss_db: float = 0.3
    implementation_loss_db: float = 1.5
    rain_allowance_db: float = 1.0
    other_loss_db: float = 0.0
    #: Link margin the operator insists on before declaring the link usable.
    required_margin_db: float = 3.0

    # -- static terms -------------------------------------------------------

    @property
    def eirp_dbw(self) -> float:
        return self.transmitter.eirp_dbw

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_MPS / self.frequency_hz

    @property
    def threshold_ebn0_db(self) -> float:
        """Eb/N0 the demodulator needs, including implementation loss."""
        return self.modcod.required_ebn0_db + self.implementation_loss_db

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.data_rate_bps / self.modcod.spectral_efficiency_bps_per_hz

    # -- geometry-dependent terms ------------------------------------------

    def fspl_db(self, slant_range_m):
        """Free-space path loss, 20*log10(4*pi*d*f/c)."""
        slant_range_m = np.asarray(slant_range_m, dtype=float)
        return 20.0 * np.log10(
            4.0 * np.pi * slant_range_m * self.frequency_hz / SPEED_OF_LIGHT_MPS
        )

    def total_loss_db(self, slant_range_m, elevation_rad):
        """Every subtractive term between EIRP and the LNA input."""
        return (
            self.fspl_db(slant_range_m)
            + self.station.atmos_loss_db(elevation_rad)
            + self.rain_allowance_db
            + self.polarization_loss_db
            + self.station.pointing_loss_db
            + self.other_loss_db
        )

    def cn0_dbhz(self, slant_range_m, elevation_rad):
        """Carrier-to-noise-density ratio at the receiver [dB-Hz]."""
        return (
            self.eirp_dbw
            - self.total_loss_db(slant_range_m, elevation_rad)
            + self.station.g_over_t_dbk(self.frequency_hz, elevation_rad)
            + BOLTZMANN_DB
        )

    def ebn0_db(self, slant_range_m, elevation_rad, data_rate_bps=None):
        rate = self.data_rate_bps if data_rate_bps is None else data_rate_bps
        return self.cn0_dbhz(slant_range_m, elevation_rad) - to_db(rate)

    def margin_db(self, slant_range_m, elevation_rad, data_rate_bps=None):
        """Excess Eb/N0 over the demodulator threshold [dB]. Positive == closes."""
        return self.ebn0_db(slant_range_m, elevation_rad, data_rate_bps) - self.threshold_ebn0_db

    def closes(self, slant_range_m, elevation_rad, data_rate_bps=None):
        """Boolean: does the link meet threshold *and* the required margin?"""
        return (
            self.margin_db(slant_range_m, elevation_rad, data_rate_bps) >= self.required_margin_db
        )

    def max_data_rate_bps(self, slant_range_m, elevation_rad):
        """Highest rate this modcod supports while still holding the required margin."""
        available = (
            self.cn0_dbhz(slant_range_m, elevation_rad)
            - self.threshold_ebn0_db
            - self.required_margin_db
        )
        return from_db(available)

    def shannon_capacity_bps(self, slant_range_m, elevation_rad, bandwidth_hz=None):
        """Hard information-theoretic ceiling for the occupied bandwidth."""
        bandwidth = self.occupied_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        snr_db = self.cn0_dbhz(slant_range_m, elevation_rad) - to_db(bandwidth)
        return bandwidth * np.log2(1.0 + from_db(snr_db))

    def min_closing_elevation_rad(self, orbit_altitude_m, planet_radius_m=6378136.6,
                                  data_rate_bps=None, search_deg=(0.1, 90.0), points=4000):
        """Lowest elevation at which the link still closes for a circular orbit.

        Solved by a dense sweep rather than a root-finder because the margin is
        not monotonic in closed form once the noise-temperature term is included.
        Returns ``nan`` if the link never closes, even at zenith.
        """
        elevations = np.radians(np.linspace(search_deg[0], search_deg[1], points))
        ranges = slant_range_from_elevation(elevations, orbit_altitude_m, planet_radius_m)
        ok = self.margin_db(ranges, elevations, data_rate_bps) >= self.required_margin_db
        if not np.any(ok):
            return float("nan")
        return float(elevations[np.argmax(ok)])

    def breakdown(self, slant_range_m, elevation_rad, data_rate_bps=None):
        """Line-by-line budget at a single geometry, as an ordered dict.

        Useful for printing a conventional link-budget table.
        """
        rate = self.data_rate_bps if data_rate_bps is None else data_rate_bps
        elevation_rad = float(elevation_rad)
        slant_range_m = float(slant_range_m)
        max_rate = float(self.max_data_rate_bps(slant_range_m, elevation_rad))
        return {
            "Frequency [GHz]": self.frequency_hz / 1e9,
            "Slant range [km]": slant_range_m / 1e3,
            "Elevation [deg]": np.degrees(elevation_rad),
            "Tx power [dBW]": self.transmitter.power_dbw,
            "Tx antenna gain [dBi]": self.transmitter.antenna_gain_dbi,
            "Tx line loss [dB]": -self.transmitter.line_loss_db,
            "Tx pointing loss [dB]": -self.transmitter.pointing_loss_db,
            "EIRP [dBW]": self.eirp_dbw,
            "Free-space path loss [dB]": -float(self.fspl_db(slant_range_m)),
            "Atmospheric loss [dB]": -float(self.station.atmos_loss_db(elevation_rad)),
            "Rain allowance [dB]": -self.rain_allowance_db,
            "Polarization loss [dB]": -self.polarization_loss_db,
            "Rx pointing loss [dB]": -self.station.pointing_loss_db,
            "Other loss [dB]": -self.other_loss_db,
            "Rx antenna gain [dBi]": self.station.antenna_gain_dbi(self.frequency_hz),
            "Rx system noise temp [K]": float(self.station.system_noise_temp_k(elevation_rad)),
            "Rx G/T [dB/K]": float(self.station.g_over_t_dbk(self.frequency_hz, elevation_rad)),
            "Boltzmann term [dB]": BOLTZMANN_DB,
            "C/N0 [dB-Hz]": float(self.cn0_dbhz(slant_range_m, elevation_rad)),
            "Data rate [Mbps]": rate / 1e6,
            "Eb/N0 [dB]": float(self.ebn0_db(slant_range_m, elevation_rad, rate)),
            "Required Eb/N0 [dB]": self.threshold_ebn0_db,
            "Margin [dB]": float(self.margin_db(slant_range_m, elevation_rad, rate)),
            "Required margin [dB]": self.required_margin_db,
            "Occupied bandwidth [MHz]": self.occupied_bandwidth_hz / 1e6,
            "Shannon capacity in that BW [Mbps]": float(
                self.shannon_capacity_bps(slant_range_m, elevation_rad)
            )
            / 1e6,
            "Max supportable rate [Mbps]": max_rate / 1e6,
            "Bandwidth needed at max rate [MHz]": max_rate
            / self.modcod.spectral_efficiency_bps_per_hz
            / 1e6,
        }

    def degraded(self, **overrides) -> "LinkBudget":
        """Return a copy with fields overridden -- handy for sensitivity tests."""
        return replace(self, **overrides)


def slant_range_from_elevation(elevation_rad, altitude_m, planet_radius_m=6378136.6):
    """Slant range to a circular orbit seen at a given elevation.

    Spherical-earth law of cosines solved for the range:
    ``d = -Rs*sin(el) + sqrt((Rs*sin(el))^2 + h^2 + 2*Rs*h)`` with ``Rs`` the
    site radius and ``h`` the altitude above it.
    """
    elevation_rad = np.asarray(elevation_rad, dtype=float)
    site_radius = planet_radius_m
    orbit_radius = planet_radius_m + altitude_m
    sin_el = np.sin(elevation_rad)
    return -site_radius * sin_el + np.sqrt(
        (site_radius * sin_el) ** 2 + orbit_radius**2 - site_radius**2
    )


# ---------------------------------------------------------------------------
# Reference configurations
# ---------------------------------------------------------------------------


def x_band_leo_downlink() -> LinkBudget:
    """Nominal 50 Mbps X-band downlink from a 500 km LEO to a 5.4 m dish."""
    return LinkBudget(
        frequency_hz=8.2e9,
        transmitter=Transmitter(
            power_w=5.0, antenna_gain_dbi=6.0, line_loss_db=1.0, pointing_loss_db=1.0
        ),
        station=GroundStation(
            dish_diameter_m=5.4,
            aperture_efficiency=0.60,
            feed_loss_db=0.5,
            lna_noise_temp_k=75.0,
            zenith_atmos_loss_db=0.25,
        ),
        modcod=MODCODS["QPSK-LDPC1/2"],
        data_rate_bps=50e6,
        required_margin_db=3.0,
    )


def s_band_leo_downlink() -> LinkBudget:
    """Conservative 2 Mbps S-band cubesat downlink to a 3 m dish."""
    return LinkBudget(
        frequency_hz=2.2e9,
        transmitter=Transmitter(
            power_w=2.0, antenna_gain_dbi=2.0, line_loss_db=1.5, pointing_loss_db=2.0
        ),
        station=GroundStation(
            dish_diameter_m=3.0,
            aperture_efficiency=0.55,
            feed_loss_db=0.6,
            lna_noise_temp_k=100.0,
            zenith_atmos_loss_db=0.05,
        ),
        modcod=MODCODS["QPSK-conv1/2-RS"],
        data_rate_bps=2e6,
        required_margin_db=3.0,
    )
