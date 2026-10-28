"""Pytest fixtures shared by the whole suite.

Lives at the project root (rather than in ``tests/``) so that pytest prepends
the root to ``sys.path`` and ``import linkbudget`` resolves without an install.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkbudget import analysis, rf, scenario

#: Skip decorator for everything that needs a built simulator.
requires_basilisk = pytest.mark.skipif(
    not scenario.HAVE_BASILISK,
    reason=f"Basilisk is not importable: {scenario.BASILISK_IMPORT_ERROR!r}",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a full Basilisk simulation")


@pytest.fixture(scope="session")
def comm_config() -> scenario.CommScenarioConfig:
    """Reference scenario: 500 km SSO, CU Boulder station, X-band at 50 Mbps.

    Sampling is deliberately locked to the integration step so that data-volume
    conservation can be checked without interpolation error.
    """
    return scenario.CommScenarioConfig(
        duration_s=12 * 3600.0,
        time_step_s=1.0,
        sample_step_s=1.0,
        min_elevation_deg=10.0,
        downlink_baud_bps=50e6,
        instrument_baud_bps=1e6,
        packet_size_bits=1e6,
        initial_buffer_bits=10e9,
        storage_capacity_bits=100e9,
    )


@pytest.fixture(scope="session")
def budget() -> rf.LinkBudget:
    """Nominal X-band budget matching ``comm_config``'s downlink rate."""
    return rf.x_band_leo_downlink()


@pytest.fixture(scope="session")
def bsk_telemetry(comm_config):
    """One Basilisk run, shared by every simulator test in the session."""
    if not scenario.HAVE_BASILISK:
        pytest.skip(f"Basilisk is not importable: {scenario.BASILISK_IMPORT_ERROR!r}")
    return scenario.run_scenario(comm_config)


@pytest.fixture(scope="session")
def truth_telemetry(comm_config):
    """The analytic reference run for the same configuration."""
    return scenario.run_synthetic_scenario(comm_config)


@pytest.fixture(scope="session")
def bsk_series(bsk_telemetry, budget):
    return analysis.evaluate_link(bsk_telemetry, budget)


@pytest.fixture(scope="session")
def bsk_passes(bsk_series, comm_config):
    return analysis.summarize_passes(bsk_series, comm_config.sample_step_s)


@pytest.fixture(scope="session")
def access_mask(bsk_telemetry):
    """Boolean mask of samples where the ground station reports access."""
    return np.asarray(bsk_telemetry.has_access).astype(bool)
