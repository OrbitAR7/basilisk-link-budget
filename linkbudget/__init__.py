"""Link-budget test harness for the Basilisk astrodynamics framework.

Layers, bottom up:

* :mod:`linkbudget.rf`        -- RF link budget math, pure numpy.
* :mod:`linkbudget.geometry`  -- independent orbit/access truth model, pure numpy.
* :mod:`linkbudget.scenario`  -- the Basilisk simulation itself.
* :mod:`linkbudget.analysis`  -- couples the budget to Basilisk telemetry.

Only :mod:`linkbudget.scenario` needs Basilisk, and it imports it lazily so the
rest of the package works on a machine where the simulator has not been built.
"""

from .rf import (
    MODCODS,
    GroundStation,
    LinkBudget,
    ModCod,
    Transmitter,
    s_band_leo_downlink,
    x_band_leo_downlink,
)

__all__ = [
    "LinkBudget",
    "Transmitter",
    "GroundStation",
    "ModCod",
    "MODCODS",
    "x_band_leo_downlink",
    "s_band_leo_downlink",
]
