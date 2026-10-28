"""Basilisk communications scenario: LEO spacecraft downlinking to a ground station.

Wires up the full Basilisk comm chain --

    spacecraft ---> groundLocation ---(AccessMsg)---> spaceToGroundTransmitter
    simpleInstrument ---(DataNodeUsageMsg)---> storage unit <---(downlink)

-- runs it, and returns the telemetry as plain numpy arrays so that the
link-budget layer and the tests never have to touch Basilisk objects.

Earth rotation is supplied by a small Python ``SysModel`` that writes the
``SpicePlanetStateMsg`` itself, rather than by the SPICE interface.  That keeps
the run free of ephemeris kernels and, more importantly, makes the planet
attitude an exact closed-form function of time that ``geometry.py`` can
reproduce independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry

BASILISK_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - import guard
    from Basilisk.architecture import messaging, sysModel
    from Basilisk.simulation import (
        groundLocation,
        simpleInstrument,
        simpleStorageUnit,
        spaceToGroundTransmitter,
        spacecraft,
    )
    from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody

    HAVE_BASILISK = True
except Exception as exc:  # pragma: no cover - import guard
    HAVE_BASILISK = False
    BASILISK_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CommScenarioConfig:
    """Everything that defines the simulation run."""

    # Orbit -- a 500 km sun-synchronous-ish LEO.
    altitude_m: float = 500e3
    eccentricity: float = 0.0
    inclination_deg: float = 97.4
    #: Chosen so a 12 h run yields three Boulder passes of differing geometry
    #: (~180 s low, ~300 s medium, ~380 s near-overhead).
    raan_deg: float = 60.0
    arg_periapsis_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    # Ground station -- CU Boulder / LASP.
    station_lat_deg: float = 40.009971
    station_lon_deg: float = -105.243895
    station_alt_m: float = 1624.0
    min_elevation_deg: float = 10.0
    max_range_m: float = -1.0  # negative == unlimited

    # Planet
    planet_radius_m: float = geometry.EARTH_RADIUS_M
    earth_rotation: bool = True
    earth_omega_radps: float = geometry.EARTH_OMEGA_RADPS
    earth_theta0_rad: float = 0.0

    # Timing
    duration_s: float = 12 * 3600.0
    time_step_s: float = 1.0
    sample_step_s: float = 5.0

    # Data system.  The instrument rate is sized so that 12 h of collection
    # roughly matches 12 h of downlink opportunity (~855 s of access at
    # 50 Mbps), keeping the buffer off both of its clamps for the whole run.
    instrument_baud_bps: float = 1e6
    storage_capacity_bits: float = 100e9
    #: Downlink rate; stored positive here, handed to Basilisk negated because
    #: the transmitter is a data *sink*.
    downlink_baud_bps: float = 50e6
    #: Minimum data that must be buffered before a packet starts, in bits.
    packet_size_bits: float = 1e6
    #: Pre-load the buffer so even the earliest pass can run at full rate for
    #: its whole duration; otherwise the transmitter idles once the buffer
    #: empties and the reported volume becomes buffer-limited, not link-limited.
    initial_buffer_bits: float = 10e9

    def classical_elements(self) -> geometry.ClassicalElements:
        """The same orbit expressed for the independent truth model."""
        return geometry.ClassicalElements(
            a=self.planet_radius_m + self.altitude_m,
            e=self.eccentricity,
            i=np.radians(self.inclination_deg),
            raan=np.radians(self.raan_deg),
            arg_periapsis=np.radians(self.arg_periapsis_deg),
            true_anomaly=np.radians(self.true_anomaly_deg),
        )


@dataclass
class CommTelemetry:
    """Recorded simulation output, all as numpy arrays of equal length."""

    time_s: np.ndarray
    r_BN_N: np.ndarray
    v_BN_N: np.ndarray
    r_LN_N: np.ndarray
    has_access: np.ndarray
    slant_range_m: np.ndarray
    elevation_rad: np.ndarray
    azimuth_rad: np.ndarray
    range_rate_mps: np.ndarray
    tx_baud_bps: np.ndarray
    instrument_baud_bps: np.ndarray
    stored_bits: np.ndarray
    net_baud_bps: np.ndarray
    config: CommScenarioConfig = field(default=None)
    source: str = "basilisk"

    def __len__(self) -> int:
        return int(self.time_s.size)


# ---------------------------------------------------------------------------
# Rotating-Earth planet-state module
# ---------------------------------------------------------------------------


def _make_rotating_planet_module(omega_radps: float, theta0_rad: float):
    """A Python SysModel that publishes ``SpicePlanetStateMsg`` each step."""

    class RotatingPlanetState(sysModel.SysModel):
        def __init__(self):
            super().__init__()
            self.ModelTag = "rotatingEarth"
            self.omega_radps = omega_radps
            self.theta0_rad = theta0_rad
            self.planetStateOutMsg = messaging.SpicePlanetStateMsg()

        # Reset runs during InitializeSimulation, and this module holds the
        # highest task priority, so the message is always written before
        # groundLocation reads it -- including at t = 0.
        def Reset(self, CurrentSimNanos):
            self._write(CurrentSimNanos)

        def UpdateState(self, CurrentSimNanos):
            self._write(CurrentSimNanos)

        def _write(self, current_sim_nanos):
            seconds = current_sim_nanos * 1.0e-9
            theta = self.theta0_rad + self.omega_radps * seconds
            payload = messaging.SpicePlanetStateMsgPayload()
            payload.PositionVector = [0.0, 0.0, 0.0]
            payload.VelocityVector = [0.0, 0.0, 0.0]
            payload.J20002Pfix = geometry.planet_dcm(theta).tolist()
            payload.J20002Pfix_dot = geometry.planet_dcm_dot(theta, self.omega_radps).tolist()
            self.planetStateOutMsg.write(payload, int(current_sim_nanos), self.moduleID)

    return RotatingPlanetState()


def _make_static_planet_message():
    """Non-rotating Earth: identity attitude, written once."""
    payload = messaging.SpicePlanetStateMsgPayload()
    payload.PositionVector = [0.0, 0.0, 0.0]
    payload.VelocityVector = [0.0, 0.0, 0.0]
    payload.J20002Pfix = np.eye(3).tolist()
    payload.J20002Pfix_dot = np.zeros((3, 3)).tolist()
    return messaging.SpicePlanetStateMsg().write(payload)


def _attach_gravity(grav_factory, sc_object):
    """Bridge the two Basilisk APIs for attaching gravity bodies."""
    if hasattr(grav_factory, "addBodiesTo"):
        grav_factory.addBodiesTo(sc_object)
    else:  # older Basilisk
        sc_object.gravField.gravBodies = spacecraft.GravBodyVector(
            list(grav_factory.gravBodies.values())
        )


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def run_scenario(cfg: CommScenarioConfig | None = None) -> CommTelemetry:
    """Build and execute the Basilisk comm scenario, returning its telemetry."""
    if not HAVE_BASILISK:
        raise ImportError(
            "Basilisk is not importable, so the simulation cannot run. "
            "Build it from https://github.com/AVSLab/basilisk, or use "
            "run_synthetic_scenario() for the analytic reference model. "
            f"Original import error: {BASILISK_IMPORT_ERROR!r}"
        )

    cfg = cfg or CommScenarioConfig()

    sim = SimulationBaseClass.SimBaseClass()
    process_name, task_name = "commProcess", "commTask"
    process = sim.CreateNewProcess(process_name)
    process.addTask(sim.CreateNewTask(task_name, macros.sec2nano(cfg.time_step_s)))

    # -- spacecraft and gravity --------------------------------------------
    sc_object = spacecraft.Spacecraft()
    sc_object.ModelTag = "downlinkSat"

    grav_factory = simIncludeGravBody.gravBodyFactory()
    earth = grav_factory.createEarth()
    earth.isCentralBody = True
    _attach_gravity(grav_factory, sc_object)

    oe = orbitalMotion.ClassicElements()
    oe.a = cfg.planet_radius_m + cfg.altitude_m
    oe.e = cfg.eccentricity
    oe.i = np.radians(cfg.inclination_deg)
    oe.Omega = np.radians(cfg.raan_deg)
    oe.omega = np.radians(cfg.arg_periapsis_deg)
    oe.f = np.radians(cfg.true_anomaly_deg)
    r_init, v_init = orbitalMotion.elem2rv(earth.mu, oe)
    sc_object.hub.r_CN_NInit = r_init
    sc_object.hub.v_CN_NInit = v_init

    # -- planet attitude ----------------------------------------------------
    planet_module = None
    if cfg.earth_rotation:
        planet_module = _make_rotating_planet_module(cfg.earth_omega_radps, cfg.earth_theta0_rad)
        sim.AddModelToTask(task_name, planet_module, 200)
        planet_msg = planet_module.planetStateOutMsg
    else:
        planet_msg = _make_static_planet_message()

    sim.AddModelToTask(task_name, sc_object, 150)

    # -- ground station -----------------------------------------------------
    station = groundLocation.GroundLocation()
    station.ModelTag = "boulderStation"
    station.planetRadius = cfg.planet_radius_m
    station.minimumElevation = np.radians(cfg.min_elevation_deg)
    station.maximumRange = cfg.max_range_m
    station.specifyLocation(
        np.radians(cfg.station_lat_deg), np.radians(cfg.station_lon_deg), cfg.station_alt_m
    )
    station.planetInMsg.subscribeTo(planet_msg)
    station.addSpacecraftToModel(sc_object.scStateOutMsg)
    sim.AddModelToTask(task_name, station, 120)

    # -- payload instrument -------------------------------------------------
    instrument = simpleInstrument.SimpleInstrument()
    instrument.ModelTag = "imager"
    instrument.nodeBaudRate = cfg.instrument_baud_bps
    instrument.nodeDataName = "Imager"
    sim.AddModelToTask(task_name, instrument, 100)

    # -- transmitter --------------------------------------------------------
    # Basilisk expects a data *sink*, hence the negated rate and packet size.
    transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
    transmitter.ModelTag = "xBandTransmitter"
    transmitter.nodeBaudRate = -abs(cfg.downlink_baud_bps)
    transmitter.packetSize = -abs(cfg.packet_size_bits)
    transmitter.numBuffers = 1
    transmitter.dataStatus = 1
    transmitter.addAccessMsgToTransmitter(station.accessOutMsgs[0])
    sim.AddModelToTask(task_name, transmitter, 90)

    # -- storage ------------------------------------------------------------
    storage = simpleStorageUnit.SimpleStorageUnit()
    storage.ModelTag = "solidStateRecorder"
    storage.storageCapacity = int(cfg.storage_capacity_bits)
    storage.addDataNodeToModel(instrument.nodeDataOutMsg)
    storage.addDataNodeToModel(transmitter.nodeDataOutMsg)
    sim.AddModelToTask(task_name, storage, 80)

    transmitter.addStorageUnitToTransmitter(storage.storageUnitDataOutMsg)

    # -- recorders ----------------------------------------------------------
    sampling = macros.sec2nano(cfg.sample_step_s)
    sc_log = sc_object.scStateOutMsg.recorder(sampling)
    access_log = station.accessOutMsgs[0].recorder(sampling)
    ground_log = station.currentGroundStateOutMsg.recorder(sampling)
    tx_log = transmitter.nodeDataOutMsg.recorder(sampling)
    inst_log = instrument.nodeDataOutMsg.recorder(sampling)
    storage_log = storage.storageUnitDataOutMsg.recorder(sampling)
    for recorder in (sc_log, access_log, ground_log, tx_log, inst_log, storage_log):
        sim.AddModelToTask(task_name, recorder, 10)

    # -- run ----------------------------------------------------------------
    # SimpleStorageUnit lumps everything into a single "STORED DATA" partition,
    # so its setDataBuffer takes only an amount.  Seed it before initialisation
    # so the partition exists on the first integration step.
    if cfg.initial_buffer_bits > 0:
        storage.setDataBuffer(int(cfg.initial_buffer_bits))

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(cfg.duration_s))
    sim.ExecuteSimulation()

    return CommTelemetry(
        time_s=np.asarray(sc_log.times()) * macros.NANO2SEC,
        r_BN_N=np.asarray(sc_log.r_BN_N),
        v_BN_N=np.asarray(sc_log.v_BN_N),
        r_LN_N=np.asarray(ground_log.r_LN_N),
        has_access=np.asarray(access_log.hasAccess).astype(int),
        slant_range_m=np.asarray(access_log.slantRange, dtype=float),
        elevation_rad=np.asarray(access_log.elevation, dtype=float),
        azimuth_rad=np.asarray(access_log.azimuth, dtype=float),
        range_rate_mps=np.asarray(access_log.range_dot, dtype=float),
        tx_baud_bps=np.asarray(tx_log.baudRate, dtype=float),
        instrument_baud_bps=np.asarray(inst_log.baudRate, dtype=float),
        stored_bits=np.asarray(storage_log.storageLevel, dtype=float),
        net_baud_bps=np.asarray(storage_log.currentNetBaud, dtype=float),
        config=cfg,
        source="basilisk",
    )


def run_synthetic_scenario(cfg: CommScenarioConfig | None = None) -> CommTelemetry:
    """Analytic stand-in for :func:`run_scenario` that needs no Basilisk.

    Reproduces the access geometry exactly and models the data system with the
    same rules Basilisk's modules implement (downlink only during access, only
    once a full packet is buffered, never below empty).  Useful for exercising
    the link-budget pipeline before the simulator has been built, and as a
    cross-check afterwards.
    """
    cfg = cfg or CommScenarioConfig()
    omega = cfg.earth_omega_radps if cfg.earth_rotation else 0.0

    truth = geometry.synthetic_access_telemetry(
        cfg.classical_elements(),
        cfg.station_lat_deg,
        cfg.station_lon_deg,
        cfg.station_alt_m,
        cfg.duration_s,
        cfg.sample_step_s,
        cfg.min_elevation_deg,
        cfg.max_range_m,
        cfg.earth_theta0_rad,
        cfg.planet_radius_m,
        omega,
    )

    times = truth["time_s"]
    access = truth["has_access"].astype(bool)
    dt = cfg.sample_step_s

    stored = np.zeros_like(times)
    tx_baud = np.zeros_like(times)
    level = float(cfg.initial_buffer_bits)
    for idx in range(times.size):
        stored[idx] = level
        rate = 0.0
        if access[idx] and level >= cfg.packet_size_bits:
            rate = -min(cfg.downlink_baud_bps, level / dt)
        tx_baud[idx] = rate
        level = max(0.0, level + (cfg.instrument_baud_bps + rate) * dt)
        level = min(level, cfg.storage_capacity_bits)

    return CommTelemetry(
        time_s=times,
        r_BN_N=truth["r_BN_N"],
        v_BN_N=truth["v_BN_N"],
        r_LN_N=truth["r_LN_N"],
        has_access=truth["has_access"],
        slant_range_m=truth["slant_range_m"],
        elevation_rad=truth["elevation_rad"],
        azimuth_rad=np.zeros_like(times),
        range_rate_mps=np.gradient(truth["slant_range_m"], times),
        tx_baud_bps=tx_baud,
        instrument_baud_bps=np.full_like(times, cfg.instrument_baud_bps),
        stored_bits=stored,
        net_baud_bps=np.full_like(times, cfg.instrument_baud_bps) + tx_baud,
        config=cfg,
        source="synthetic",
    )
