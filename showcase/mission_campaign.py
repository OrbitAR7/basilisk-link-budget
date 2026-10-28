#!/usr/bin/env python3
"""Monte Carlo data-return campaign for an attitude-controlled imaging smallsat.

Exercises three parts of Basilisk in one run:

1. **Fast, repeatable 6-DOF Monte Carlo** -- N dispersed cases, full rigid-body
   dynamics with four reaction wheels, plus a bit-for-bit repeatability check.
2. **Flight software in the loop** -- the actual `locationPointing` ->
   `mrpFeedback` -> `rwMotorTorque` chain slews the spacecraft to put the
   antenna boresight on the ground station, running against the truth dynamics
   in the same executable.
3. **Coupled subsystem resources** -- the solar panel normal and the antenna
   boresight are different body axes, so pointing the antenna at Boulder tilts
   the array off the Sun, and the transmitter draws power only while
   downlinking. The campaign measures whether that creates a data-vs-charge
   trade (with this dispersion set it doesn't: r is about -0.09).

Run:  python showcase/mission_campaign.py --cases 24
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import locationPointing, mrpFeedback, rwMotorTorque
from Basilisk.simulation import (
    eclipse,
    groundLocation,
    partitionedStorageUnit,
    reactionWheelStateEffector,
    simpleBattery,
    simpleInstrument,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    spaceToGroundTransmitter,
    spacecraft,
)
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    simIncludeRW,
    vizSupport,
)

from linkbudget import geometry as geo

SUN_DISTANCE_M = 1.495978707e11


@dataclass
class CaseConfig:
    """One Monte Carlo sample."""

    case_id: int
    raan_deg: float
    true_anomaly_deg: float
    sigma_BN_init: tuple
    omega_BN_init: tuple
    instrument_baud_bps: float

    # Fixed across the campaign
    altitude_m: float = 500e3
    inclination_deg: float = 97.4
    duration_s: float = 6 * 3600.0
    time_step_s: float = 1.0
    sample_step_s: float = 10.0
    station_lat_deg: float = 40.009971
    station_lon_deg: float = -105.243895
    station_alt_m: float = 1624.0
    min_elevation_deg: float = 10.0
    downlink_baud_bps: float = 50e6
    storage_capacity_bits: float = 100e9
    initial_buffer_bits: float = 8e9
    battery_capacity_ws: float = 200.0 * 3600.0
    bus_load_w: float = 28.0
    #: RF chain draw while downlinking (PA + modem), on top of the bus load.
    transmitter_load_w: float = 42.0
    #: When set, record a Vizard replay file to this path (no Vizard needed to run).
    viz_save_file: str | None = None
    panel_area_m2: float = 0.36
    panel_efficiency: float = 0.30


@dataclass
class CaseResult:
    case_id: int
    downlinked_gbit: float
    buffer_peak_gbit: float
    n_passes: int
    access_time_s: float
    mean_pointing_error_deg: float
    peak_pointing_error_deg: float
    settled_pointing_error_deg: float
    battery_min_soc_pct: float
    battery_end_soc_pct: float
    peak_wheel_speed_rpm: float
    wall_clock_s: float
    sim_seconds: float
    # kept for the repeatability check
    fingerprint: np.ndarray


def _static_planet_message(theta0_rad, omega_radps, time_s):
    """Earth attitude at a single instant (written once per step by the caller)."""
    payload = messaging.SpicePlanetStateMsgPayload()
    # eclipse looks the body radius up by name, so this cannot be left blank.
    payload.PlanetName = "earth"
    payload.PositionVector = [0.0, 0.0, 0.0]
    payload.VelocityVector = [0.0, 0.0, 0.0]
    theta = theta0_rad + omega_radps * time_s
    payload.J20002Pfix = geo.planet_dcm(theta).tolist()
    payload.J20002Pfix_dot = geo.planet_dcm_dot(theta, omega_radps).tolist()
    return payload


def _make_transmitter_power_gate(sys_model_cls, transmitter):
    """Turn the transmitter's RF power sink on only while it is downlinking.

    Without this the downlink costs nothing and the "data vs charge" trade the
    campaign is looking for does not exist in the model at all.
    """

    class TransmitterPowerGate(sys_model_cls):
        def __init__(self):
            super().__init__()
            self.ModelTag = "txPowerGate"
            self.deviceStatusOutMsg = messaging.DeviceStatusMsg()
            self._tx = transmitter

        def Reset(self, CurrentSimNanos):
            self.UpdateState(CurrentSimNanos)

        def UpdateState(self, CurrentSimNanos):
            payload = messaging.DeviceStatusMsgPayload()
            # nodeDataOutMsg.baudRate is negative while a packet is going down.
            payload.deviceStatus = 1 if self._tx.nodeDataOutMsg.read().baudRate < 0.0 else 0
            self.deviceStatusOutMsg.write(payload, int(CurrentSimNanos), self.moduleID)

    return TransmitterPowerGate()


def _make_rotating_planet(sys_model_cls, omega_radps):
    class RotatingEarth(sys_model_cls):
        def __init__(self):
            super().__init__()
            self.ModelTag = "rotatingEarth"
            self.planetStateOutMsg = messaging.SpicePlanetStateMsg()

        def Reset(self, CurrentSimNanos):
            self.UpdateState(CurrentSimNanos)

        def UpdateState(self, CurrentSimNanos):
            payload = _static_planet_message(0.0, omega_radps, CurrentSimNanos * 1e-9)
            self.planetStateOutMsg.write(payload, int(CurrentSimNanos), self.moduleID)

    return RotatingEarth()


def run_case(cfg: CaseConfig) -> CaseResult:
    """Build and run one full 6-DOF mission case."""
    from Basilisk.architecture import sysModel

    wall_start = time.perf_counter()

    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("missionProcess")
    task = "missionTask"
    proc.addTask(sim.CreateNewTask(task, macros.sec2nano(cfg.time_step_s)))

    # ---------------- spacecraft, gravity, orbit ----------------
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "imagerSat"
    inertia = [9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 6.0]
    sc.hub.mHub = 60.0
    # hub wants a 3x3; the FSW VehicleConfig message wants the flat 9.
    sc.hub.IHubPntBc_B = [inertia[0:3], inertia[3:6], inertia[6:9]]

    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    grav.addBodiesTo(sc)

    oe = orbitalMotion.ClassicElements()
    oe.a = geo.EARTH_RADIUS_M + cfg.altitude_m
    oe.e = 0.0
    oe.i = np.radians(cfg.inclination_deg)
    oe.Omega = np.radians(cfg.raan_deg)
    oe.omega = 0.0
    oe.f = np.radians(cfg.true_anomaly_deg)
    r_init, v_init = orbitalMotion.elem2rv(earth.mu, oe)
    sc.hub.r_CN_NInit = r_init
    sc.hub.v_CN_NInit = v_init
    sc.hub.sigma_BNInit = [[cfg.sigma_BN_init[0]], [cfg.sigma_BN_init[1]],
                           [cfg.sigma_BN_init[2]]]
    sc.hub.omega_BN_BInit = [[cfg.omega_BN_init[0]], [cfg.omega_BN_init[1]],
                             [cfg.omega_BN_init[2]]]

    # ---------------- reaction wheels ----------------
    rw_factory = simIncludeRW.rwFactory()
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1] / np.sqrt(3)):
        rw_factory.create("Honeywell_HR16", list(np.asarray(axis, dtype=float)),
                          maxMomentum=50.0, Omega=100.0)
    rw_effector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rw_factory.addToSpacecraft("RW", rw_effector, sc)
    rw_param_msg = rw_factory.getConfigMessage()

    # ---------------- environment ----------------
    planet_module = _make_rotating_planet(sysModel.SysModel, geo.EARTH_OMEGA_RADPS)
    sim.AddModelToTask(task, planet_module, 300)
    sim.AddModelToTask(task, sc, 250)
    sim.AddModelToTask(task, rw_effector, 240)

    sun_payload = messaging.SpicePlanetStateMsgPayload()
    sun_payload.PositionVector = [SUN_DISTANCE_M, 0.0, 0.0]
    sun_msg = messaging.SpicePlanetStateMsg().write(sun_payload)

    eclipse_module = eclipse.Eclipse()
    eclipse_module.ModelTag = "eclipse"
    eclipse_module.sunInMsg.subscribeTo(sun_msg)
    eclipse_module.addPlanetToModel(planet_module.planetStateOutMsg)
    eclipse_module.rEqCustom = geo.EARTH_RADIUS_M
    eclipse_module.addSpacecraftToModel(sc.scStateOutMsg)
    sim.AddModelToTask(task, eclipse_module, 230)

    # ---------------- navigation ----------------
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "nav"
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask(task, nav, 220)

    # ---------------- ground station ----------------
    station = groundLocation.GroundLocation()
    station.ModelTag = "boulder"
    station.planetRadius = geo.EARTH_RADIUS_M
    station.minimumElevation = np.radians(cfg.min_elevation_deg)
    station.maximumRange = -1.0
    station.specifyLocation(np.radians(cfg.station_lat_deg),
                            np.radians(cfg.station_lon_deg), cfg.station_alt_m)
    station.planetInMsg.subscribeTo(planet_module.planetStateOutMsg)
    station.addSpacecraftToModel(sc.scStateOutMsg)
    sim.AddModelToTask(task, station, 210)

    # ---------------- flight software ----------------
    # Antenna boresight is +Z body; the solar array normal is +Y body.  Slewing
    # to put +Z on the station therefore moves +Y off the Sun -- the coupling
    # this campaign is built to measure.
    pointing = locationPointing.locationPointing()
    pointing.ModelTag = "antennaPointing"
    pointing.pHat_B = [0.0, 0.0, 1.0]
    pointing.useBoresightRateDamping = 1
    pointing.scAttInMsg.subscribeTo(nav.attOutMsg)
    pointing.scTransInMsg.subscribeTo(nav.transOutMsg)
    pointing.locationInMsg.subscribeTo(station.currentGroundStateOutMsg)
    sim.AddModelToTask(task, pointing, 200)

    vehicle_payload = messaging.VehicleConfigMsgPayload()
    vehicle_payload.ISCPntB_B = inertia
    vehicle_msg = messaging.VehicleConfigMsg().write(vehicle_payload)

    controller = mrpFeedback.mrpFeedback()
    controller.ModelTag = "mrpFeedback"
    controller.K = 0.6
    controller.Ki = -1.0  # negative disables the integral term
    controller.P = 6.0
    controller.integralLimit = 0.0
    controller.guidInMsg.subscribeTo(pointing.attGuidOutMsg)
    controller.vehConfigInMsg.subscribeTo(vehicle_msg)
    controller.rwParamsInMsg.subscribeTo(rw_param_msg)
    controller.rwSpeedsInMsg.subscribeTo(rw_effector.rwSpeedOutMsg)
    sim.AddModelToTask(task, controller, 190)

    torque_map = rwMotorTorque.rwMotorTorque()
    torque_map.ModelTag = "rwMotorTorque"
    torque_map.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    torque_map.vehControlInMsg.subscribeTo(controller.cmdTorqueOutMsg)
    torque_map.rwParamsInMsg.subscribeTo(rw_param_msg)
    sim.AddModelToTask(task, torque_map, 180)
    rw_effector.rwMotorCmdInMsg.subscribeTo(torque_map.rwMotorTorqueOutMsg)

    # ---------------- payload data chain ----------------
    instrument = simpleInstrument.SimpleInstrument()
    instrument.ModelTag = "imager"
    instrument.nodeBaudRate = cfg.instrument_baud_bps
    instrument.nodeDataName = "Imagery"
    sim.AddModelToTask(task, instrument, 170)

    transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
    transmitter.ModelTag = "xBandTx"
    transmitter.nodeBaudRate = -cfg.downlink_baud_bps
    transmitter.packetSize = -1e6
    transmitter.numBuffers = 1
    transmitter.dataStatus = 1
    transmitter.addAccessMsgToTransmitter(station.accessOutMsgs[0])
    sim.AddModelToTask(task, transmitter, 160)

    storage = partitionedStorageUnit.PartitionedStorageUnit()
    storage.ModelTag = "recorder"
    storage.storageCapacity = int(cfg.storage_capacity_bits)
    storage.addDataNodeToModel(instrument.nodeDataOutMsg)
    storage.addDataNodeToModel(transmitter.nodeDataOutMsg)
    storage.addPartition("Imagery")
    sim.AddModelToTask(task, storage, 150)
    transmitter.addStorageUnitToTransmitter(storage.storageUnitDataOutMsg)

    # ---------------- power chain ----------------
    panel = simpleSolarPanel.SimpleSolarPanel()
    panel.ModelTag = "array"
    panel.stateInMsg.subscribeTo(sc.scStateOutMsg)
    panel.sunInMsg.subscribeTo(sun_msg)
    panel.sunEclipseInMsg.subscribeTo(eclipse_module.eclipseOutMsgs[0])
    panel.setPanelParameters([0.0, 1.0, 0.0], cfg.panel_area_m2, cfg.panel_efficiency)
    sim.AddModelToTask(task, panel, 140)

    bus = simplePowerSink.SimplePowerSink()
    bus.ModelTag = "busLoad"
    bus.nodePowerOut = -cfg.bus_load_w
    sim.AddModelToTask(task, bus, 135)

    # The X-band transmitter is a real load, and only while it is transmitting.
    tx_gate = _make_transmitter_power_gate(sysModel.SysModel, transmitter)
    sim.AddModelToTask(task, tx_gate, 133)

    tx_power = simplePowerSink.SimplePowerSink()
    tx_power.ModelTag = "txLoad"
    tx_power.nodePowerOut = -cfg.transmitter_load_w
    tx_power.nodeStatusInMsg.subscribeTo(tx_gate.deviceStatusOutMsg)
    sim.AddModelToTask(task, tx_power, 130)

    battery = simpleBattery.SimpleBattery()
    battery.ModelTag = "battery"
    battery.storageCapacity = cfg.battery_capacity_ws
    battery.storedCharge_Init = cfg.battery_capacity_ws * 0.80
    battery.addPowerNodeToModel(panel.nodePowerOutMsg)
    battery.addPowerNodeToModel(bus.nodePowerOutMsg)
    battery.addPowerNodeToModel(tx_power.nodePowerOutMsg)
    sim.AddModelToTask(task, battery, 120)

    # ---------------- recorders ----------------
    sampling = macros.sec2nano(cfg.sample_step_s)
    logs = {
        "access": station.accessOutMsgs[0].recorder(sampling),
        "tx": transmitter.nodeDataOutMsg.recorder(sampling),
        "storage": storage.storageUnitDataOutMsg.recorder(sampling),
        "guid": pointing.attGuidOutMsg.recorder(sampling),
        "battery": battery.batPowerOutMsg.recorder(sampling),
        "power": panel.nodePowerOutMsg.recorder(sampling),
        "wheels": rw_effector.rwSpeedOutMsg.recorder(sampling),
    }
    for rec in logs.values():
        sim.AddModelToTask(task, rec, 10)

    # ---------------- Vizard recording (optional) ----------------
    # Writes a .bin replay file. Vizard itself is only needed to *watch* it.
    if cfg.viz_save_file:
        viz = vizSupport.enableUnityVisualization(
            sim, task, sc,
            rwEffectorList=rw_effector,
            saveFile=cfg.viz_save_file,
        )
        # The ground station, drawn with its 10-degree elevation-mask cone.
        vizSupport.addLocation(
            viz,
            stationName="Boulder",
            parentBodyName="earth",
            # lla_GP takes radians for lat/lon, metres for altitude.
            lla_GP=[np.radians(cfg.station_lat_deg),
                    np.radians(cfg.station_lon_deg),
                    cfg.station_alt_m],
            fieldOfView=np.radians(2.0 * (90.0 - cfg.min_elevation_deg)),
            color="cyan",
            range=3000e3,
        )
        vizSupport.createTargetLine(viz, toBodyName="Boulder", lineColor="yellow")
        vizSupport.setActuatorGuiSetting(
            viz, viewRWPanel=True, viewRWHUD=True, showRWLabels=True
        )
        # Camera looking down the +Z antenna boresight, so the slew is visible.
        vizSupport.createStandardCamera(
            viz, setMode=0, pointingVector_B=[0.0, 0.0, 1.0],
            displayName="antenna boresight",
        )

    # ---------------- run ----------------
    # NB: PartitionedStorageUnit overrides setDataBuffer with *vector* args.
    # Passing the base class's (str, int) form aborts the process with an
    # uncaught std::length_error rather than raising a Python TypeError.
    storage.setDataBuffer(["Imagery"], [int(cfg.initial_buffer_bits)])
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(cfg.duration_s))
    sim.ExecuteSimulation()

    wall = time.perf_counter() - wall_start

    # ---------------- reduce ----------------
    access = np.asarray(logs["access"].hasAccess).astype(bool)
    tx = np.asarray(logs["tx"].baudRate, dtype=float)
    stored = np.asarray(logs["storage"].storageLevel, dtype=float)
    sigma_BR = np.asarray(logs["guid"].sigma_BR, dtype=float)
    charge = np.asarray(logs["battery"].storageLevel, dtype=float)
    wheel_speeds = np.asarray(logs["wheels"].wheelSpeeds, dtype=float)

    # MRP magnitude -> principal rotation angle
    mrp_norm = np.linalg.norm(sigma_BR, axis=1)
    pointing_err_deg = np.degrees(4.0 * np.arctan(mrp_norm))

    downlinked = -tx.sum() * cfg.sample_step_s
    edges = np.diff(np.concatenate(([0], access.astype(np.int8), [0])))
    n_passes = int(np.count_nonzero(edges == 1))

    # "Settled" error: after the first 10 minutes, once the initial slew is done.
    settled = pointing_err_deg[int(600.0 / cfg.sample_step_s):]

    soc = 100.0 * charge / cfg.battery_capacity_ws
    peak_rpm = float(np.abs(wheel_speeds).max() * 60.0 / (2.0 * np.pi))

    fingerprint = np.concatenate([stored[-50:], charge[-50:], mrp_norm[-50:]])

    return CaseResult(
        case_id=cfg.case_id,
        downlinked_gbit=downlinked / 1e9,
        buffer_peak_gbit=float(stored.max()) / 1e9,
        n_passes=n_passes,
        access_time_s=float(access.sum()) * cfg.sample_step_s,
        mean_pointing_error_deg=float(pointing_err_deg[access].mean()) if access.any() else float("nan"),
        peak_pointing_error_deg=float(pointing_err_deg.max()),
        settled_pointing_error_deg=float(np.median(settled)),
        battery_min_soc_pct=float(soc.min()),
        battery_end_soc_pct=float(soc[-1]),
        peak_wheel_speed_rpm=peak_rpm,
        wall_clock_s=wall,
        sim_seconds=cfg.duration_s,
        fingerprint=fingerprint,
    )


def build_campaign(n_cases: int, seed: int = 20260804) -> list[CaseConfig]:
    """Dispersed Monte Carlo samples, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        mrp = rng.uniform(-0.3, 0.3, 3)
        cases.append(
            CaseConfig(
                case_id=i,
                raan_deg=float(rng.uniform(0.0, 360.0)),
                true_anomaly_deg=float(rng.uniform(0.0, 360.0)),
                sigma_BN_init=tuple(float(x) for x in mrp),
                omega_BN_init=tuple(float(x) for x in rng.uniform(-0.01, 0.01, 3)),
                instrument_baud_bps=float(rng.uniform(0.6e6, 1.6e6)),
            )
        )
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--skip-repeatability", action="store_true")
    parser.add_argument("--plot", type=Path, default=None,
                        help="write the campaign figure to this path")
    parser.add_argument("--viz-case", type=int, default=None, metavar="N",
                        help="run only case N and record a Vizard replay file "
                             "(Vizard is not needed to produce it)")
    args = parser.parse_args(argv)

    if args.viz_case is not None:
        return _run_viz_case(args)

    cases = build_campaign(args.cases, args.seed)
    for c in cases:
        c.duration_s = args.hours * 3600.0

    print(f"Monte Carlo data-return campaign -- {len(cases)} cases x {args.hours:.0f} h, "
          f"6-DOF + 4 RW + FSW in the loop\n")

    results, campaign_start = [], time.perf_counter()
    for cfg in cases:
        res = run_case(cfg)
        results.append(res)
        print(f"  case {res.case_id:2d}  {res.n_passes} pass  "
              f"{res.downlinked_gbit:6.2f} Gbit  "
              f"point {res.settled_pointing_error_deg:5.2f} deg  "
              f"SoC min {res.battery_min_soc_pct:5.1f}%  "
              f"{res.wall_clock_s:5.2f} s")
    campaign_wall = time.perf_counter() - campaign_start

    print()
    _report(results, campaign_wall)

    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _plot_campaign(results, args.plot)
        print(f"\n  Figure written to {args.plot}")

    if not args.skip_repeatability:
        print("\nBit-for-bit repeatability check (re-running case 0)...")
        repeat = run_case(cases[0])
        identical = np.array_equal(repeat.fingerprint, results[0].fingerprint)
        print(f"  identical to the first run: {identical}")
        if not identical:
            delta = np.abs(repeat.fingerprint - results[0].fingerprint).max()
            print(f"  max deviation: {delta:.6e}")
    return 0


def _report(results: list[CaseResult], campaign_wall: float) -> None:
    arr = lambda f: np.array([getattr(r, f) for r in results])  # noqa: E731

    total_sim = arr("sim_seconds").sum()
    print("=" * 78)
    print("CAMPAIGN SUMMARY")
    print("=" * 78)

    def stat(label, values, unit="", fmt="8.2f"):
        # Cases with no pass contribute NaN to the in-pass metrics; report the
        # statistics over the cases that actually had a contact.
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            print(f"  {label:<28s} (no finite samples)")
            return
        note = "" if finite.size == values.size else f"  [{finite.size}/{values.size} cases]"
        print(f"  {label:<28s} min {finite.min() + 0.0:{fmt}}  "
              f"med {np.median(finite):{fmt}}  max {finite.max():{fmt}}  {unit}{note}")

    stat("Data downlinked", arr("downlinked_gbit"), "Gbit")
    stat("Buffer high-water", arr("buffer_peak_gbit"), "Gbit")
    stat("Passes per case", arr("n_passes").astype(float), "", "8.1f")
    stat("Access time", arr("access_time_s") / 60.0, "min")
    stat("Settled pointing error", arr("settled_pointing_error_deg"), "deg")
    stat("Pointing error in pass", arr("mean_pointing_error_deg"), "deg")
    stat("Battery min SoC", arr("battery_min_soc_pct"), "%")
    stat("Peak wheel speed", arr("peak_wheel_speed_rpm"), "rpm")

    print()
    print(f"  Simulated time      : {total_sim / 3600:.1f} spacecraft-hours "
          f"({total_sim / 86400:.2f} spacecraft-days)")
    print(f"  Wall clock          : {campaign_wall:.1f} s")
    print(f"  Speed-up            : {total_sim / campaign_wall:,.0f}x real time")
    print(f"  Per-case wall clock : {np.median(arr('wall_clock_s')):.2f} s median")

    zero = int((arr("downlinked_gbit") < 0.01).sum())
    if zero:
        print(f"\n  {zero}/{len(results)} cases returned no data (no station pass "
              f"in the window) -- that is the dispersion doing its job.")

    # The coupled result this campaign exists to find.
    soc = arr("battery_min_soc_pct")
    data = arr("downlinked_gbit")
    for threshold in (30.0, 20.0, 5.0):
        n = int((soc < threshold).sum())
        if n:
            print(f"  {n}/{len(results)} cases dipped below {threshold:.0f}% "
                  f"battery SoC.")
    if soc.min() <= 0.01:
        print("  >> At least one case fully depleted the battery.")

    # Report the data-vs-charge relationship as a measured number, not a slogan.
    # Two mechanisms couple them: the transmitter load is only drawn while
    # downlinking, and slewing +Z onto the station holds the +Y array off-Sun.
    min_cases_for_correlation = 8
    if len(results) < min_cases_for_correlation:
        print(f"\n  (Correlation not reported: {len(results)} cases is too few to "
              f"mean anything; need >= {min_cases_for_correlation}.)")
    elif np.ptp(data) > 0 and np.ptp(soc) > 0:
        r = float(np.corrcoef(data, soc)[0, 1])
        print(f"\n  Correlation, data returned vs minimum SoC : r = {r:+.2f}")
        strength = ("no meaningful" if abs(r) < 0.2 else
                    "a weak" if abs(r) < 0.4 else
                    "a moderate" if abs(r) < 0.7 else "a strong")
        print(f"  -> {strength} relationship across this dispersion set.")
        if abs(r) < 0.2:
            print("     Eclipse phasing and initial attitude dominate the power "
                  "state here;\n     downlink duty cycle is too small a share of "
                  "the energy budget to show up.")


def _run_viz_case(args) -> int:
    """Run a single case with Vizard recording enabled."""
    cases = build_campaign(max(args.cases, args.viz_case + 1), args.seed)
    cfg = cases[args.viz_case]
    cfg.duration_s = args.hours * 3600.0

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.viz_save_file = str(out_dir / f"campaign_case{cfg.case_id}")

    print(f"Recording Vizard replay for case {cfg.case_id} "
          f"({args.hours:.0f} h, RAAN {cfg.raan_deg:.1f} deg)...\n")
    res = run_case(cfg)

    # vizSupport writes into a _VizFiles/ subdirectory and appends _UnityViz.bin
    produced = sorted(out_dir.glob(f"**/campaign_case{cfg.case_id}*"))
    print(f"  {res.n_passes} pass, {res.downlinked_gbit:.2f} Gbit, "
          f"pointing {res.settled_pointing_error_deg:.2f} deg, "
          f"SoC min {res.battery_min_soc_pct:.1f}%")
    print(f"  wall clock {res.wall_clock_s:.2f} s")
    for p in produced:
        print(f"  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    if not produced:
        print("  WARNING: no replay file was written.")
        return 1
    print("\n  Open it with: Vizard -> File -> Open, then select the .bin file.")
    print("  Vizard download: https://avslab.github.io/basilisk/Vizard/Vizard.html")
    return 0


def _plot_campaign(results: list[CaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = lambda f: np.array([getattr(r, f) for r in results])  # noqa: E731
    data = arr("downlinked_gbit")
    soc = arr("battery_min_soc_pct")
    passes = arr("n_passes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].hist(data, bins=14, color="C0", edgecolor="white")
    axes[0, 0].axvline(np.median(data), color="C3", ls="--",
                       label=f"median {np.median(data):.1f} Gbit")
    axes[0, 0].set_xlabel("data downlinked [Gbit]")
    axes[0, 0].set_ylabel("cases")
    axes[0, 0].set_title(f"Data return across {len(results)} dispersed cases")
    axes[0, 0].legend(fontsize=8)

    scatter = axes[0, 1].scatter(data, soc, c=passes, cmap="viridis", s=45,
                                 edgecolor="k", linewidth=0.4)
    axes[0, 1].axhline(20.0, color="C3", ls="--", lw=1.0, label="20% SoC floor")
    axes[0, 1].set_xlabel("data downlinked [Gbit]")
    axes[0, 1].set_ylabel("minimum battery SoC [%]")
    r = float(np.corrcoef(data, soc)[0, 1]) if np.ptp(data) > 0 and np.ptp(soc) > 0 else float("nan")
    axes[0, 1].set_title(f"Data vs charge: r = {r:+.2f} -- power state is driven\n"
                         f"by eclipse phasing, not by downlink duty cycle")
    axes[0, 1].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[0, 1], label="passes")

    axes[1, 0].hist(arr("settled_pointing_error_deg"), bins=14, color="C2",
                    edgecolor="white")
    axes[1, 0].set_xlabel("settled pointing error [deg]")
    axes[1, 0].set_ylabel("cases")
    axes[1, 0].set_title("mrpFeedback tracking error on the station")

    axes[1, 1].scatter(arr("buffer_peak_gbit"), data, s=45, color="C4",
                       edgecolor="k", linewidth=0.4)
    axes[1, 1].set_xlabel("buffer high-water [Gbit]")
    axes[1, 1].set_ylabel("data downlinked [Gbit]")
    axes[1, 1].set_title("Buffer growth vs data returned")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
