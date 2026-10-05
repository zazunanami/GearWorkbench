from __future__ import annotations

import itertools
import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
gui = pytest.importorskip("makelpro.gearbox_gui", exc_type=ImportError)

EX = np.array([1.0, 0.0, 0.0])
# (input rotation, gear 2 hand, gear 4 hand, shafts A and C on opposite sides, Fa*r couples)
CASES = list(itertools.product(["CW", "CCW"], ["Right", "Left"], ["Right", "Left"], [True, False], [False, True]))


def _compute_all(**overrides):
    return gui.GearboxWorkbench._computeAll(None, gui.GearboxInputs(**overrides))


def _unit(v):
    return v / np.linalg.norm(v)


def _contact_force_on_driven(o_driver, o_driven, r_driver, omega_driver, ft, fr, psi, hand_driven):
    """Mesh force on the driven gear from first principles (no GUI helpers)."""
    p = o_driver + r_driver * _unit(o_driven - o_driver)
    velocity = np.cross(omega_driver * EX, p - o_driver)
    e_r = _unit(p - o_driven)
    tooth_line = math.cos(psi) * EX + hand_driven * math.sin(psi) * np.cross(EX, e_r)
    u_t = _unit(velocity)
    axial = -ft * np.dot(u_t, tooth_line) / np.dot(EX, tooth_line)
    omega_driven = np.dot(velocity, np.cross(EX, p - o_driven)) / np.dot(p - o_driven, p - o_driven)
    return p, ft * u_t - fr * e_r + axial * EX, omega_driven


class _Shaft3D:
    """Simply supported shaft (bearings at x = 0 and x = L) under 3D gear forces."""

    def __init__(self, length, loads, with_couples):
        # loads: (x, offset of the mesh point from the shaft axis, force)
        self.length = length
        self.points = []
        for x, offset, force in loads:
            radial_part = force * np.array([0.0, 1.0, 1.0])
            self.points.append((x, np.array([x, 0.0, 0.0]) + offset, radial_part))
            axial_arm = np.array([x, 0.0, 0.0]) + (offset if with_couples else 0.0)
            self.points.append((x, axial_arm, force * EX))
        total_force = sum(f for _, _, f in self.points)
        total_moment = sum(np.cross(arm, f) for _, arm, f in self.points)
        r2 = np.array([0.0, -total_moment[2] / length, total_moment[1] / length])
        self.r1 = np.array([0.0, -total_force[1], -total_force[2]]) - r2
        self.r2 = r2
        self.torque = -total_moment[0]

    def bending(self, xs, side):
        moment = np.cross(np.array([-xs, 0.0, 0.0]), self.r1)
        for x, arm, force in self.points:
            if x < xs or (side > 0 and x == xs):
                moment = moment + np.cross(arm - np.array([xs, 0.0, 0.0]), force)
        return moment[1:]

    def unit_load_integral(self, weight, breaks):
        """Exact integral of the bending moment times a piecewise linear weight."""
        cuts = sorted({0.0, self.length, *[p[0] for p in self.points], *breaks})
        total = np.zeros(2)
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (a + b)
            total += (b - a) / 6.0 * (
                self.bending(a, 1) * weight(a) + 4.0 * self.bending(mid, 1) * weight(mid) + self.bending(b, -1) * weight(b)
            )
        return total


def test_couple_jump_is_sampled_on_both_sides_and_diagram_closes() -> None:
    length, couple = 200.0, 1000.0
    moments = [(100.0, couple)]
    reactions = gui.multiLoadReactions(length, [], moments)
    x, _, moment = gui.buildShearMoment(length, reactions, [], moments)
    assert abs(moment[-1]) < 1e-9
    idx = np.flatnonzero(x == 100.0)
    assert idx.size == 2
    # A CCW couple at midspan makes the moment jump from +C/2 down to -C/2
    assert abs(moment[idx[0]] - 0.5 * couple) < 1e-9
    assert abs(moment[idx[1]] + 0.5 * couple) < 1e-9


@pytest.mark.parametrize("case", CASES)
def test_all_moment_diagrams_close(case) -> None:
    rotation, hand2, hand4, opposite, couples = case
    _, shaft_data = _compute_all(
        inputRotation=rotation, gear2Hand=hand2, gear4Hand=hand4,
        shaftsACOppositeSides=opposite, includeAxialMoment=couples,
    )
    for key in "ABC":
        assert abs(shaft_data[key]["M_t"][-1]) < 1e-6
        assert abs(shaft_data[key]["M_r"][-1]) < 1e-6


@pytest.mark.parametrize("opposite_sides", [True, False])
def test_shaft_b_loads_respect_torque_balance(opposite_sides: bool) -> None:
    results, _ = _compute_all(shaftsACOppositeSides=opposite_sides)
    ft23, fr23 = results["mesh23"]["Ft"], results["mesh23"]["Fr"]
    ft45, fr45 = results["mesh45"]["Ft"], results["mesh45"]["Fr"]
    assert abs(ft23 * results["d3"] / 2.0 - ft45 * results["d4"] / 2.0) < 1e-6

    tangential_total = sum(results["reactions"]["B_t"])
    radial_total = sum(results["reactions"]["B_r"])
    if opposite_sides:
        assert abs(abs(tangential_total) - (ft23 + ft45)) < 1e-6
        assert abs(abs(radial_total) - (fr23 - fr45)) < 1e-6
    else:
        assert abs(abs(tangential_total) - (ft23 - ft45)) < 1e-6
        assert abs(abs(radial_total) - (fr23 + fr45)) < 1e-6


@pytest.mark.parametrize("case", CASES)
def test_shaft_loads_match_independent_3d_statics(case) -> None:
    rotation, hand2, hand4, opposite, couples = case
    inputs = gui.GearboxInputs(
        inputRotation=rotation, gear2Hand=hand2, gear4Hand=hand4,
        shaftsACOppositeSides=opposite, includeAxialMoment=couples,
    )
    results, shaft_data = gui.GearboxWorkbench._computeAll(None, inputs)

    psi = math.radians(inputs.helixAngleDeg)
    d2, d4 = inputs.gear2DiameterMm, inputs.gear4DiameterMm
    d3, d5 = d2 / inputs.stage1Ratio, d4 / inputs.stage2Ratio
    t_a = inputs.inputTorqueNm
    t_b = t_a / inputs.stage1Ratio
    t_c = t_b / inputs.stage2Ratio
    ft23, fr23, _, _ = gui.meshForces(t_a, d2, inputs.normalPressureAngleDeg, inputs.helixAngleDeg)
    ft45, fr45, _, _ = gui.meshForces(t_b, d4, inputs.normalPressureAngleDeg, inputs.helixAngleDeg)

    center_a = np.array([0.0, (d2 + d3) / 2.0, 0.0])
    center_b = np.zeros(3)
    center_c = np.array([0.0, (d4 + d5) / 2.0 * (-1.0 if opposite else 1.0), 0.0])
    hand = {"Right": 1.0, "Left": -1.0}
    omega_a = 1.0 if rotation == "CW" else -1.0  # CW seen from x = 0 is a spin about +X
    p23, on_gear3, omega_b = _contact_force_on_driven(center_a, center_b, d2 / 2.0, omega_a, ft23, fr23, psi, -hand[hand2])
    p45, on_gear5, omega_c = _contact_force_on_driven(center_b, center_c, d4 / 2.0, omega_b, ft45, fr45, psi, -hand[hand4])
    on_gear2, on_gear4 = -on_gear3, -on_gear5

    geo = results["geometry"]
    shafts = {
        "A": _Shaft3D(geo["LA"], [(geo["x2"], p23 - center_a, on_gear2)], couples),
        "B": _Shaft3D(geo["LB"], [(geo["x3"], p23 - center_b, on_gear3), (geo["x4"], p45 - center_b, on_gear4)], couples),
        "C": _Shaft3D(geo["LC"], [(geo["x5"], p45 - center_c, on_gear5)], couples),
    }
    gear_xs = {"A": [geo["x2"]], "B": [geo["x3"], geo["x4"]], "C": [geo["x5"]]}
    diameters = {"A": inputs.shaftADiameterMm, "B": inputs.shaftBDiameterMm, "C": inputs.shaftCDiameterMm}

    # Shaft speeds and torques from the no-slip condition and equilibrium
    assert math.isclose(abs(omega_b), inputs.stage1Ratio, rel_tol=1e-12)
    assert math.isclose(abs(omega_c), inputs.stage1Ratio * inputs.stage2Ratio, rel_tol=1e-12)
    assert math.isclose(abs(shafts["A"].torque), t_a * 1000.0, rel_tol=1e-9)
    assert abs(shafts["B"].torque) < 1e-6 * t_b * 1000.0
    assert math.isclose(abs(shafts["C"].torque), t_c * 1000.0, rel_tol=1e-9)

    for gear, force in (("2", on_gear2), ("3", on_gear3), ("4", on_gear4), ("5", on_gear5)):
        assert math.isclose(results["axial"][gear], force[0], rel_tol=1e-9)

    for key, shaft in shafts.items():
        radial = results["reactions"][key + "_r"]
        tangential = results["reactions"][key + "_t"]
        assert np.allclose(np.abs(radial), np.abs([shaft.r1[1], shaft.r2[1]]), rtol=1e-9, atol=1e-6)
        assert np.allclose(np.abs(tangential), np.abs([shaft.r1[2], shaft.r2[2]]), rtol=1e-9, atol=1e-6)

        ei = inputs.elasticModulusMpa * math.pi * diameters[key] ** 4 / 64.0
        length = shaft.length
        for x0 in gear_xs[key]:
            expected = max(np.hypot(*shaft.bending(x0, -1)), np.hypot(*shaft.bending(x0, 1)))
            assert math.isclose(gui.resultantMomentAt(shaft_data[key], x0), expected, rel_tol=1e-9)

            def unit_load(x, x0=x0):
                return (1.0 - x0 / length) * x if x <= x0 else x0 * (1.0 - x / length)

            deflection = np.hypot(*shaft.unit_load_integral(unit_load, [x0])) / ei
            computed = float(np.interp(x0, shaft_data[key]["x_t"], shaft_data[key]["y"]))
            assert math.isclose(computed, deflection, rel_tol=1e-6)

        slope_left = np.hypot(*shaft.unit_load_integral(lambda x: 1.0 - x / length, [])) / ei
        slope_right = np.hypot(*shaft.unit_load_integral(lambda x: x / length, [])) / ei
        assert math.isclose(shaft_data[key]["th"][0], slope_left, rel_tol=1e-6)
        assert math.isclose(shaft_data[key]["th"][-1], slope_right, rel_tol=1e-6)


def test_results_without_couples_do_not_depend_on_rotation_or_hands() -> None:
    _, reference = _compute_all()
    for rotation, hand2, hand4 in itertools.product(["CW", "CCW"], ["Right", "Left"], ["Right", "Left"]):
        _, shaft_data = _compute_all(inputRotation=rotation, gear2Hand=hand2, gear4Hand=hand4)
        for key in "ABC":
            for field in ("M_res", "y", "th"):
                assert np.allclose(shaft_data[key][field], reference[key][field], rtol=1e-12, atol=1e-12)


def test_default_hands_make_shaft_b_thrusts_oppose() -> None:
    results, _ = _compute_all()
    fa23, fa45 = results["mesh23"]["Fa"], results["mesh45"]["Fa"]
    axial = results["axial"]
    assert axial["3"] * axial["4"] < 0.0
    assert math.isclose(abs(axial["3"] + axial["4"]), fa23 - fa45, rel_tol=1e-12)
    # Gear 4 with the other hand (same hand as gear 2) makes the thrusts add up
    results, _ = _compute_all(gear4Hand="Right")
    assert math.isclose(abs(results["axial"]["3"] + results["axial"]["4"]), fa23 + fa45, rel_tol=1e-12)


def test_midspan_couple_does_not_deflect_midspan() -> None:
    _, without_couple = _compute_all()
    _, with_couple = _compute_all(includeAxialMoment=True)
    x_a = without_couple["A"]["x_t"]
    y_without = float(np.interp(100.0, x_a, without_couple["A"]["y"]))
    y_with = float(np.interp(100.0, with_couple["A"]["x_t"], with_couple["A"]["y"]))
    assert abs(y_with - y_without) < 1e-9


def test_gerber_safety_factor_satisfies_criterion() -> None:
    res = gui.equivalentStressDEGerber(
        MresNmm=150000.0, torqueNmm=300000.0, dMm=30.0, sutMpa=900.0, syMpa=650.0,
        SeMpa=250.0, kfBending=1.85, kfsTorsion=1.51,
    )
    n = res["nFatigue"]
    residual = n * res["sigmaEq_a"] / 250.0 + (n * res["sigmaEq_m"] / 900.0) ** 2
    assert abs(residual - 1.0) < 1e-12


def test_gerber_limits_and_zero_load() -> None:
    torsion_only = gui.equivalentStressDEGerber(0.0, 300000.0, 30.0, 900.0, 650.0, 250.0, 1.0, 1.0)
    assert abs(torsion_only["nFatigue"] - 900.0 / torsion_only["sigmaEq_m"]) < 1e-9

    bending_only = gui.equivalentStressDEGerber(150000.0, 0.0, 30.0, 900.0, 650.0, 250.0, 1.0, 1.0)
    assert abs(bending_only["nFatigue"] - 250.0 / bending_only["sigmaEq_a"]) < 1e-9

    for moment in (0.0, 1e-11):  # exact zero and moment-diagram round-off at a support
        unloaded = gui.equivalentStressDEGerber(moment, 0.0, 30.0, 900.0, 650.0, 250.0, 1.85, 1.51)
        assert math.isinf(unloaded["nFatigue"])
        assert math.isinf(unloaded["nYield"])


def test_resultant_moment_uses_larger_side_of_couple_jump() -> None:
    length = 200.0
    loads = [(100.0, -1000.0)]
    moments = [(100.0, 30000.0)]
    reactions = gui.multiLoadReactions(length, loads, moments)
    x, _, moment = gui.buildShearMoment(length, reactions, loads, moments)
    shaft = {"x_t": x, "M_res": np.abs(moment)}
    # Left side: Rleft * 100 = 65000 Nmm; right side after the CCW jump: 35000 Nmm
    assert abs(gui.resultantMomentAt(shaft, 100.0) - 65000.0) < 1e-6


def test_torque_only_on_the_transmitting_span() -> None:
    results, _ = _compute_all()
    spans = results["torqueSpans"]
    assert spans == {"A": (0.0, 100.0), "B": (100.0, 300.0), "C": (100.0, 200.0)}
    assert gui.transmittedTorqueNmm(0.0, 1.0, spans["A"]) == 1.0
    assert gui.transmittedTorqueNmm(200.0, 1.0, spans["A"]) == 0.0
    assert gui.transmittedTorqueNmm(0.0, 1.0, spans["B"]) == 0.0
    assert gui.transmittedTorqueNmm(400.0, 1.0, spans["B"]) == 0.0
    assert gui.transmittedTorqueNmm(100.0, 1.0, spans["B"]) == 1.0
    assert gui.transmittedTorqueNmm(0.0, 1.0, spans["C"]) == 0.0
    assert gui.transmittedTorqueNmm(200.0, 1.0, spans["C"]) == 1.0


def test_speeds_power_and_pitch_line_velocity() -> None:
    results, _ = _compute_all()
    assert results["speeds"] == {"A": 40.0, "B": 200.0, "C": 1200.0}
    assert math.isclose(results["power_kW"], 300.0 * 2.0 * math.pi * 40.0 / 60.0 / 1000.0, rel_tol=1e-12)
    assert math.isclose(results["pitchLineVelocity"]["23"], math.pi * 0.3 * 40.0 / 60.0, rel_tol=1e-12)
    assert math.isclose(results["pitchLineVelocity"]["45"], math.pi * 0.3 * 200.0 / 60.0, rel_tol=1e-12)


@pytest.mark.parametrize(
    ("diameter", "key"),
    [(5.0, "-"), (10.0, "3x3"), (17.0, "5x5"), (30.0, "8x7"), (37.77, "10x8"), (80.0, "22x14"), (250.0, "56x32")],
)
def test_key_suggestion_follows_din_6885(diameter: float, key: str) -> None:
    assert gui.GearboxWorkbench._keySuggestion(None, diameter) == key


def test_shaft_drawings_follow_the_analysis_geometry() -> None:
    from makelpro.technic_draw import build_demo_shafts

    inputs = gui.GearboxInputs()
    expected = {
        "Shaft A (Input)": (inputs.shaftALengthMm, [inputs.gear2PosFromAmm]),
        "Shaft B (Intermediate)": (inputs.shaftBLengthMm, [inputs.gear3PosFromCmm, inputs.gear4PosFromCmm]),
        "Shaft C (Output)": (inputs.shaftCLengthMm, [inputs.gear5PosFromEmm]),
    }
    for name, segments, keyways, total, _ in build_demo_shafts():
        starts = np.cumsum([0.0] + [length for length, _, _ in segments])
        bearings = [a + length / 2.0 for a, (length, _, kind) in zip(starts, segments) if kind == "bearing"]
        gears = [a + length / 2.0 for a, (length, _, kind) in zip(starts, segments) if kind == "gear"]
        span, gear_positions = expected[name]
        assert starts[-1] == total
        assert len(bearings) == 2 and bearings[1] - bearings[0] == span
        assert [g - bearings[0] for g in gears] == gear_positions
        for key_start, key_length, _ in keyways:
            assert any(
                kind in ("gear", "coupling") and a <= key_start and key_start + key_length <= a + length
                for a, (length, _, kind) in zip(starts, segments)
            )


def test_free_body_diagrams_match_the_analysis() -> None:
    from makelpro.fbd import build_demo_cases

    results, _ = _compute_all()
    geo = results["geometry"]
    figures = dict(build_demo_cases())
    tangential, radial = figures["fbd_shaft_b.png"]
    for panel, plane, forces in (
        (tangential, "B_t", [results["mesh23"]["Ft"], results["mesh45"]["Ft"]]),
        (radial, "B_r", [results["mesh23"]["Fr"], results["mesh45"]["Fr"]]),
    ):
        _, length, supports, loads = panel
        assert length == geo["LB"]
        assert [x for x, _, _ in supports] == [0, length]
        assert [round(value) for _, _, value in supports] == [round(r) for r in results["reactions"][plane]]
        assert [x for x, _, _, _ in loads] == [geo["x3"], geo["x4"]]
        assert [round(value) for _, _, value, _ in loads] == [round(f) for f in forces]
    # Opposite-sides layout: tangential loads act together, radial loads oppose
    assert [down for _, _, _, down in tangential[3]] == [True, True]
    assert [down for _, _, _, down in radial[3]] == [True, False]

    # Single-gear shafts use the resultant of both planes
    for name, ft, fr in (
        ("fbd_shaft_a.png", results["mesh23"]["Ft"], results["mesh23"]["Fr"]),
        ("fbd_shaft_c.png", results["mesh45"]["Ft"], results["mesh45"]["Fr"]),
    ):
        (_, _, supports, loads), = figures[name]
        assert abs(loads[0][2] - math.hypot(ft, fr)) < 1.0
        assert all(abs(value - math.hypot(ft, fr) / 2.0) < 1.0 for _, _, value in supports)


def test_gui_recompute_highlight_reset_and_results_window(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    def fail_on_dialog(*args, **kwargs):
        pytest.fail(f"Unexpected error dialog: {args[2] if len(args) > 2 else args}")

    monkeypatch.setattr(gui.QMessageBox, "critical", staticmethod(fail_on_dialog))
    monkeypatch.setattr(gui.QMessageBox, "warning", staticmethod(fail_on_dialog))
    app = QApplication.instance() or QApplication([])
    window = gui.GearboxWorkbench()
    try:
        fatigue = window.fatigueYieldTable
        rows = {(fatigue.item(r, 0).text(), fatigue.item(r, 1).text()): r for r in range(fatigue.rowCount())}
        assert fatigue.item(rows[("B", "Bearing C shoulder")], 7).text() == "inf"
        assert fatigue.item(rows[("A", "Bearing B shoulder")], 7).text() == "inf"
        assert fatigue.item(rows[("A", "Bearing A shoulder")], 7).text() == "6.08"

        # A stricter target flips bold status cells and changes plain cells, highlighting both
        window.targetFatigueSpin.setValue(10.0)
        window._recomputeAndRedraw()
        fatigue_status = fatigue.item(rows[("A", "Gear 2 seat/keyway")], 10)
        final_status = window.finalDesignTable.item(0, 9)
        final_required = window.finalDesignTable.item(0, 2)
        assert (fatigue_status.text(), final_status.text()) == ("FAIL", "INCREASE")
        assert final_required.font().bold() and window.resetButton.isEnabled()
        window._resetHighlights()
        assert fatigue_status.font().bold() and final_status.font().bold()
        assert not final_required.font().bold()
        assert not window.resetButton.isEnabled()

        # The recommended diameter has to meet the target once it is analysed
        recommended = math.ceil(float(window.finalDesignTable.item(0, 5).text()) * 100.0) / 100.0
        window.dShaftASpin.setValue(recommended)
        window._recomputeAndRedraw()
        assert fatigue.item(rows[("A", "Gear 2 seat/keyway")], 10).text() == "OK"
        assert window.finalDesignTable.item(0, 9).text() == "PASS"

        window._openResultsWindow()
        tabs = window.resultsDialog.tabs
        names = [tabs.tabText(k) for k in range(tabs.count())]
        assert {"Mesh Forces", "Speeds and Power", "Axial Thrust", "Reactions", "Max Bending", "Geometry"} <= set(names)
    finally:
        if window.resultsDialog is not None:
            window.resultsDialog.close()
        window.close()
    assert app is not None
