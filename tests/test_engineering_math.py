from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

gui = pytest.importorskip("makelpro.gearbox_gui", exc_type=ImportError)


def _compute_all(opposite_sides: bool = True, include_axial_moment: bool = False):
    window = SimpleNamespace(oppositeOnShaftB=SimpleNamespace(isChecked=lambda: opposite_sides))
    inputs = gui.GearboxInputs(includeAxialMoment=include_axial_moment)
    return gui.GearboxWorkbench._computeAll(window, inputs)


def test_couple_moment_diagram_closes_at_right_support() -> None:
    length, couple = 200.0, 1000.0
    moments = [(100.0, couple)]
    reactions = gui.multiLoadReactions(length, [], moments)
    x, _, moment = gui.buildShearMoment(length, reactions, [], moments)
    assert abs(moment[-1]) < 1e-6
    idx = int(np.where(x == 100.0)[0][0])
    assert abs(moment[idx - 1] - 0.5 * couple * 99.0 / 100.0) < 1e-6
    assert abs(moment[idx] - (-0.5 * couple)) < 1e-6


@pytest.mark.parametrize("opposite_sides", [True, False])
def test_all_moment_diagrams_close_with_axial_moment(opposite_sides: bool) -> None:
    _, shaft_data = _compute_all(opposite_sides, include_axial_moment=True)
    for key in "ABC":
        assert abs(shaft_data[key]["M_t"][-1]) < 1e-6
        assert abs(shaft_data[key]["M_r"][-1]) < 1e-6


@pytest.mark.parametrize("opposite_sides", [True, False])
def test_shaft_b_loads_respect_torque_balance(opposite_sides: bool) -> None:
    results, _ = _compute_all(opposite_sides)
    ft23, fr23 = results["mesh23"]["Ft"], results["mesh23"]["Fr"]
    ft45, fr45 = results["mesh45"]["Ft"], results["mesh45"]["Fr"]
    # Gear 3 is driven and gear 4 drives, so their tangential torques must cancel.
    assert abs(ft23 * results["d3"] / 2.0 - ft45 * results["d4"] / 2.0) < 1e-6

    tangential_total = sum(results["reactions"]["B_t"])
    radial_total = sum(results["reactions"]["B_r"])
    if opposite_sides:
        assert abs(tangential_total - (ft23 + ft45)) < 1e-6
        assert abs(radial_total - (fr23 - fr45)) < 1e-6
    else:
        assert abs(tangential_total - (ft23 - ft45)) < 1e-6
        assert abs(radial_total - (fr23 + fr45)) < 1e-6


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

    unloaded = gui.equivalentStressDEGerber(0.0, 0.0, 30.0, 900.0, 650.0, 250.0, 1.0, 1.0)
    assert math.isinf(unloaded["nFatigue"])
    assert math.isinf(unloaded["nYield"])


def test_resultant_moment_uses_larger_side_of_couple_jump() -> None:
    length = 200.0
    loads = [(100.0, -1000.0)]
    moments = [(100.0, 30000.0)]
    reactions = gui.multiLoadReactions(length, loads, moments)
    x, _, moment = gui.buildShearMoment(length, reactions, loads, moments)
    shaft = {"x_t": x, "M_t": np.zeros_like(x), "x_r": x, "M_r": moment, "moms_r": moments}
    # Left limit: Rleft * 100 = 65000 Nmm; right limit after the CCW jump: 35000 Nmm.
    assert abs(gui.resultantMomentAt(shaft, 100.0) - 65000.0) < 1e-6
