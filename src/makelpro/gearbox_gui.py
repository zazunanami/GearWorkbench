import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.join(os.path.dirname(__file__), "assets")
    return os.path.join(base_path, relative_path)

import math
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTabWidget,
    QTabBar,
    QScrollArea,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QFrame,
    QDoubleSpinBox,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
    QHeaderView,
)

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------
# Data model
# ----------------------------


@dataclass
class GearboxInputs:
    # Operating conditions
    inputTorqueNm: float = 300.0
    inputSpeedRpm: float = 40.0

    # Gear geometry
    normalPressureAngleDeg: float = 20.0
    helixAngleDeg: float = 30.0

    stage1Ratio: float = 5.0
    stage2Ratio: float = 6.0

    gear2DiameterMm: float = 300.0
    gear4DiameterMm: float = 300.0

    # Shaft geometry (project defaults)
    shaftALengthMm: float = 200.0
    gear2PosFromAmm: float = 100.0

    shaftBLengthMm: float = 400.0
    gear3PosFromCmm: float = 100.0
    gear4PosFromCmm: float = 300.0

    shaftCLengthMm: float = 200.0
    gear5PosFromEmm: float = 100.0

    # Face widths (visual only)
    faceWidthBigMm: float = 100.0
    faceWidthSmallMm: float = 80.0

    # Shaft baseline diameters for checks
    shaftADiameterMm: float = 30.0
    shaftBDiameterMm: float = 36.0
    shaftCDiameterMm: float = 25.0

    # Deflection and slope limits
    allowDeflectionMm: float = 0.12
    allowSlopeRad: float = 0.001

    # Material
    elasticModulusMpa: float = 210000.0
    sutMpa: float = 900.0
    syMpa: float = 650.0

    # Fatigue design targets
    targetFatigueN: float = 2.0
    targetYieldN: float = 3.0

    # Stress concentration (Kt, Kts, q)
    ktBending: float = 2.0   
    ktsTorsion: float = 1.6  
    notchSensQ: float = 0.85 

    # Marin factors inputs (Individual Surface Finish)
    surfaceFinishA: str = "Machined"
    surfaceFinishB: str = "Machined"
    surfaceFinishC: str = "Machined"
    
    reliability: str = "90%"
    miscFactor: float = 1.0
    loadFactor: float = 1.0
    
    # Calculation Options
    includeAxialMoment: bool = False 


# ----------------------------
# UI helpers
# ----------------------------


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event):
        event.ignore()

    def widget(self):
        return self


class NoWheelComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class NoWheelTabBar(QTabBar):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabBar(NoWheelTabBar(self))

    def wheelEvent(self, event):
        event.ignore()


def _tableItem(text: str, alignCenter: bool = True, highlight: bool = False, bold: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    if alignCenter:
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    
    try:
        font = QFont("Segoe UI")
        font.setPointSize(9)
        
        if bold:
            font.setWeight(QFont.Weight.Bold)
        
        if highlight:
            item.setForeground(QColor("#ffb300")) # Orange for highlight
            font.setWeight(QFont.Weight.Bold)
        else:
            item.setForeground(QColor("#c9d1d9")) # Default Text Color
        
        item.setFont(font)
    except:
        pass
        
    return item


def _optimizeTableLayout(table: QTableWidget):
    """Automatically resize table columns for better fit."""
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if table.columnCount() > 0:
        header.setSectionResizeMode(table.columnCount() - 1, QHeaderView.ResizeMode.Stretch)


class PlotCanvas(FigureCanvas):
    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = "", heightInches: float = 2.4):
        self.figure = Figure(figsize=(10, heightInches), dpi=100)
        self.figure.patch.set_facecolor('#0d1117') 

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#161b22') 
        
        text_color = '#c9d1d9'
        grid_color = '#30363d'
        
        self.ax.set_title(title, color=text_color, fontweight='bold')
        self.ax.set_xlabel(xlabel, color=text_color)
        self.ax.set_ylabel(ylabel, color=text_color)
        
        self.ax.tick_params(axis='x', colors=text_color)
        self.ax.tick_params(axis='y', colors=text_color)
        
        self.ax.spines['bottom'].set_color(text_color)
        self.ax.spines['top'].set_color(text_color)
        self.ax.spines['left'].set_color(text_color)
        self.ax.spines['right'].set_color(text_color)
        
        self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)

        super().__init__(self.figure)

    def clear(self, title: str | None = None):
        self.ax.clear()
        self.ax.set_facecolor('#161b22')
        text_color = '#c9d1d9'
        grid_color = '#30363d'
        
        if title:
            self.ax.set_title(title, color=text_color, fontweight='bold')
            
        self.ax.tick_params(axis='x', colors=text_color)
        self.ax.tick_params(axis='y', colors=text_color)
        self.ax.spines['bottom'].set_color(text_color)
        self.ax.spines['top'].set_color(text_color)
        self.ax.spines['left'].set_color(text_color)
        self.ax.spines['right'].set_color(text_color)
        self.ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)

    def wheelEvent(self, event):
        parent = self.parent()
        scrollArea = None
        while parent is not None:
            if isinstance(parent, QScrollArea):
                scrollArea = parent
                break
            parent = parent.parent()

        if scrollArea is None:
            event.ignore()
            return

        delta = event.angleDelta().y()
        sb = scrollArea.verticalScrollBar()
        sb.setValue(sb.value() - int(delta / 2))
        event.accept()

    @staticmethod
    def niceYLim(ax, y: np.ndarray, padFrac: float = 0.2):
        try:
            yMin = float(np.min(y))
            yMax = float(np.max(y))
        except Exception:
            return
        if abs(yMin - yMax) < 1e-9:
            pad = 1.0 if yMin == 0.0 else abs(yMin) * 0.25
            ax.set_ylim(yMin - pad, yMax + pad)
            return
        pad = (yMax - yMin) * padFrac
        ax.set_ylim(yMin - pad, yMax + pad)


# ----------------------------
# Engineering math
# ----------------------------


def degToRad(deg: float) -> float:
    return deg * math.pi / 180.0


def transversePressureAngleDeg(phiNDeg: float, helixDeg: float) -> float:
    phiNRad = degToRad(phiNDeg)
    psiRad = degToRad(helixDeg)
    tanPhiT = math.tan(phiNRad) / math.cos(psiRad)
    return math.degrees(math.atan(tanPhiT))


def meshForces(torqueNm: float, driverDiameterMm: float, phiNDeg: float, helixDeg: float):
    phiTDeg = transversePressureAngleDeg(phiNDeg, helixDeg)
    tanPhiT = math.tan(degToRad(phiTDeg))
    tanPsi = math.tan(degToRad(helixDeg))

    Ft = 2000.0 * torqueNm / max(1e-9, driverDiameterMm)
    Fr = Ft * tanPhiT
    Fa = Ft * tanPsi
    return Ft, Fr, Fa, phiTDeg


def multiLoadReactions(Lmm: float, loadsSignedUpN: list[tuple[float, float]], momentsCCWNmm: list[tuple[float, float]] | None = None):
    if momentsCCWNmm is None:
        momentsCCWNmm = []

    sumFy = sum(Fy for _, Fy in loadsSignedUpN)
    sumForceMomentLeft = sum(Fy * x for x, Fy in loadsSignedUpN)
    sumConcentratedMoments = sum(M for _, M in momentsCCWNmm)

    Rright = -(sumForceMomentLeft + sumConcentratedMoments) / max(1e-9, Lmm)
    Rleft = -(sumFy + Rright)

    return Rleft, Rright


def buildShearMoment(
    Lmm: float,
    reactionsUp: tuple[float, float],
    loadsSignedUp: list[tuple[float, float]],
    momentsSigned: list[tuple[float, float]] | None = None,
    stepMm: float = 1.0,
):
    if momentsSigned is None:
        momentsSigned = []

    x = np.arange(0.0, Lmm + stepMm, stepMm)
    V = np.zeros_like(x)
    M = np.zeros_like(x)

    Rleft, _ = reactionsUp
    V[:] = Rleft

    for xLoad, Fy in loadsSignedUp:
        idx = np.where(x >= xLoad)[0]
        V[idx] += Fy

    currentM = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        currentM += V[i - 1] * dx

        prevPos = x[i - 1]
        currPos = x[i]
        for xMom, mVal in momentsSigned:
            # Couples are CCW-positive (same convention as multiLoadReactions), so a
            # CCW couple makes the sagging-positive internal moment jump down by mVal.
            if prevPos < xMom <= currPos:
                currentM -= mVal

        M[i] = currentM

    return x, V, M


def resultantMomentAt(shaft: dict, xPos: float) -> float:
    """Resultant bending moment at xPos, taking the larger side of a couple jump.

    buildShearMoment stores the value after the jump at a couple location, so the
    value just before it is recovered by undoing the jump.
    """
    mT = float(np.interp(xPos, shaft["x_t"], shaft["M_t"]))
    mRAfter = float(np.interp(xPos, shaft["x_r"], shaft["M_r"]))
    jump = sum(mVal for xMom, mVal in shaft.get("moms_r", []) if abs(xMom - xPos) < 1e-9)
    mRBefore = mRAfter + jump
    return max(math.hypot(mT, mRBefore), math.hypot(mT, mRAfter))


def computeDeflectionAndSlope(
    xMm: np.ndarray,
    momentNmm: np.ndarray,
    elasticModulusMpa: float,
    shaftDiameterMm: float,
):
    xMm = np.asarray(xMm, dtype=float)
    momentNmm = np.asarray(momentNmm, dtype=float)

    if len(xMm) < 2:
        return np.zeros_like(xMm), np.zeros_like(xMm)

    Lmm = float(xMm[-1] - xMm[0])
    if Lmm <= 0:
        return np.zeros_like(xMm), np.zeros_like(xMm)

    I = (math.pi * (shaftDiameterMm ** 4)) / 64.0
    if I <= 0 or elasticModulusMpa <= 0:
        return np.zeros_like(xMm), np.zeros_like(xMm)

    kappa = momentNmm / (elasticModulusMpa * I)

    dx = np.diff(xMm)

    theta0 = np.zeros_like(xMm)
    theta0[1:] = np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * dx)

    y0 = np.zeros_like(xMm)
    y0[1:] = np.cumsum(0.5 * (theta0[1:] + theta0[:-1]) * dx)

    C1 = -y0[-1] / Lmm
    theta = theta0 + C1
    y = y0 + C1 * (xMm - xMm[0])

    return y, theta


# ----------------------------
# Marin factors
# ----------------------------


def enduranceLimitPrimeMpa(sutMpa: float) -> float:
    return min(0.5 * sutMpa, 700.0)


def surfaceFinishFactorKa(sutMpa: float, finish: str) -> float:
    finish = (finish or "Machined").strip().lower()
    if "ground" in finish:
        a, b = 1.58, -0.085
    elif "machined" in finish or "cold" in finish:
        a, b = 4.51, -0.265
    elif "hot" in finish:
        a, b = 57.7, -0.718
    elif "forged" in finish:
        a, b = 272.0, -0.995
    else:
        a, b = 4.51, -0.265
    return a * (max(1.0, sutMpa) ** b)


def sizeFactorKbBending(dMm: float) -> float:
    dMm = max(1e-6, float(dMm))
    if dMm <= 7.62:
        return 1.0
    if dMm <= 51.0:
        return (dMm / 7.62) ** -0.107
    if dMm <= 254.0:
        return 1.51 * (dMm ** -0.157)
    return max(0.3, 1.51 * (254.0 ** -0.157))


def reliabilityFactorKe(label: str) -> float:
    label = (label or "90%").strip()
    mapping = {
        "50%": 1.000,
        "90%": 0.897,
        "95%": 0.868,
        "99%": 0.814,
        "99.9%": 0.753,
    }
    return mapping.get(label, 0.897)


def correctedEnduranceLimitMpa(sutMpa, dMm, surfaceFinish, reliability, kLoad, kMisc):
    sePrime = enduranceLimitPrimeMpa(sutMpa)
    ka = surfaceFinishFactorKa(sutMpa, surfaceFinish)
    kb = sizeFactorKbBending(dMm)
    kc = float(kLoad)
    kd = 1.0
    ke = reliabilityFactorKe(reliability)
    kf = float(kMisc)
    se = sePrime * ka * kb * kc * kd * ke * kf
    return {"SePrime": sePrime, "ka": ka, "kb": kb, "kc": kc, "kd": kd, "ke": ke, "kMisc": kf, "Se": se}


def equivalentStressDEGerber(MresNmm, torqueNmm, dMm, sutMpa, syMpa, SeMpa, kfBending, kfsTorsion):
    dMm = max(1e-9, float(dMm))
    MresNmm = float(MresNmm)
    torqueNmm = float(torqueNmm)
    sigmaB_a = (32.0 * abs(MresNmm) * max(1.0, kfBending)) / (math.pi * (dMm ** 3))
    tauM = (16.0 * abs(torqueNmm) * max(1.0, kfsTorsion)) / (math.pi * (dMm ** 3))
    sigmaEq_a = sigmaB_a
    sigmaEq_m = math.sqrt(3.0) * tauM
    SeMpa = max(1e-9, float(SeMpa))
    sutMpa = max(1e-9, float(sutMpa))
    # Gerber: n*sigma_a/Se + (n*sigma_m/Sut)^2 = 1, i.e. a*n^2 + b*n - 1 = 0.
    # The rationalized positive root also covers a = 0 (n = Se/sigma_a).
    a = (sigmaEq_m / sutMpa) ** 2
    b = sigmaEq_a / SeMpa
    denom = b + math.sqrt(b * b + 4.0 * a)
    nFatigue = 2.0 / denom if denom > 0.0 else math.inf
    sigmaVM = math.sqrt((sigmaB_a ** 2) + 3.0 * (tauM ** 2))
    nYield = float(syMpa) / sigmaVM if sigmaVM > 0.0 else math.inf
    return {
        "sigmaB_a": sigmaB_a,
        "tauM": tauM,
        "sigmaEq_a": sigmaEq_a,
        "sigmaEq_m": sigmaEq_m,
        "nFatigue": nFatigue,
        "sigmaVM": sigmaVM,
        "nYield": nYield,
    }


# ----------------------------
# Results dialog
# ----------------------------


class ResultsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gearbox Results (Detailed)")
        self.setMinimumSize(1100, 750)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

    def setTables(self, tables: list[dict]):
        self.tabs.clear()
        for t in tables:
            tab = QWidget()
            v = QVBoxLayout(tab)
            v.setContentsMargins(10, 10, 10, 10)
            v.setSpacing(10)

            lbl = QLabel(f"<b>{t['title']}</b>")
            v.addWidget(lbl)

            table = QTableWidget()
            table.setColumnCount(len(t["headers"]))
            table.setHorizontalHeaderLabels(t["headers"])
            table.setRowCount(len(t["rows"]))

            for r, row in enumerate(t["rows"]):
                for c, val in enumerate(row):
                    table.setItem(r, c, _tableItem(val))

            table.horizontalHeader().setStretchLastSection(True)
            table.setAlternatingRowColors(True)
            table.setWordWrap(False)
            
            _optimizeTableLayout(table)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(table)

            v.addWidget(scroll)
            self.tabs.addTab(tab, t["tabName"])


# ----------------------------
# Main GUI
# ----------------------------


class GearboxWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gearbox Design Workbench")
        self.setWindowIcon(QIcon(resource_path("gear.ico")))
        self.setMinimumSize(1280, 800)

        self.inputs = GearboxInputs()
        self.previous_table_state = {} 

        # Central
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        # Left: Inputs (scroll)
        self.inputsGroup = QGroupBox("Inputs")
        self.inputsLayout = QFormLayout(self.inputsGroup)
        self.inputsLayout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.inputsLayout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self.inputsLayout.setVerticalSpacing(10)

        # Right: Tabs area
        self.rightPanel = QWidget()
        self.rightLayout = QVBoxLayout(self.rightPanel)
        self.rightLayout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = NoWheelTabWidget()
        
        # Bottom Bar: Button + Version
        bottomBar = QWidget()
        bottomLayout = QHBoxLayout(bottomBar)
        bottomLayout.setContentsMargins(0, 5, 5, 5)
        bottomLayout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Reset Highlights Button
        self.resetButton = QPushButton("Clear Compared Variables")
        self.resetButton.setToolTip("Acknowledge changes and reset highlights")
        # Default State: Disabled
        self.resetButton.setEnabled(False)
        self.resetButton.setStyleSheet("""
            QPushButton {
                background-color: #21262d; 
                color: #3fb950; 
                font-weight: bold; 
                border: 1px solid #30363d; 
                border-radius: 5px; 
                padding: 4px 10px;
            }
            QPushButton:disabled {
                color: #7d8590;
                background-color: #161b22;
                border-color: #30363d;
            }
        """)
        
        self.versionLabel = QLabel("Licensed under the Apache License 2.0 by Zazu Nanami")
        self.versionLabel.setStyleSheet("color: #505050; font-size: 11px; font-weight: bold; margin-left: 10px;")
        
        bottomLayout.addWidget(self.resetButton)
        bottomLayout.addWidget(self.versionLabel)

        self.rightLayout.addWidget(self.tabs, 1)
        self.rightLayout.addWidget(bottomBar, 0)

        root.addWidget(self._wrapInputsScroll(self.inputsGroup), 0)
        root.addWidget(self.rightPanel, 1)

        # Build UI
        self._buildInputs()
        self._buildTabs()
        self._applyDarkTheme()

        self.isDirty = False
        self.recalcButton.setEnabled(False)

        self.resultsDialog = None
        self.lastResults = None
        self.lastShaftData = None

        self._lastInputSignature = self._currentInputSignature()
        self._connectSignals()

        self._recomputeAndRedraw()

    # ----------------------------
    # UI construction
    # ----------------------------

    def _wrapInputsScroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(430)
        
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        v.addWidget(widget)

        scroll.setWidget(container)
        return scroll

    def _wrapInScroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        v = QVBoxLayout(container)
        v.setAlignment(Qt.AlignmentFlag.AlignTop)
        v.addWidget(widget)

        scroll.setWidget(container)
        return scroll

    def _buildInputs(self):
        # Operating
        self.torqueSpin = NoWheelDoubleSpinBox()
        self.torqueSpin.setRange(0.0, 1e6)
        self.torqueSpin.setDecimals(2)
        self.torqueSpin.setSingleStep(10.0)
        self.torqueSpin.setSuffix(" Nm")
        self.torqueSpin.setValue(self.inputs.inputTorqueNm)

        self.speedSpin = NoWheelDoubleSpinBox()
        self.speedSpin.setRange(0.0, 1e6)
        self.speedSpin.setDecimals(2)
        self.speedSpin.setSingleStep(5.0)
        self.speedSpin.setSuffix(" rpm")
        self.speedSpin.setValue(self.inputs.inputSpeedRpm)

        # Angles
        self.phiNSpin = NoWheelDoubleSpinBox()
        self.phiNSpin.setRange(0.0, 45.0)
        self.phiNSpin.setDecimals(2)
        self.phiNSpin.setSingleStep(0.5)
        self.phiNSpin.setSuffix(" deg")
        self.phiNSpin.setValue(self.inputs.normalPressureAngleDeg)

        self.psiSpin = NoWheelDoubleSpinBox()
        self.psiSpin.setRange(0.0, 45.0)
        self.psiSpin.setDecimals(2)
        self.psiSpin.setSingleStep(0.5)
        self.psiSpin.setSuffix(" deg")
        self.psiSpin.setValue(self.inputs.helixAngleDeg)

        # Ratios
        self.i1Spin = NoWheelDoubleSpinBox()
        self.i1Spin.setRange(1.0, 50.0)
        self.i1Spin.setDecimals(3)
        self.i1Spin.setSingleStep(0.1)
        self.i1Spin.setValue(self.inputs.stage1Ratio)

        self.i2Spin = NoWheelDoubleSpinBox()
        self.i2Spin.setRange(1.0, 50.0)
        self.i2Spin.setDecimals(3)
        self.i2Spin.setSingleStep(0.1)
        self.i2Spin.setValue(self.inputs.stage2Ratio)

        # Diameters
        self.d2Spin = NoWheelDoubleSpinBox()
        self.d2Spin.setRange(10.0, 2000.0)
        self.d2Spin.setDecimals(2)
        self.d2Spin.setSingleStep(10.0)
        self.d2Spin.setSuffix(" mm")
        self.d2Spin.setValue(self.inputs.gear2DiameterMm)

        self.d4Spin = NoWheelDoubleSpinBox()
        self.d4Spin.setRange(10.0, 2000.0)
        self.d4Spin.setDecimals(2)
        self.d4Spin.setSingleStep(10.0)
        self.d4Spin.setSuffix(" mm")
        self.d4Spin.setValue(self.inputs.gear4DiameterMm)

        # Shaft diameters
        self.dShaftASpin = NoWheelDoubleSpinBox()
        self.dShaftASpin.setRange(5.0, 300.0)
        self.dShaftASpin.setDecimals(2)
        self.dShaftASpin.setSingleStep(1.0)
        self.dShaftASpin.setSuffix(" mm")
        self.dShaftASpin.setValue(self.inputs.shaftADiameterMm)

        self.dShaftBSpin = NoWheelDoubleSpinBox()
        self.dShaftBSpin.setRange(5.0, 300.0)
        self.dShaftBSpin.setDecimals(2)
        self.dShaftBSpin.setSingleStep(1.0)
        self.dShaftBSpin.setSuffix(" mm")
        self.dShaftBSpin.setValue(self.inputs.shaftBDiameterMm)

        self.dShaftCSpin = NoWheelDoubleSpinBox()
        self.dShaftCSpin.setRange(5.0, 300.0)
        self.dShaftCSpin.setDecimals(2)
        self.dShaftCSpin.setSingleStep(1.0)
        self.dShaftCSpin.setSuffix(" mm")
        self.dShaftCSpin.setValue(self.inputs.shaftCDiameterMm)

        # Load direction checkbox
        self.oppositeOnShaftB = QCheckBox("Shafts A and C on opposite sides of Shaft B")
        self.oppositeOnShaftB.setToolTip(
            "Checked: Stage1/Stage2 radial forces oppose on Shaft B, tangential forces add.\n"
            "Unchecked: radial forces add, tangential forces oppose.\n"
            "Tangential directions follow from torque balance on Shaft B."
        )
        self.oppositeOnShaftB.setChecked(True)
        
        # Checkbox for Fa moment
        self.includeAxialMomentBox = QCheckBox("Include overturning moment from axial force (Fa * r)")
        self.includeAxialMomentBox.setChecked(False)

        # Deflection limits
        self.allowDeflSpin = NoWheelDoubleSpinBox()
        self.allowDeflSpin.setRange(0.0, 10.0)
        self.allowDeflSpin.setDecimals(4)
        self.allowDeflSpin.setSingleStep(0.001)
        self.allowDeflSpin.setSuffix(" mm")
        self.allowDeflSpin.setValue(self.inputs.allowDeflectionMm)

        self.allowSlopeSpin = NoWheelDoubleSpinBox()
        self.allowSlopeSpin.setRange(0.0, 0.1)
        self.allowSlopeSpin.setDecimals(6)
        self.allowSlopeSpin.setSingleStep(0.00001)
        self.allowSlopeSpin.setSuffix(" rad")
        self.allowSlopeSpin.setValue(self.inputs.allowSlopeRad)

        # Material
        self.elasticModulusSpin = NoWheelDoubleSpinBox()
        self.elasticModulusSpin.setRange(1.0, 400000.0)
        self.elasticModulusSpin.setDecimals(2)
        self.elasticModulusSpin.setSingleStep(1000.0)
        self.elasticModulusSpin.setSuffix(" MPa")
        self.elasticModulusSpin.setValue(self.inputs.elasticModulusMpa)

        self.sutSpin = NoWheelDoubleSpinBox()
        self.sutSpin.setRange(1.0, 3000.0)
        self.sutSpin.setDecimals(2)
        self.sutSpin.setSingleStep(10.0)
        self.sutSpin.setSuffix(" MPa")
        self.sutSpin.setValue(self.inputs.sutMpa)

        self.sySpin = NoWheelDoubleSpinBox()
        self.sySpin.setRange(1.0, 3000.0)
        self.sySpin.setDecimals(2)
        self.sySpin.setSingleStep(10.0)
        self.sySpin.setSuffix(" MPa")
        self.sySpin.setValue(self.inputs.syMpa)

        # UPDATED: Stress concentrations (Kt, Kts, q)
        self.ktSpin = NoWheelDoubleSpinBox()
        self.ktSpin.setRange(1.0, 10.0)
        self.ktSpin.setDecimals(3)
        self.ktSpin.setSingleStep(0.05)
        self.ktSpin.setValue(self.inputs.ktBending)

        self.ktsSpin = NoWheelDoubleSpinBox()
        self.ktsSpin.setRange(1.0, 10.0)
        self.ktsSpin.setDecimals(3)
        self.ktsSpin.setSingleStep(0.05)
        self.ktsSpin.setValue(self.inputs.ktsTorsion)
        
        self.qSpin = NoWheelDoubleSpinBox()
        self.qSpin.setRange(0.0, 1.0)
        self.qSpin.setDecimals(3)
        self.qSpin.setSingleStep(0.05)
        self.qSpin.setValue(self.inputs.notchSensQ)

        # Targets
        self.targetFatigueSpin = NoWheelDoubleSpinBox()
        self.targetFatigueSpin.setRange(0.1, 20.0)
        self.targetFatigueSpin.setDecimals(2)
        self.targetFatigueSpin.setSingleStep(0.1)
        self.targetFatigueSpin.setValue(self.inputs.targetFatigueN)

        self.targetYieldSpin = NoWheelDoubleSpinBox()
        self.targetYieldSpin.setRange(0.1, 20.0)
        self.targetYieldSpin.setDecimals(2)
        self.targetYieldSpin.setSingleStep(0.1)
        self.targetYieldSpin.setValue(self.inputs.targetYieldN)

        # Marin - UPDATED: Individual Shafts
        finishes = ["Ground", "Machined", "Hot-rolled", "As-forged"]
        
        self.finishACombo = NoWheelComboBox()
        self.finishACombo.addItems(finishes)
        self.finishACombo.setCurrentText(self.inputs.surfaceFinishA)
        
        self.finishBCombo = NoWheelComboBox()
        self.finishBCombo.addItems(finishes)
        self.finishBCombo.setCurrentText(self.inputs.surfaceFinishB)
        
        self.finishCCombo = NoWheelComboBox()
        self.finishCCombo.addItems(finishes)
        self.finishCCombo.setCurrentText(self.inputs.surfaceFinishC)

        self.reliabilityCombo = NoWheelComboBox()
        self.reliabilityCombo.addItems(["50%", "90%", "95%", "99%", "99.9%"])
        self.reliabilityCombo.setCurrentText(self.inputs.reliability)

        self.miscFactorSpin = NoWheelDoubleSpinBox()
        self.miscFactorSpin.setRange(0.1, 1.5)
        self.miscFactorSpin.setDecimals(3)
        self.miscFactorSpin.setSingleStep(0.01)
        self.miscFactorSpin.setValue(self.inputs.miscFactor)

        self.loadFactorSpin = NoWheelDoubleSpinBox()
        self.loadFactorSpin.setRange(0.2, 1.0)
        self.loadFactorSpin.setDecimals(3)
        self.loadFactorSpin.setSingleStep(0.05)
        self.loadFactorSpin.setValue(self.inputs.loadFactor)

        # Buttons
        self.recalcButton = QPushButton("Recompute")
        self.resultsButton = QPushButton("Open Results Window")

        # Layout in inputs
        self.inputsLayout.addRow(QLabel("Input torque"), self.torqueSpin)
        self.inputsLayout.addRow(QLabel("Input speed"), self.speedSpin)

        self.inputsLayout.addRow(self._divider("Gear / Helical"))
        self.inputsLayout.addRow(QLabel("Normal pressure angle"), self.phiNSpin)
        self.inputsLayout.addRow(QLabel("Helix angle"), self.psiSpin)
        self.inputsLayout.addRow(QLabel("Stage-1 ratio (d2/d3)"), self.i1Spin)
        self.inputsLayout.addRow(QLabel("Stage-2 ratio (d4/d5)"), self.i2Spin)
        self.inputsLayout.addRow(QLabel("Gear 2 diameter"), self.d2Spin)
        self.inputsLayout.addRow(QLabel("Gear 4 diameter"), self.d4Spin)
        self.inputsLayout.addRow(self.oppositeOnShaftB)
        self.inputsLayout.addRow(self.includeAxialMomentBox)

        self.inputsLayout.addRow(self._divider("Shaft baseline diameters"))
        self.inputsLayout.addRow(QLabel("Shaft A diameter"), self.dShaftASpin)
        self.inputsLayout.addRow(QLabel("Shaft B diameter"), self.dShaftBSpin)
        self.inputsLayout.addRow(QLabel("Shaft C diameter"), self.dShaftCSpin)

        self.inputsLayout.addRow(self._divider("Deflection and slope limits"))
        self.inputsLayout.addRow(QLabel("Allow deflection at gears"), self.allowDeflSpin)
        self.inputsLayout.addRow(QLabel("Allow slope at bearings"), self.allowSlopeSpin)

        self.inputsLayout.addRow(self._divider("Material / fatigue"))
        self.inputsLayout.addRow(QLabel("Elastic modulus E"), self.elasticModulusSpin)
        self.inputsLayout.addRow(QLabel("Ultimate strength Sut"), self.sutSpin)
        self.inputsLayout.addRow(QLabel("Yield strength Sy"), self.sySpin)
        self.inputsLayout.addRow(QLabel("Kt (Theor. Bending)"), self.ktSpin)
        self.inputsLayout.addRow(QLabel("Kts (Theor. Torsion)"), self.ktsSpin)
        self.inputsLayout.addRow(QLabel("q (Notch Sensitivity)"), self.qSpin)

        self.inputsLayout.addRow(self._divider("Marin factors"))
        self.inputsLayout.addRow(QLabel("Surf. Finish (Shaft A)"), self.finishACombo)
        self.inputsLayout.addRow(QLabel("Surf. Finish (Shaft B)"), self.finishBCombo)
        self.inputsLayout.addRow(QLabel("Surf. Finish (Shaft C)"), self.finishCCombo)
        self.inputsLayout.addRow(QLabel("Reliability"), self.reliabilityCombo)
        self.inputsLayout.addRow(QLabel("Load factor kc"), self.loadFactorSpin)
        self.inputsLayout.addRow(QLabel("Misc factor k_misc"), self.miscFactorSpin)

        self.inputsLayout.addRow(self._divider("Safety factor targets"))
        self.inputsLayout.addRow(QLabel("Target fatigue n"), self.targetFatigueSpin)
        self.inputsLayout.addRow(QLabel("Target yield n"), self.targetYieldSpin)

        self.inputsLayout.addRow(self.recalcButton)
        self.inputsLayout.addRow(self.resultsButton)

    def _divider(self, text: str) -> QLabel:
        lbl = QLabel(f"<b>{text}</b>")
        return lbl

    def _buildTabs(self):
        self.schematicCanvas = PlotCanvas(
            title="Schematic (side view)",
            xlabel="global x (mm)",
            ylabel="y (mm)",
            heightInches=5.2,
        )
        self.tabs.addTab(self._wrapInScroll(self._simpleContainer(self.schematicCanvas)), "Schematic")

        self.shaftACanvases = self._createShaftTab("Shaft A")
        self.shaftBCanvases = self._createShaftTab("Shaft B")
        self.shaftCCanvases = self._createShaftTab("Shaft C")

        self.tabs.addTab(self.shaftACanvases["scroll"], "Shaft A")
        self.tabs.addTab(self.shaftBCanvases["scroll"], "Shaft B")
        self.tabs.addTab(self.shaftCCanvases["scroll"], "Shaft C")

        self.fatigueTab = self._createFatigueYieldTab()
        self.deflectionTab = self._createDeflectionSlopeTab()
        self.finalDesignTab = self._createFinalDesignTab()

        self.tabs.addTab(self.fatigueTab["scroll"], "Fatigue and Yield")
        self.tabs.addTab(self.deflectionTab["scroll"], "Deflection and Slope")
        self.tabs.addTab(self.finalDesignTab["scroll"], "Final Shaft Design")

    def _simpleContainer(self, widget: QWidget) -> QWidget:
        c = QWidget()
        v = QVBoxLayout(c)
        v.setContentsMargins(10, 10, 10, 10)
        v.addWidget(widget)
        return c

    def _createShaftTab(self, shaftName: str):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(14)

        canvTanV = PlotCanvas(title=f"{shaftName} - Tangential - Shear", xlabel="x (mm)", ylabel="V (N)", heightInches=2.6)
        canvTanM = PlotCanvas(title=f"{shaftName} - Tangential - Moment", xlabel="x (mm)", ylabel="M (Nmm)", heightInches=2.6)
        canvRadV = PlotCanvas(title=f"{shaftName} - Radial - Shear", xlabel="x (mm)", ylabel="V (N)", heightInches=2.6)
        canvRadM = PlotCanvas(title=f"{shaftName} - Radial - Moment", xlabel="x (mm)", ylabel="M (Nmm)", heightInches=2.6)

        layout.addWidget(canvTanV)
        layout.addWidget(canvTanM)
        layout.addWidget(canvRadV)
        layout.addWidget(canvRadM)

        scroll = self._wrapInScroll(container)
        return {
            "scroll": scroll,
            "tanV": canvTanV,
            "tanM": canvTanM,
            "radV": canvRadV,
            "radM": canvRadM,
        }

    def _createFatigueYieldTab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(14)

        intro = QLabel(
            "<b>Identify critical locations + Fatigue and Yield Criteria</b><br>"
            "This page uses: Distortion-Energy (von Mises) + Gerber fatigue criterion.<br>"
            "Marin factors are computed automatically (Se' -> Se).<br>"
            "Kf is calculated from Kt and q: Kf = 1 + q*(Kt-1)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        marinGroup = QGroupBox("Marin factors breakdown (per shaft diameter)")
        marinLayout = QVBoxLayout(marinGroup)
        self.marinTable = QTableWidget()
        self.marinTable.setColumnCount(10) # Added column for surface finish name
        self.marinTable.setHorizontalHeaderLabels([
            "Shaft", "Finish", "d (mm)", "Se' (MPa)", "ka", "kb", "kc", "ke", "k_misc", "Se (MPa)"
        ])
        _optimizeTableLayout(self.marinTable)
        marinLayout.addWidget(self.marinTable)
        layout.addWidget(marinGroup)

        critGroup = QGroupBox("Critical locations (auto list)")
        critLayout = QVBoxLayout(critGroup)
        self.criticalLocTable = QTableWidget()
        self.criticalLocTable.setColumnCount(6)
        self.criticalLocTable.setHorizontalHeaderLabels([
            "Shaft", "Location", "x (mm)", "d (mm)", "Mres (Nmm)", "Torque (Nmm)"
        ])
        _optimizeTableLayout(self.criticalLocTable)
        critLayout.addWidget(self.criticalLocTable)
        layout.addWidget(critGroup)

        tableGroup = QGroupBox("Fatigue and Yield results (DE-Gerber)")
        tableLayout = QVBoxLayout(tableGroup)
        self.fatigueYieldTable = QTableWidget()
        self.fatigueYieldTable.setColumnCount(12)
        self.fatigueYieldTable.setHorizontalHeaderLabels([
            "Shaft", "Location", "x (mm)", "d (mm)", "Se (MPa)",
            "sigma_a (MPa)", "sigma_m (MPa)", "n_fatigue", "sigma_vm (MPa)", "n_yield",
            "Fatigue OK", "Yield OK",
        ])
        _optimizeTableLayout(self.fatigueYieldTable)
        tableLayout.addWidget(self.fatigueYieldTable)
        layout.addWidget(tableGroup)

        scroll = self._wrapInScroll(container)
        return {"scroll": scroll}

    def _createDeflectionSlopeTab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(14)

        intro = QLabel(
            "<b>Deflection and Slope Checks</b><br>"
            "Checks: deflection at gear locations and slope at bearings (resultant).\n"
            "Default limits are editable in the left panel."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        tableGroup = QGroupBox("Summary")
        tableLayout = QVBoxLayout(tableGroup)
        self.deflectionTable = QTableWidget()
        self.deflectionTable.setColumnCount(8)
        self.deflectionTable.setHorizontalHeaderLabels([
            "Shaft", "Gear x\n(mm)", "|y|@gear\n(mm)", "|y|max\n(mm)",
            "|theta|@left\n(rad)", "|theta|@right\n(rad)", "Allow y\n(mm)", "Allow theta\n(rad)"
        ])
        _optimizeTableLayout(self.deflectionTable)
        tableLayout.addWidget(self.deflectionTable)
        layout.addWidget(tableGroup)

        plotsGroup = QGroupBox("Deflection and slope plots (resultant)")
        plotsLayout = QVBoxLayout(plotsGroup)

        self.deflPlotA = PlotCanvas("Shaft A - Deflection", "x (mm)", "y (mm)", heightInches=2.6)
        self.slopePlotA = PlotCanvas("Shaft A - Slope", "x (mm)", "theta (rad)", heightInches=2.6)
        self.deflPlotB = PlotCanvas("Shaft B - Deflection", "x (mm)", "y (mm)", heightInches=2.6)
        self.slopePlotB = PlotCanvas("Shaft B - Slope", "x (mm)", "theta (rad)", heightInches=2.6)
        self.deflPlotC = PlotCanvas("Shaft C - Deflection", "x (mm)", "y (mm)", heightInches=2.6)
        self.slopePlotC = PlotCanvas("Shaft C - Slope", "x (mm)", "theta (rad)", heightInches=2.6)

        plotsLayout.addWidget(self.deflPlotA)
        plotsLayout.addWidget(self.slopePlotA)
        plotsLayout.addWidget(self.deflPlotB)
        plotsLayout.addWidget(self.slopePlotB)
        plotsLayout.addWidget(self.deflPlotC)
        plotsLayout.addWidget(self.slopePlotC)

        layout.addWidget(plotsGroup)
        return {"scroll": self._wrapInScroll(container)}

    def _createFinalDesignTab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(14)

        intro = QLabel(
            "<b>Final Shaft Design</b><br>"
            "Select shaft diameters, fillet radii, shoulders, key sizes and groove locations.<br>"
            "If deflection or stress limits are exceeded, the tool suggests a larger diameter." 
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        group = QGroupBox("Recommended design updates")
        v = QVBoxLayout(group)
        self.finalDesignTable = QTableWidget()
        self.finalDesignTable.setColumnCount(10)
        self.finalDesignTable.setHorizontalHeaderLabels([
            "Shaft", "d_now\n(mm)", "d_req\n(stress)", "d_req\n(defl)",
            "d_req\n(slope)", "d_rec\n(mm)", "Fillet r\n(mm)", "Shoulder\nD/d",
            "Key\nSize", "Status"
        ])
        _optimizeTableLayout(self.finalDesignTable)
        v.addWidget(self.finalDesignTable)
        layout.addWidget(group)

        notes = QLabel(
            "<span style='color:#8b949e'>Notes:</span><br>"
            "- Fillet radius suggestion: ~3% of diameter.<br>"
            "- Shoulder D/d: simple 1.2 placeholder (you can adjust in CAD).<br>"
            "- Key suggestion: quick metric mapping by shaft diameter.<br>"
            "- Groove/relief: place at each shoulder + keyway ends." 
        )
        notes.setWordWrap(True)
        layout.addWidget(notes)

        return {"scroll": self._wrapInScroll(container)}

    def _applyDarkTheme(self):
        self.setStyleSheet(
            """
            QWidget { background-color: #0d1117; color: #c9d1d9; font-size: 12px; }
            QGroupBox { border: 1px solid #30363d; border-radius: 10px; margin-top: 12px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 2px 6px; color: #e6edf3; }
            QPushButton { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px 12px; }
            QPushButton:hover { background-color: #1f2630; }
            QPushButton:disabled { color: #7d8590; }
            QTabWidget::pane { border: 1px solid #30363d; border-radius: 10px; padding: 4px; }
            QTabBar::tab { background: #161b22; border: 1px solid #30363d; border-bottom: none; padding: 8px 14px; margin-right: 4px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
            QTabBar::tab:selected { background: #1f2630; color: #ffffff; }
            QDoubleSpinBox { background-color: #0b0f15; border: 1px solid #30363d; border-radius: 8px; padding: 6px; min-height: 28px; }
            QComboBox { background-color: #0b0f15; border: 1px solid #30363d; border-radius: 8px; padding: 6px; min-height: 28px; }
            QCheckBox { padding: 4px; }
            QScrollArea { background-color: transparent; }
            QScrollBar:vertical {
                background: #0d1117;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                min-height: 20px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #0d1117;
                height: 14px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #30363d;
                min-width: 20px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            """
        )

    def _connectSignals(self):
        valueWidgets = [
            self.torqueSpin, self.speedSpin, self.phiNSpin, self.psiSpin,
            self.i1Spin, self.i2Spin, self.d2Spin, self.d4Spin,
            self.dShaftASpin, self.dShaftBSpin, self.dShaftCSpin,
            self.allowDeflSpin, self.allowSlopeSpin,
            self.elasticModulusSpin, self.sutSpin, self.sySpin,
            self.ktSpin, self.ktsSpin, self.qSpin,
            self.targetFatigueSpin, self.targetYieldSpin,
            self.miscFactorSpin, self.loadFactorSpin,
        ]
        for w in valueWidgets:
            w.valueChanged.connect(self._markDirty)

        self.finishACombo.currentIndexChanged.connect(self._markDirty)
        self.finishBCombo.currentIndexChanged.connect(self._markDirty)
        self.finishCCombo.currentIndexChanged.connect(self._markDirty)
        self.reliabilityCombo.currentIndexChanged.connect(self._markDirty)
        self.oppositeOnShaftB.stateChanged.connect(self._markDirty)
        self.includeAxialMomentBox.stateChanged.connect(self._markDirty)

        self.recalcButton.clicked.connect(self._recomputeAndRedraw)
        self.resultsButton.clicked.connect(self._openResultsWindow)
        
        # New Reset Button Signal
        self.resetButton.clicked.connect(self._resetHighlights)

    # ----------------------------
    # Dirty tracking
    # ----------------------------

    def _currentInputSignature(self) -> tuple:
        sig = (
            round(self.torqueSpin.value(), 6),
            round(self.speedSpin.value(), 6),
            round(self.phiNSpin.value(), 6),
            round(self.psiSpin.value(), 6),
            round(self.i1Spin.value(), 6),
            round(self.i2Spin.value(), 6),
            round(self.d2Spin.value(), 6),
            round(self.d4Spin.value(), 6),
            round(self.dShaftASpin.value(), 6),
            round(self.dShaftBSpin.value(), 6),
            round(self.dShaftCSpin.value(), 6),
            round(self.allowDeflSpin.value(), 8),
            round(self.allowSlopeSpin.value(), 10),
            round(self.elasticModulusSpin.value(), 6),
            round(self.sutSpin.value(), 6),
            round(self.sySpin.value(), 6),
            round(self.ktSpin.value(), 6),
            round(self.ktsSpin.value(), 6),
            round(self.qSpin.value(), 6),
            round(self.targetFatigueSpin.value(), 6),
            round(self.targetYieldSpin.value(), 6),
            self.finishACombo.currentText(),
            self.finishBCombo.currentText(),
            self.finishCCombo.currentText(),
            self.reliabilityCombo.currentText(),
            round(self.miscFactorSpin.value(), 6),
            round(self.loadFactorSpin.value(), 6),
            bool(self.oppositeOnShaftB.isChecked()),
            bool(self.includeAxialMomentBox.isChecked()),
        )
        return sig

    def _markDirty(self):
        currentSig = self._currentInputSignature()
        dirty = currentSig != getattr(self, "_lastInputSignature", None)
        self.isDirty = dirty
        self.recalcButton.setEnabled(dirty)

    # ----------------------------
    # Computation
    # ----------------------------

    def _readInputs(self) -> GearboxInputs:
        i = GearboxInputs()
        i.inputTorqueNm = self.torqueSpin.value()
        i.inputSpeedRpm = self.speedSpin.value()
        i.normalPressureAngleDeg = self.phiNSpin.value()
        i.helixAngleDeg = self.psiSpin.value()
        i.stage1Ratio = max(1.0, self.i1Spin.value())
        i.stage2Ratio = max(1.0, self.i2Spin.value())
        i.gear2DiameterMm = max(1.0, self.d2Spin.value())
        i.gear4DiameterMm = max(1.0, self.d4Spin.value())

        i.shaftADiameterMm = max(1.0, self.dShaftASpin.value())
        i.shaftBDiameterMm = max(1.0, self.dShaftBSpin.value())
        i.shaftCDiameterMm = max(1.0, self.dShaftCSpin.value())

        i.allowDeflectionMm = max(0.0, self.allowDeflSpin.value())
        i.allowSlopeRad = max(0.0, self.allowSlopeSpin.value())

        i.elasticModulusMpa = max(1.0, self.elasticModulusSpin.value())
        i.sutMpa = max(1.0, self.sutSpin.value())
        i.syMpa = max(1.0, self.sySpin.value())

        i.ktBending = max(1.0, self.ktSpin.value())
        i.ktsTorsion = max(1.0, self.ktsSpin.value())
        i.notchSensQ = max(0.0, min(1.0, self.qSpin.value()))

        i.targetFatigueN = max(0.1, self.targetFatigueSpin.value())
        i.targetYieldN = max(0.1, self.targetYieldSpin.value())

        i.surfaceFinishA = self.finishACombo.currentText()
        i.surfaceFinishB = self.finishBCombo.currentText()
        i.surfaceFinishC = self.finishCCombo.currentText()
        
        i.reliability = self.reliabilityCombo.currentText()
        i.miscFactor = max(0.1, self.miscFactorSpin.value())
        i.loadFactor = max(0.2, min(1.0, self.loadFactorSpin.value()))
        
        i.includeAxialMoment = self.includeAxialMomentBox.isChecked()
        
        return i

    def _recomputeAndRedraw(self):
        try:
            self.inputs = self._readInputs()
            results, shaftData = self._computeAll(self.inputs)

            self.lastResults = results
            self.lastShaftData = shaftData

            self._updateSchematic(results)
            self._updateShaftPlots(shaftData)
            self._updateFatigueYield(results, shaftData)
            self._updateDeflection(results, shaftData)
            self._updateFinalDesign(results, shaftData)

            self._lastInputSignature = self._currentInputSignature()
            self.isDirty = False
            self.recalcButton.setEnabled(False)
            
            # Check for changes to enable the reset button
            self._updateResetButtonState()

        except Exception as e:
            QMessageBox.critical(self, "Compute Error", f"{type(e).__name__}: {e}")
            
    def _updateResetButtonState(self):
        """Scans tables for any highlighted (orange) items."""
        tables_to_check = [
            self.marinTable,
            self.criticalLocTable,
            self.fatigueYieldTable,
            self.deflectionTable,
            self.finalDesignTable
        ]
        
        has_highlights = False
        for table in tables_to_check:
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    item = table.item(r, c)
                    # Check if item exists and foreground is Orange
                    if item and item.foreground().color().name() == "#ffb300":
                        has_highlights = True
                        break
                if has_highlights: break
            if has_highlights: break
            
        self.resetButton.setEnabled(has_highlights)
            
    def _resetHighlights(self):
        """Forcefully clears highlights from all tables and disables button."""
        tables_to_clear = [
            self.marinTable,
            self.criticalLocTable,
            self.fatigueYieldTable,
            self.deflectionTable,
            self.finalDesignTable
        ]
        
        for table in tables_to_clear:
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    item = table.item(r, c)
                    if item:
                        # Reset to default gray color and normal font weight
                        item.setForeground(QColor("#c9d1d9")) 
                        font = item.font()
                        font.setBold(False)
                        item.setFont(font)
        
        # After clearing, disable the button
        self.resetButton.setEnabled(False)

    def _computeAll(self, i: GearboxInputs):
        # Derived diameters
        d2 = i.gear2DiameterMm
        d4 = i.gear4DiameterMm
        d3 = d2 / i.stage1Ratio
        d5 = d4 / i.stage2Ratio

        # Torques
        T_a = i.inputTorqueNm
        T_b = T_a / i.stage1Ratio
        T_c = T_b / i.stage2Ratio

        # Mesh forces
        Ft23, Fr23, Fa23, phiT = meshForces(T_a, d2, i.normalPressureAngleDeg, i.helixAngleDeg)
        Ft45, Fr45, Fa45, _ = meshForces(T_b, d4, i.normalPressureAngleDeg, i.helixAngleDeg)

        # Overturning moments
        moment_factor = 1.0 if i.includeAxialMoment else 0.0
        Mover2 = (Fa23 * (d2 / 2.0)) * moment_factor
        Mover3 = (Fa23 * (d3 / 2.0)) * moment_factor
        Mover4 = (Fa45 * (d4 / 2.0)) * moment_factor
        Mover5 = (Fa45 * (d5 / 2.0)) * moment_factor

        # Geometry
        LA = i.shaftALengthMm
        LB = i.shaftBLengthMm
        LC = i.shaftCLengthMm

        x2 = i.gear2PosFromAmm
        x3 = i.gear3PosFromCmm
        x4 = i.gear4PosFromCmm
        x5 = i.gear5PosFromEmm

        oppose = self.oppositeOnShaftB.isChecked()

        # Shaft A loads
        loadsA_t = [(x2, -Ft23)]
        loadsA_r = [(x2, -Fr23)]
        momsA_r = [(x2, -Mover2)]

        RA_t = multiLoadReactions(LA, loadsA_t)
        RA_r = multiLoadReactions(LA, loadsA_r, momsA_r)

        xA_t, VA_t, MA_t = buildShearMoment(LA, RA_t, loadsA_t)
        xA_r, VA_r, MA_r = buildShearMoment(LA, RA_r, loadsA_r, momsA_r)

        # Shaft B loads
        # Radial (separating) forces point toward the shaft axis, so they oppose when
        # the two meshes sit on opposite sides of Shaft B. Torque balance
        # (Ft23 * d3/2 = Ft45 * d4/2) then requires the tangential forces to act in
        # the same direction; on the same side both relations flip.
        radialSignStage2 = +1.0 if oppose else -1.0
        tangentialSignStage2 = -radialSignStage2
        loadsB_t = [(x3, -Ft23), (x4, tangentialSignStage2 * Ft45)]
        loadsB_r = [(x3, -Fr23), (x4, radialSignStage2 * Fr45)]
        momsB_r = [(x3, -Mover3), (x4, radialSignStage2 * Mover4)]

        RB_t = multiLoadReactions(LB, loadsB_t)
        RB_r = multiLoadReactions(LB, loadsB_r, momsB_r)

        xB_t, VB_t, MB_t = buildShearMoment(LB, RB_t, loadsB_t)
        xB_r, VB_r, MB_r = buildShearMoment(LB, RB_r, loadsB_r, momsB_r)

        # Shaft C loads
        loadsC_t = [(x5, -Ft45)]
        loadsC_r = [(x5, -Fr45)]
        momsC_r = [(x5, -Mover5)]

        RC_t = multiLoadReactions(LC, loadsC_t)
        RC_r = multiLoadReactions(LC, loadsC_r, momsC_r)

        xC_t, VC_t, MC_t = buildShearMoment(LC, RC_t, loadsC_t)
        xC_r, VC_r, MC_r = buildShearMoment(LC, RC_r, loadsC_r, momsC_r)

        # Resultant moments
        MA_res = np.sqrt(MA_t**2 + MA_r**2)
        MB_res = np.sqrt(MB_t**2 + MB_r**2)
        MC_res = np.sqrt(MC_t**2 + MC_r**2)

        # Deflection + slope
        yA_t, thA_t = computeDeflectionAndSlope(xA_t, MA_t, i.elasticModulusMpa, i.shaftADiameterMm)
        yA_r, thA_r = computeDeflectionAndSlope(xA_r, MA_r, i.elasticModulusMpa, i.shaftADiameterMm)
        yA = np.sqrt(yA_t**2 + yA_r**2)
        thA = np.sqrt(thA_t**2 + thA_r**2)

        yB_t, thB_t = computeDeflectionAndSlope(xB_t, MB_t, i.elasticModulusMpa, i.shaftBDiameterMm)
        yB_r, thB_r = computeDeflectionAndSlope(xB_r, MB_r, i.elasticModulusMpa, i.shaftBDiameterMm)
        yB = np.sqrt(yB_t**2 + yB_r**2)
        thB = np.sqrt(thB_t**2 + thB_r**2)

        yC_t, thC_t = computeDeflectionAndSlope(xC_t, MC_t, i.elasticModulusMpa, i.shaftCDiameterMm)
        yC_r, thC_r = computeDeflectionAndSlope(xC_r, MC_r, i.elasticModulusMpa, i.shaftCDiameterMm)
        yC = np.sqrt(yC_t**2 + yC_r**2)
        thC = np.sqrt(thC_t**2 + thC_r**2)

        results = {
            "phiT_deg": phiT,
            "d2": d2,
            "d3": d3,
            "d4": d4,
            "d5": d5,
            "T_a": T_a,
            "T_b": T_b,
            "T_c": T_c,
            "mesh23": {"Ft": Ft23, "Fr": Fr23, "Fa": Fa23, "Mover2": Mover2, "Mover3": Mover3},
            "mesh45": {"Ft": Ft45, "Fr": Fr45, "Fa": Fa45, "Mover4": Mover4, "Mover5": Mover5},
            "geometry": {"LA": LA, "LB": LB, "LC": LC, "x2": x2, "x3": x3, "x4": x4, "x5": x5},
            "reactions": {"A_t": RA_t, "A_r": RA_r, "B_t": RB_t, "B_r": RB_r, "C_t": RC_t, "C_r": RC_r},
        }

        shaftData = {
            "A": {"x_t": xA_t, "V_t": VA_t, "M_t": MA_t, "x_r": xA_r, "V_r": VA_r, "M_r": MA_r, "M_res": MA_res, "y": yA, "th": thA, "moms_r": momsA_r},
            "B": {"x_t": xB_t, "V_t": VB_t, "M_t": MB_t, "x_r": xB_r, "V_r": VB_r, "M_r": MB_r, "M_res": MB_res, "y": yB, "th": thB, "moms_r": momsB_r},
            "C": {"x_t": xC_t, "V_t": VC_t, "M_t": MC_t, "x_r": xC_r, "V_r": VC_r, "M_r": MC_r, "M_res": MC_res, "y": yC, "th": thC, "moms_r": momsC_r},
        }

        return results, shaftData

    # ----------------------------
    # Updates
    # ----------------------------

    def _updateSchematic(self, results: dict):
        ax = self.schematicCanvas.ax
        self.schematicCanvas.clear("Schematic (side view)")

        d2 = results["d2"]
        d3 = results["d3"]
        d4 = results["d4"]
        d5 = results["d5"]

        geo = results["geometry"]
        LA, LB, LC = geo["LA"], geo["LB"], geo["LC"]
        x2, x3, x4, x5 = geo["x2"], geo["x3"], geo["x4"], geo["x5"]

        a23 = (d2 + d3) / 2.0
        a45 = (d4 + d5) / 2.0

        xA0 = 0.0
        xB0 = 0.0
        xC0 = (xB0 + x4) - x5

        yA = +a23
        yB = 0.0
        yC = -a45

        colorA = '#58a6ff' 
        colorB = '#f0883e' 
        colorC = '#3fb950' 

        ax.plot([xA0, xA0 + LA], [yA, yA], linewidth=4, color=colorA, label="Shaft A")
        ax.plot([xB0, xB0 + LB], [yB, yB], linewidth=4, color=colorB, label="Shaft B")
        ax.plot([xC0, xC0 + LC], [yC, yC], linewidth=4, color=colorC, label="Shaft C")

        ax.scatter([xA0, xA0 + LA], [yA, yA], s=120, marker="s", color='#8b949e')
        ax.scatter([xB0, xB0 + LB], [yB, yB], s=120, marker="s", color='#8b949e')
        ax.scatter([xC0, xC0 + LC], [yC, yC], s=120, marker="s", color='#8b949e')

        fwStage1 = 80.0
        fwStage2 = 100.0

        def drawGear(centerX, centerY, diameter, faceW, label):
            leftX = centerX - faceW / 2.0
            bottomY = centerY - diameter / 2.0
            rect = matplotlib.patches.Rectangle(
                (leftX, bottomY),
                faceW,
                diameter,
                fill=False,
                linewidth=2.0,
                edgecolor='#d2a8ff',
                zorder=10
            )
            ax.add_patch(rect)
            ax.text(centerX, centerY, label, ha="center", va="center", fontsize=12, fontweight="bold", color='#ffffff', zorder=12)

        def drawHub(centerX, centerY, diameter):
            hubW = fwStage1 + 40.0 if diameter < 200 else fwStage2 + 40.0
            hubD = diameter * 0.4
            leftX = centerX - hubW/2.0
            bottomY = centerY - hubD/2.0
            rect = matplotlib.patches.Rectangle(
                (leftX, bottomY),
                hubW,
                hubD,
                fill=True,
                facecolor='#21262d',
                edgecolor='#8b949e',
                linewidth=1.0,
                linestyle='--',
                zorder=5
            )
            ax.add_patch(rect)

        drawHub(xA0 + x2, yA, d2)
        drawHub(xC0 + x5, yC, d5)

        drawGear(xA0 + x2, yA, d2, fwStage1, "G2")
        drawGear(xB0 + x3, yB, d3, fwStage1, "G3")
        drawGear(xB0 + x4, yB, d4, fwStage2, "G4")
        drawGear(xC0 + x5, yC, d5, fwStage2, "G5")

        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), 
                           frameon=True, facecolor='#161b22', edgecolor='#30363d')
        
        for text in legend.get_texts():
            text.set_color('#c9d1d9')

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        xMin = min(xA0, xB0, xC0) - 80
        xMax = max(xA0 + LA, xB0 + LB, xC0 + LC) + 80
        yMin = yC - d4 * 0.7
        yMax = yA + d2 * 0.7
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)

        self.schematicCanvas.figure.tight_layout(rect=[0, 0, 0.85, 1])
        self.schematicCanvas.draw()

    def _plotShearMoment(self, canvas: PlotCanvas, x: np.ndarray, y: np.ndarray, labelMax: bool = True):
        ax = canvas.ax
        ax.plot(x, y, linewidth=2.0, color='#3fb950', drawstyle='steps-post')
        PlotCanvas.niceYLim(ax, y)

        if labelMax and len(x) > 2:
            yMaxVal = np.max(y)
            idxMax = int(np.argmax(y))
            xMaxVal = x[idxMax]
            
            yMinVal = np.min(y)
            idxMin = int(np.argmin(y))
            xMinVal = x[idxMin]

            ax.scatter([xMaxVal], [yMaxVal], s=40, color='#f0883e', zorder=5)
            ax.annotate(
                f"{yMaxVal:.1f}",
                xy=(xMaxVal, yMaxVal),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, color='#c9d1d9', fontweight='bold'
            )

            ax.scatter([xMinVal], [yMinVal], s=40, color='#f0883e', zorder=5)
            ax.annotate(
                f"{yMinVal:.1f}",
                xy=(xMinVal, yMinVal),
                xytext=(0, -15), textcoords="offset points",
                ha="center", va="top",
                fontsize=9, color='#c9d1d9', fontweight='bold'
            )

        canvas.figure.tight_layout()
        canvas.draw()

    def _updateShaftPlots(self, shaftData: dict):
        mapping = {
            "A": self.shaftACanvases,
            "B": self.shaftBCanvases,
            "C": self.shaftCCanvases,
        }
        for key, canv in mapping.items():
            data = shaftData[key]

            canv["tanV"].clear(canv["tanV"].ax.get_title())
            self._plotShearMoment(canv["tanV"], data["x_t"], data["V_t"], labelMax=True)

            canv["tanM"].clear(canv["tanM"].ax.get_title())
            self._plotShearMoment(canv["tanM"], data["x_t"], data["M_t"], labelMax=True)

            canv["radV"].clear(canv["radV"].ax.get_title())
            self._plotShearMoment(canv["radV"], data["x_r"], data["V_r"], labelMax=True)

            canv["radM"].clear(canv["radM"].ax.get_title())
            self._plotShearMoment(canv["radM"], data["x_r"], data["M_r"], labelMax=True)

    def _updateFatigueYield(self, results: dict, shaftData: dict):
        i = self.inputs

        # Calculate Kf and Kfs from Kt/Kts and q
        # Kf = 1 + q(Kt - 1)
        kfCalc = 1.0 + i.notchSensQ * (i.ktBending - 1.0)
        kfsCalc = 1.0 + i.notchSensQ * (i.ktsTorsion - 1.0)

        prev_rows = self.previous_table_state.get("marin", [])
        
        # Individual shaft params
        shafts = [
            ("A", i.shaftADiameterMm, i.surfaceFinishA),
            ("B", i.shaftBDiameterMm, i.surfaceFinishB),
            ("C", i.shaftCDiameterMm, i.surfaceFinishC)
        ]

        rowsMarin = []
        for name, d, finish in shafts:
            mf = correctedEnduranceLimitMpa(
                sutMpa=i.sutMpa,
                dMm=d,
                surfaceFinish=finish,
                reliability=i.reliability,
                kLoad=i.loadFactor,
                kMisc=i.miscFactor,
            )
            rowsMarin.append([
                name,
                finish,
                f"{d:.2f}",
                f"{mf['SePrime']:.1f}",
                f"{mf['ka']:.3f}",
                f"{mf['kb']:.3f}",
                f"{mf['kc']:.3f}",
                f"{mf['ke']:.3f}",
                f"{mf['kMisc']:.3f}",
                f"{mf['Se']:.1f}",
            ])

        self.marinTable.setRowCount(len(rowsMarin))
        for r, row in enumerate(rowsMarin):
            for c, val in enumerate(row):
                is_changed = False
                if prev_rows and r < len(prev_rows) and c < len(prev_rows[r]):
                    if prev_rows[r][c] != val:
                        is_changed = True
                
                self.marinTable.setItem(r, c, _tableItem(val, highlight=is_changed))
        
        self.previous_table_state["marin"] = rowsMarin

        # Critical locations
        geo = results["geometry"]
        crit = []

        TA = results["T_a"] * 1000.0
        TB = results["T_b"] * 1000.0
        TC = results["T_c"] * 1000.0

        def interpMres(shaftKey: str, xPos: float) -> float:
            return resultantMomentAt(shaftData[shaftKey], xPos)

        crit.append(("A", "Gear 2 seat/keyway", geo["x2"], i.shaftADiameterMm, interpMres("A", geo["x2"]), TA))
        crit.append(("A", "Bearing A shoulder", 0.0, i.shaftADiameterMm, interpMres("A", 0.0), TA))
        crit.append(("A", "Bearing B shoulder", geo["LA"], i.shaftADiameterMm, interpMres("A", geo["LA"]), TA))

        crit.append(("B", "Gear 3 seat/keyway", geo["x3"], i.shaftBDiameterMm, interpMres("B", geo["x3"]), TB))
        crit.append(("B", "Gear 4 seat/keyway", geo["x4"], i.shaftBDiameterMm, interpMres("B", geo["x4"]), TB))
        crit.append(("B", "Bearing C shoulder", 0.0, i.shaftBDiameterMm, interpMres("B", 0.0), TB))
        crit.append(("B", "Bearing D shoulder", geo["LB"], i.shaftBDiameterMm, interpMres("B", geo["LB"]), TB))

        crit.append(("C", "Gear 5 seat/keyway", geo["x5"], i.shaftCDiameterMm, interpMres("C", geo["x5"]), TC))
        crit.append(("C", "Bearing E shoulder", 0.0, i.shaftCDiameterMm, interpMres("C", 0.0), TC))
        crit.append(("C", "Bearing F shoulder", geo["LC"], i.shaftCDiameterMm, interpMres("C", geo["LC"]), TC))

        prev_crit = self.previous_table_state.get("crit", [])
        self.criticalLocTable.setRowCount(len(crit))
        current_crit_rows = []
        
        for r, row in enumerate(crit):
            row_vals = []
            for c, val in enumerate(row):
                str_val = str(val)
                if isinstance(val, float):
                    if c in (2, 3):
                        str_val = f"{val:.2f}"
                    else:
                        str_val = f"{val:.1f}"
                row_vals.append(str_val)
                
                is_changed = False
                if prev_crit and r < len(prev_crit) and c < len(prev_crit[r]):
                    if prev_crit[r][c] != str_val:
                        is_changed = True

                self.criticalLocTable.setItem(r, c, _tableItem(str_val, highlight=is_changed))
            current_crit_rows.append(row_vals)
        
        self.previous_table_state["crit"] = current_crit_rows

        # Fatigue computations
        fatigueRows = []
        
        # Map shaft name to surface finish
        finishMap = {
            "A": i.surfaceFinishA,
            "B": i.surfaceFinishB,
            "C": i.surfaceFinishC
        }

        for (shaftName, loc, xPos, dMm, mRes, tNmm) in crit:
            mf = correctedEnduranceLimitMpa(
                sutMpa=i.sutMpa,
                dMm=dMm,
                surfaceFinish=finishMap[shaftName],
                reliability=i.reliability,
                kLoad=i.loadFactor,
                kMisc=i.miscFactor,
            )
            Se = mf["Se"]

            res = equivalentStressDEGerber(
                MresNmm=mRes,
                torqueNmm=tNmm,
                dMm=dMm,
                sutMpa=i.sutMpa,
                syMpa=i.syMpa,
                SeMpa=Se,
                kfBending=kfCalc,
                kfsTorsion=kfsCalc,
            )

            okFat = "OK" if res["nFatigue"] >= i.targetFatigueN else "FAIL"
            okY = "OK" if res["nYield"] >= i.targetYieldN else "FAIL"

            fatigueRows.append([
                shaftName,
                loc,
                f"{xPos:.1f}",
                f"{dMm:.2f}",
                f"{Se:.1f}",
                f"{res['sigmaEq_a']:.1f}",
                f"{res['sigmaEq_m']:.1f}",
                f"{res['nFatigue']:.2f}",
                f"{res['sigmaVM']:.1f}",
                f"{res['nYield']:.2f}",
                okFat,
                okY,
            ])

        prev_fat = self.previous_table_state.get("fatigue", [])
        self.fatigueYieldTable.setRowCount(len(fatigueRows))
        
        for r, row in enumerate(fatigueRows):
            for c, val in enumerate(row):
                is_changed = False
                if prev_fat and r < len(prev_fat) and c < len(prev_fat[r]):
                    if prev_fat[r][c] != val:
                        is_changed = True
                
                is_bold_col = (c >= 10)
                self.fatigueYieldTable.setItem(r, c, _tableItem(val, highlight=is_changed, bold=is_bold_col))
        
        self.previous_table_state["fatigue"] = fatigueRows

    def _updateDeflection(self, results: dict, shaftData: dict):
        i = self.inputs
        geo = results["geometry"]

        def summarize(shaftKey: str, gearX: float):
            x = shaftData[shaftKey]["x_t"]
            y = shaftData[shaftKey]["y"]
            th = shaftData[shaftKey]["th"]
            yGear = float(np.interp(gearX, x, y))
            yMax = float(np.max(np.abs(y)))
            thLeft = float(abs(th[0]))
            thRight = float(abs(th[-1]))
            return yGear, yMax, thLeft, thRight

        rows = []
        yA_g, yA_m, thA_l, thA_r = summarize("A", geo["x2"])
        yB_g3, yB_m, thB_l, thB_r = summarize("B", geo["x3"])
        yB_g4 = float(np.interp(geo["x4"], shaftData["B"]["x_t"], shaftData["B"]["y"]))
        yB_g = max(abs(yB_g3), abs(yB_g4))

        yC_g, yC_m, thC_l, thC_r = summarize("C", geo["x5"])

        rows.append(["A", f"{geo['x2']:.1f}", f"{abs(yA_g):.4f}", f"{yA_m:.4f}", f"{thA_l:.6f}", f"{thA_r:.6f}", f"{i.allowDeflectionMm:.4f}", f"{i.allowSlopeRad:.6f}"])
        rows.append(["B", f"{geo['x3']:.1f}/{geo['x4']:.1f}", f"{abs(yB_g):.4f}", f"{yB_m:.4f}", f"{thB_l:.6f}", f"{thB_r:.6f}", f"{i.allowDeflectionMm:.4f}", f"{i.allowSlopeRad:.6f}"])
        rows.append(["C", f"{geo['x5']:.1f}", f"{abs(yC_g):.4f}", f"{yC_m:.4f}", f"{thC_l:.6f}", f"{thC_r:.6f}", f"{i.allowDeflectionMm:.4f}", f"{i.allowSlopeRad:.6f}"])

        prev_defl = self.previous_table_state.get("defl", [])
        self.deflectionTable.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                is_changed = False
                if prev_defl and r < len(prev_defl) and c < len(prev_defl[r]):
                    if prev_defl[r][c] != val:
                        is_changed = True
                self.deflectionTable.setItem(r, c, _tableItem(val, highlight=is_changed))
        
        self.previous_table_state["defl"] = rows

        def plotDeflSlope(canvasD: PlotCanvas, canvasS: PlotCanvas, key: str):
            x = shaftData[key]["x_t"]
            y = shaftData[key]["y"]
            th = shaftData[key]["th"]

            canvasD.clear(canvasD.ax.get_title())
            canvasD.ax.plot(x, y, linewidth=2.0, color='#3fb950')
            PlotCanvas.niceYLim(canvasD.ax, y)
            
            yMaxVal = np.max(y)
            xMaxVal = x[int(np.argmax(y))]
            yMinVal = np.min(y)
            xMinVal = x[int(np.argmin(y))]
            
            canvasD.ax.scatter([xMaxVal], [yMaxVal], s=40, color='#f0883e', zorder=5)
            canvasD.ax.annotate(f"{yMaxVal:.4f}", xy=(xMaxVal, yMaxVal), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9, color='#c9d1d9', fontweight='bold')
            canvasD.ax.scatter([xMinVal], [yMinVal], s=40, color='#f0883e', zorder=5)
            canvasD.ax.annotate(f"{yMinVal:.4f}", xy=(xMinVal, yMinVal), xytext=(0, -15), textcoords="offset points", ha="center", va="top", fontsize=9, color='#c9d1d9', fontweight='bold')

            canvasD.figure.tight_layout()
            canvasD.draw()

            canvasS.clear(canvasS.ax.get_title())
            canvasS.ax.plot(x, th, linewidth=2.0, color='#3fb950')
            PlotCanvas.niceYLim(canvasS.ax, th)
            
            yMaxVal = np.max(th)
            xMaxVal = x[int(np.argmax(th))]
            yMinVal = np.min(th)
            xMinVal = x[int(np.argmin(th))]
            
            canvasS.ax.scatter([xMaxVal], [yMaxVal], s=40, color='#f0883e', zorder=5)
            canvasS.ax.annotate(f"{yMaxVal:.5f}", xy=(xMaxVal, yMaxVal), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9, color='#c9d1d9', fontweight='bold')
            
            canvasS.ax.scatter([xMinVal], [yMinVal], s=40, color='#f0883e', zorder=5)
            canvasS.ax.annotate(f"{yMinVal:.5f}", xy=(xMinVal, yMinVal), xytext=(0, -15), textcoords="offset points", ha="center", va="top", fontsize=9, color='#c9d1d9', fontweight='bold')

            canvasS.figure.tight_layout()
            canvasS.draw()

        plotDeflSlope(self.deflPlotA, self.slopePlotA, "A")
        plotDeflSlope(self.deflPlotB, self.slopePlotB, "B")
        plotDeflSlope(self.deflPlotC, self.slopePlotC, "C")

    def _keySuggestion(self, dMm: float) -> str:
        dMm = float(dMm)
        if dMm <= 22:
            return "6x6"
        if dMm <= 30:
            return "8x7"
        if dMm <= 38:
            return "10x8"
        if dMm <= 44:
            return "12x8"
        if dMm <= 50:
            return "14x9"
        if dMm <= 58:
            return "16x10"
        if dMm <= 65:
            return "18x11"
        return "20x12"

    def _updateFinalDesign(self, results: dict, shaftData: dict):
        i = self.inputs
        geo = results["geometry"]

        TA = results["T_a"] * 1000.0
        TB = results["T_b"] * 1000.0
        TC = results["T_c"] * 1000.0

        def interpMres(shaftKey: str, xPos: float) -> float:
            return resultantMomentAt(shaftData[shaftKey], xPos)

        shaftCrit = {
            "A": [(geo["x2"], interpMres("A", geo["x2"]), TA)],
            "B": [(geo["x3"], interpMres("B", geo["x3"]), TB), (geo["x4"], interpMres("B", geo["x4"]), TB)],
            "C": [(geo["x5"], interpMres("C", geo["x5"]), TC)],
        }

        deflSummary = {
            "A": float(np.max(np.abs(shaftData["A"]["y"]))),
            "B": float(np.max(np.abs(shaftData["B"]["y"]))),
            "C": float(np.max(np.abs(shaftData["C"]["y"]))),
        }
        slopeSummary = {
            "A": max(float(abs(shaftData["A"]["th"][0])), float(abs(shaftData["A"]["th"][-1]))),
            "B": max(float(abs(shaftData["B"]["th"][0])), float(abs(shaftData["B"]["th"][-1]))),
            "C": max(float(abs(shaftData["C"]["th"][0])), float(abs(shaftData["C"]["th"][-1]))),
        }

        kfCalc = 1.0 + i.notchSensQ * (i.ktBending - 1.0)
        kfsCalc = 1.0 + i.notchSensQ * (i.ktsTorsion - 1.0)

        finishMap = {
            "A": i.surfaceFinishA,
            "B": i.surfaceFinishB,
            "C": i.surfaceFinishC
        }

        rows = []
        for shaftKey, dNow in [("A", i.shaftADiameterMm), ("B", i.shaftBDiameterMm), ("C", i.shaftCDiameterMm)]:
            nFatMin = 1e9
            nYMin = 1e9

            mf = correctedEnduranceLimitMpa(
                sutMpa=i.sutMpa,
                dMm=dNow,
                surfaceFinish=finishMap[shaftKey],
                reliability=i.reliability,
                kLoad=i.loadFactor,
                kMisc=i.miscFactor,
            )
            Se = mf["Se"]

            for _, mRes, tNmm in shaftCrit[shaftKey]:
                res = equivalentStressDEGerber(
                    MresNmm=mRes,
                    torqueNmm=tNmm,
                    dMm=dNow,
                    sutMpa=i.sutMpa,
                    syMpa=i.syMpa,
                    SeMpa=Se,
                    kfBending=kfCalc,
                    kfsTorsion=kfsCalc,
                )
                nFatMin = min(nFatMin, res["nFatigue"])
                nYMin = min(nYMin, res["nYield"])

            dReqStress = dNow
            if nFatMin < i.targetFatigueN:
                dReqStress = max(dReqStress, dNow * ((i.targetFatigueN / max(1e-9, nFatMin)) ** (1.0 / 3.0)))
            if nYMin < i.targetYieldN:
                dReqStress = max(dReqStress, dNow * ((i.targetYieldN / max(1e-9, nYMin)) ** (1.0 / 3.0)))

            yMax = deflSummary[shaftKey]
            thMax = slopeSummary[shaftKey]

            dReqDefl = dNow
            if i.allowDeflectionMm > 0 and yMax > i.allowDeflectionMm:
                dReqDefl = dNow * ((yMax / i.allowDeflectionMm) ** 0.25)

            dReqSlope = dNow
            if i.allowSlopeRad > 0 and thMax > i.allowSlopeRad:
                dReqSlope = dNow * ((thMax / i.allowSlopeRad) ** 0.25)

            dRec = max(dNow, dReqStress, dReqDefl, dReqSlope)
            filletR = 0.03 * dRec
            shoulderRatio = 1.20
            keySug = self._keySuggestion(dRec)

            status = "PASS" if (dRec <= dNow + 1e-6) else "INCREASE"

            rows.append([
                shaftKey,
                f"{dNow:.2f}",
                f"{dReqStress:.2f}",
                f"{dReqDefl:.2f}",
                f"{dReqSlope:.2f}",
                f"{dRec:.2f}",
                f"{filletR:.2f}",
                f"{shoulderRatio:.2f}",
                keySug,
                status,
            ])

        prev_final = self.previous_table_state.get("final", [])
        self.finalDesignTable.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                is_changed = False
                if prev_final and r < len(prev_final) and c < len(prev_final[r]):
                    if prev_final[r][c] != val:
                        is_changed = True
                
                is_bold = (c == 9)
                self.finalDesignTable.setItem(r, c, _tableItem(val, highlight=is_changed, bold=is_bold))
        
        self.previous_table_state["final"] = rows

    # ----------------------------
    # Results window
    # ----------------------------

    def _openResultsWindow(self):
        if self.lastResults is None or self.lastShaftData is None:
            QMessageBox.warning(self, "No Results", "Please click 'Recompute' first to generate results.")
            return

        r = self.lastResults
        i = self.inputs

        tables = []

        tables.append({
            "tabName": "Mesh Forces",
            "title": "Helical gear forces (Ft, Fr, Fa) + overturning moment Fa*(d/2)",
            "headers": ["Mesh", "Driver gear", "d_driver\n(mm)", "Torque\n(Nm)", "Ft (N)", "Fr (N)", "Fa (N)", "Mover\n(Nmm)"],
            "rows": [
                ["2-3", "2", f"{r['d2']:.2f}", f"{r['T_a']:.2f}", f"{r['mesh23']['Ft']:.2f}", f"{r['mesh23']['Fr']:.2f}", f"{r['mesh23']['Fa']:.2f}", f"{r['mesh23']['Mover2']:.2f}"],
                ["4-5", "4", f"{r['d4']:.2f}", f"{r['T_b']:.2f}", f"{r['mesh45']['Ft']:.2f}", f"{r['mesh45']['Fr']:.2f}", f"{r['mesh45']['Fa']:.2f}", f"{r['mesh45']['Mover4']:.2f}"],
            ],
        })

        react = r["reactions"]
        tables.append({
            "tabName": "Reactions",
            "title": "Support reactions (up positive)",
            "headers": ["Shaft", "Plane", "R_left (N)", "R_right (N)"],
            "rows": [
                ["A", "Tangential", f"{react['A_t'][0]:.2f}", f"{react['A_t'][1]:.2f}"],
                ["A", "Radial", f"{react['A_r'][0]:.2f}", f"{react['A_r'][1]:.2f}"],
                ["B", "Tangential", f"{react['B_t'][0]:.2f}", f"{react['B_t'][1]:.2f}"],
                ["B", "Radial", f"{react['B_r'][0]:.2f}", f"{react['B_r'][1]:.2f}"],
                ["C", "Tangential", f"{react['C_t'][0]:.2f}", f"{react['C_t'][1]:.2f}"],
                ["C", "Radial", f"{react['C_r'][0]:.2f}", f"{react['C_r'][1]:.2f}"],
            ],
        })

        sd = self.lastShaftData
        tables.append({
            "tabName": "Max Bending",
            "title": "Maximum internal moments (absolute) per plane", 
            "headers": ["Shaft", "Max |M_t|\n(Nmm)", "Max |M_r|\n(Nmm)", "Max |M_res|\n(Nmm)"],
            "rows": [
                ["A", f"{float(np.max(np.abs(sd['A']['M_t']))):.1f}", f"{float(np.max(np.abs(sd['A']['M_r']))):.1f}", f"{float(np.max(sd['A']['M_res'])):.1f}"],
                ["B", f"{float(np.max(np.abs(sd['B']['M_t']))):.1f}", f"{float(np.max(np.abs(sd['B']['M_r']))):.1f}", f"{float(np.max(sd['B']['M_res'])):.1f}"],
                ["C", f"{float(np.max(np.abs(sd['C']['M_t']))):.1f}", f"{float(np.max(np.abs(sd['C']['M_r']))):.1f}", f"{float(np.max(sd['C']['M_res'])):.1f}"],
            ],
        })

        g = r["geometry"]
        tables.append({
            "tabName": "Geometry",
            "title": "Project geometry used", 
            "headers": ["Shaft", "Length\n(mm)", "Gear positions\n(mm)", "Gear diameters\n(mm)", "Ratios"],
            "rows": [
                ["A", f"{g['LA']:.1f}", f"x2={g['x2']:.1f}", f"d2={r['d2']:.1f}", f"i1={i.stage1Ratio:.2f}"],
                ["B", f"{g['LB']:.1f}", f"x3={g['x3']:.1f}, x4={g['x4']:.1f}", f"d3={r['d3']:.1f}, d4={r['d4']:.1f}", f"i2={i.stage2Ratio:.2f}"],
                ["C", f"{g['LC']:.1f}", f"x5={g['x5']:.1f}", f"d5={r['d5']:.1f}", f"Total={i.stage1Ratio*i.stage2Ratio:.2f}"],
            ],
        })

        if self.resultsDialog is None:
            self.resultsDialog = ResultsDialog(self)

        try:
            self.resultsDialog.setTables(tables)
            self.resultsDialog.show()
            self.resultsDialog.raise_()
        except Exception as e:
            QMessageBox.critical(self, "Results Error", f"Failed to open results window:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    win = GearboxWorkbench()
    win.showMaximized()

    def handleEsc(event):
        if event.key() == Qt.Key.Key_Escape:
            if win.isFullScreen():
                win.showMaximized()
            else:
                win.showFullScreen()
        else:
            QMainWindow.keyPressEvent(win, event)

    win.keyPressEvent = handleEsc

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
