"""
================================================================================
Fuel Cell Polarization Curve Modelling & Kinetic Parameter Extraction
================================================================================

Physical Chemistry Background
------------------------------
This script models the electrochemical behaviour of a hydrogen PEM fuel cell
using three foundational equations:

1. Butler-Volmer / Tafel Kinetics
   The full Butler-Volmer equation describes current density (j) as a function
   of activation overpotential (η):

       j = j₀ · [exp(α·F·η / R·T) - exp(-(1-α)·F·η / R·T)]

   At high overpotentials (|η| >> R·T/F ≈ 0.026 V at 298 K), the reverse
   reaction term becomes negligible and the equation collapses to the Tafel
   approximation:

       j ≈ j₀ · exp(α·F·η / R·T)

   Taking the natural log:

       ln(j) = ln(j₀) + (α·F / R·T) · η      [Tafel equation, linear form]

   Slope  → b = α·F / R·T   (Tafel slope, units: V⁻¹)
   Y-int  → ln(j₀)          (exchange current density intercept)

   WHY exclude low-overpotential data (<0.06 V)?
   At small η, the back-reaction term is NOT negligible — the full Butler-Volmer
   curvature dominates. Fitting a straight line through this region would
   underestimate the true Tafel slope and produce a biased j₀. The 0.06 V
   threshold (≈ 2.3 × R·T/F) is a widely accepted experimental convention
   ensuring we are firmly in the linear Tafel regime.

2. Ohmic Loss
   Resistive voltage drop across the membrane and contact resistances:

       η_ohmic = j · R_internal

   where R_internal is the area-specific resistance (Ω·cm²).

3. Full Cell Voltage Model
   Usable cell voltage is the theoretical OCV minus cumulative losses:

       V_cell = E_ocv - η_act(j) - η_ohmic(j)

   where η_act is solved numerically from the Tafel expression:

       η_act = (R·T / α·F) · ln(j / j₀)

================================================================================
Author  : Principal Research Scientist, Electrochemical Engineering
Version : 1.0.0
Python  : ≥ 3.10
Dependencies: numpy, pandas, scikit-learn, matplotlib
================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
FARADAY: float = 96_485.0        # C mol⁻¹
R_GAS: float = 8.314             # J mol⁻¹ K⁻¹
TEMPERATURE: float = 353.15      # K  (80 °C, typical PEM operating temp)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FuelCellParameters:
    """
    Physical and electrochemical parameters defining the fuel cell model.

    Attributes
    ----------
    j0_true : float
        True exchange current density (A cm⁻²) used for synthetic data
        generation. Typical PEM value: 1 × 10⁻⁴ A cm⁻².
    alpha : float
        Charge-transfer coefficient (dimensionless, 0 < α < 1).
        Symmetry factor for the cathodic ORR reaction; typically 0.5.
    e_ocv : float
        Ideal open-circuit voltage (V). Slightly above the thermodynamic
        1.23 V due to cross-over corrections or temperature effects (1.15 V).
    r_internal : float
        Area-specific internal resistance (Ω cm²). Encompasses membrane
        ionic resistance and contact resistances.
    noise_level : float
        Relative Gaussian noise injected into j (fraction, e.g. 0.02 = 2 %).
    eta_min : float
        Lower bound of overpotential scan (V).
    eta_max : float
        Upper bound of overpotential scan (V).
    n_points : int
        Number of data points in the synthetic dataset.
    tafel_threshold : float
        Minimum overpotential (V) for Tafel-region regression.
    rng_seed : int
        Seed for the NumPy random Generator (reproducibility).
    """

    j0_true: float = 1.0e-4          # A cm⁻²
    alpha: float = 0.5
    e_ocv: float = 1.15              # V
    r_internal: float = 0.15         # Ω cm²
    noise_level: float = 0.02        # 2 % relative
    eta_min: float = 0.005           # V
    eta_max: float = 0.300           # V
    n_points: int = 100
    tafel_threshold: float = 0.06    # V — Tafel region lower bound
    rng_seed: int = 42


@dataclass
class ExtractionResult:
    """
    Container for Tafel regression outputs.

    Attributes
    ----------
    j0_extracted : float
        Exchange current density back-calculated from linear regression (A cm⁻²).
    tafel_slope_V : float
        Tafel slope in V dec⁻¹ (more intuitive than V⁻¹).
    intercept_ln : float
        Y-intercept of ln(j) vs. η regression = ln(j₀).
    slope_ln : float
        Slope of ln(j) vs. η regression = α·F/(R·T).
    r2 : float
        Coefficient of determination for the Tafel fit.
    n_tafel_points : int
        Number of data points used in the regression.
    """

    j0_extracted: float = 0.0
    tafel_slope_V: float = 0.0
    intercept_ln: float = 0.0
    slope_ln: float = 0.0
    r2: float = 0.0
    n_tafel_points: int = 0


@dataclass
class VoltageBreakdown:
    """
    Decomposition of cell voltage losses at a target operating point.

    Attributes
    ----------
    j_target : float
        Target current density (A cm⁻²).
    eta_act : float
        Activation overpotential at j_target (V).
    eta_ohmic : float
        Ohmic overpotential at j_target (V).
    v_cell : float
        Usable cell voltage at j_target (V).
    e_ocv : float
        Open-circuit voltage (V).
    efficiency : float
        Voltage efficiency relative to E_ocv (%).
    """

    j_target: float = 0.0
    eta_act: float = 0.0
    eta_ohmic: float = 0.0
    v_cell: float = 0.0
    e_ocv: float = 0.0
    efficiency: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Core model class
# ──────────────────────────────────────────────────────────────────────────────

class FuelCellDiagnostics:
    """
    End-to-end fuel cell polarisation curve modelling and kinetic extraction.

    Workflow
    --------
    1. ``generate_dataset()``    — synthetic Butler-Volmer data + noise
    2. ``extract_tafel_params()`` — Tafel linear regression → j₀
    3. ``predict_voltage()``     — full cell voltage model at target j
    4. ``plot_results()``        — publication-quality two-panel figure

    Parameters
    ----------
    params : FuelCellParameters
        All physical and experimental configuration parameters.
    """

    def __init__(self, params: Optional[FuelCellParameters] = None) -> None:
        self.params: FuelCellParameters = params or FuelCellParameters()
        self._rng: np.random.Generator = np.random.default_rng(
            self.params.rng_seed
        )
        self.df: Optional[pd.DataFrame] = None
        self.extraction: Optional[ExtractionResult] = None
        self._model: Optional[LinearRegression] = None

    # ──────────────────────────────────────────────────────────────────────
    # 1. Data generation
    # ──────────────────────────────────────────────────────────────────────

    def _butler_volmer_current(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute ideal current density via the Butler-Volmer equation.

        Parameters
        ----------
        eta : np.ndarray
            Activation overpotential array (V).

        Returns
        -------
        np.ndarray
            Ideal current density (A cm⁻²).
        """
        p = self.params
        thermal_voltage: float = R_GAS * TEMPERATURE / FARADAY  # V
        anodic_term = np.exp(p.alpha * eta / thermal_voltage)
        cathodic_term = np.exp(-(1.0 - p.alpha) * eta / thermal_voltage)
        return p.j0_true * (anodic_term - cathodic_term)

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate a synthetic experimental dataset with instrument noise.

        Uses the modern NumPy Generator API (``np.random.default_rng``) to
        inject 2 % relative Gaussian noise. Negative current densities are
        physically non-physical for anodic overpotentials and are clipped to
        a small positive floor (10⁻⁸ A cm⁻²) to avoid ``ln(0)`` errors.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ``['eta_V', 'j_ideal_A_cm2', 'j_noisy_A_cm2', 'ln_j']``
        """
        p = self.params
        eta: np.ndarray = np.linspace(p.eta_min, p.eta_max, p.n_points)
        j_ideal: np.ndarray = self._butler_volmer_current(eta)

        # 2 % relative Gaussian noise via modern Generator API
        noise_multiplier: np.ndarray = self._rng.normal(
            loc=1.0, scale=p.noise_level, size=p.n_points
        )
        j_noisy: np.ndarray = j_ideal * noise_multiplier

        # Clip unphysical negatives — instrument noise can occasionally
        # produce j < 0 at very low overpotentials
        j_noisy = np.clip(j_noisy, a_min=1.0e-8, a_max=None)

        self.df = pd.DataFrame({
            "eta_V": eta,
            "j_ideal_A_cm2": j_ideal,
            "j_noisy_A_cm2": j_noisy,
            "ln_j": np.log(j_noisy),
        })

        print("=" * 70)
        print("  STEP 1 · DATA GENERATION & NOISE CHARACTERISATION")
        print("=" * 70)
        print(f"  Overpotential range : {p.eta_min:.3f} – {p.eta_max:.3f} V")
        print(f"  Data points         : {p.n_points}")
        print(f"  True j₀             : {p.j0_true:.2e} A cm⁻²")
        print(f"  Charge-transfer α   : {p.alpha:.2f}")
        print(f"  Noise level         : {p.noise_level * 100:.1f} % relative Gaussian")
        print(f"  RNG seed            : {p.rng_seed}")
        print(f"\n  Dataset preview (first 5 rows):\n")
        print(self.df.head().to_string(index=False, float_format="{:.5f}".format))
        print()

        return self.df

    # ──────────────────────────────────────────────────────────────────────
    # 2. Tafel kinetic parameter extraction
    # ──────────────────────────────────────────────────────────────────────

    def extract_tafel_params(self) -> ExtractionResult:
        """
        Isolate the Tafel region and extract kinetic parameters via OLS.

        The Tafel region is defined as η ≥ ``tafel_threshold`` (default 0.06 V).
        A scikit-learn ``LinearRegression`` is fitted to ``ln(j)`` vs. ``η``.

        Kinetic back-calculation
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ::

            ln(j) = ln(j₀) + (α·F / R·T) · η
            → intercept  = ln(j₀)   ⟹  j₀ = exp(intercept)
            → slope      = α·F/(R·T) [V⁻¹]

        The conventional Tafel slope (V dec⁻¹) is:

        ::

            b = 2.303 · R·T / (α·F)

        Returns
        -------
        ExtractionResult
            Extracted kinetic parameters and regression statistics.

        Raises
        ------
        RuntimeError
            If ``generate_dataset()`` has not been called first.
        """
        if self.df is None:
            raise RuntimeError("Call generate_dataset() before extract_tafel_params().")

        p = self.params
        mask_tafel: pd.Series = self.df["eta_V"] >= p.tafel_threshold
        df_tafel: pd.DataFrame = self.df[mask_tafel].copy()
        df_excluded: pd.DataFrame = self.df[~mask_tafel].copy()

        X_tafel: np.ndarray = df_tafel["eta_V"].values.reshape(-1, 1)
        y_tafel: np.ndarray = df_tafel["ln_j"].values

        self._model = LinearRegression()
        self._model.fit(X_tafel, y_tafel)

        slope_ln: float = float(self._model.coef_[0])
        intercept_ln: float = float(self._model.intercept_)
        j0_extracted: float = float(np.exp(intercept_ln))

        y_pred: np.ndarray = self._model.predict(X_tafel)
        r2: float = float(r2_score(y_tafel, y_pred))

        # Tafel slope in conventional units (V per decade)
        tafel_slope_V: float = 2.303 / slope_ln  # V dec⁻¹

        self.extraction = ExtractionResult(
            j0_extracted=j0_extracted,
            tafel_slope_V=tafel_slope_V,
            intercept_ln=intercept_ln,
            slope_ln=slope_ln,
            r2=r2,
            n_tafel_points=len(df_tafel),
        )

        print("=" * 70)
        print("  STEP 2 · TAFEL KINETIC PARAMETER EXTRACTION")
        print("=" * 70)
        print(
            f"\n  Physical rationale for excluding η < {p.tafel_threshold:.2f} V:\n"
            f"  ─────────────────────────────────────────────────────────────\n"
            f"  Below ~2×(RT/F) ≈ 0.052 V the back-reaction (cathodic) term\n"
            f"  in the Butler-Volmer equation remains significant. Fitting a\n"
            f"  straight line through this curved region inflates the Tafel\n"
            f"  slope and biases j₀ toward unphysically high values.\n"
            f"  The {p.tafel_threshold:.2f} V threshold ensures we operate\n"
            f"  firmly within the exponential (Tafel) asymptote.\n"
        )
        print(f"  Tafel region points : {self.extraction.n_tafel_points}"
              f" (η ≥ {p.tafel_threshold:.2f} V)")
        print(f"  Excluded points     : {len(df_excluded)}"
              f" (η < {p.tafel_threshold:.2f} V)")
        print(f"\n  ── Regression Results ──────────────────────────────────")
        print(f"  Slope  [α·F/(R·T)]  : {slope_ln:.4f} V⁻¹")
        print(f"  Intercept  [ln(j₀)] : {intercept_ln:.4f}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        print(f"\n  ── Extracted Parameters ────────────────────────────────")
        print(f"  j₀ extracted        : {j0_extracted:.4e} A cm⁻²")
        print(f"  j₀ true (synthetic) : {p.j0_true:.4e} A cm⁻²")
        rel_error: float = abs(j0_extracted - p.j0_true) / p.j0_true * 100
        print(f"  Relative error      : {rel_error:.3f} %")
        print(f"  Tafel slope         : {tafel_slope_V * 1000:.2f} mV dec⁻¹")
        print()

        return self.extraction

    # ──────────────────────────────────────────────────────────────────────
    # 3. Full cell voltage model & usable voltage prediction
    # ──────────────────────────────────────────────────────────────────────

    def _activation_overpotential(
        self, j: np.ndarray, j0: float
    ) -> np.ndarray:
        """
        Invert the Tafel equation to compute activation overpotential.

        ::

            η_act = (R·T / α·F) · ln(j / j₀)

        Parameters
        ----------
        j : np.ndarray
            Current density array (A cm⁻²).
        j0 : float
            Exchange current density (A cm⁻²).

        Returns
        -------
        np.ndarray
            Activation overpotential (V).
        """
        thermal_voltage: float = R_GAS * TEMPERATURE / FARADAY
        return (thermal_voltage / self.params.alpha) * np.log(j / j0)

    def build_polarisation_curve(
        self,
        j_array: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Construct the full V_cell vs. j polarisation curve.

        ::

            V_cell = E_ocv − η_act(j) − η_ohmic(j)
            η_ohmic = j · R_internal

        Parameters
        ----------
        j_array : np.ndarray, optional
            Current density sweep (A cm⁻²). Defaults to 0.001–1.5 A cm⁻².

        Returns
        -------
        pd.DataFrame
            Columns: ``['j', 'eta_act', 'eta_ohmic', 'v_cell']``

        Raises
        ------
        RuntimeError
            If ``extract_tafel_params()`` has not been called first.
        """
        if self.extraction is None:
            raise RuntimeError(
                "Call extract_tafel_params() before build_polarisation_curve()."
            )

        if j_array is None:
            j_array = np.linspace(1.0e-3, 1.5, 500)

        j0 = self.extraction.j0_extracted
        p = self.params

        eta_act: np.ndarray = self._activation_overpotential(j_array, j0)
        eta_ohmic: np.ndarray = j_array * p.r_internal
        v_cell: np.ndarray = p.e_ocv - eta_act - eta_ohmic

        return pd.DataFrame({
            "j": j_array,
            "eta_act": eta_act,
            "eta_ohmic": eta_ohmic,
            "v_cell": v_cell,
        })

    def predict_voltage(self, j_target: float = 0.6) -> VoltageBreakdown:
        """
        Predict usable cell voltage and loss breakdown at a target j.

        Parameters
        ----------
        j_target : float
            Target operating current density (A cm⁻²). Default: 0.6 A cm⁻².

        Returns
        -------
        VoltageBreakdown
            Full decomposition of voltage losses at j_target.

        Raises
        ------
        RuntimeError
            If ``extract_tafel_params()`` has not been called first.
        """
        if self.extraction is None:
            raise RuntimeError(
                "Call extract_tafel_params() before predict_voltage()."
            )

        p = self.params
        j0 = self.extraction.j0_extracted
        j_arr = np.array([j_target])

        eta_act: float = float(self._activation_overpotential(j_arr, j0)[0])
        eta_ohmic: float = float(j_target * p.r_internal)
        v_cell: float = float(p.e_ocv - eta_act - eta_ohmic)
        efficiency: float = v_cell / p.e_ocv * 100.0

        result = VoltageBreakdown(
            j_target=j_target,
            eta_act=eta_act,
            eta_ohmic=eta_ohmic,
            v_cell=v_cell,
            e_ocv=p.e_ocv,
            efficiency=efficiency,
        )

        print("=" * 70)
        print("  STEP 3 · USABLE CELL VOLTAGE PREDICTION")
        print("=" * 70)
        print(f"\n  Target operating current density : {j_target:.3f} A cm⁻²\n")
        print(f"  ── Voltage Budget ───────────────────────────────────────")
        print(f"  E_ocv (open-circuit voltage)     : +{p.e_ocv:.4f} V")
        print(f"  η_act (activation overpotential) : −{eta_act:.4f} V")
        print(f"  η_ohmic (ohmic overpotential)    : −{eta_ohmic:.4f} V")
        print(f"  {'─' * 50}")
        print(f"  V_cell (usable cell voltage)     :  {v_cell:.4f} V")
        print(f"\n  Voltage efficiency               :  {efficiency:.2f} %")
        print(f"  Activation loss fraction         :  "
              f"{eta_act / (eta_act + eta_ohmic) * 100:.1f} %")
        print(f"  Ohmic loss fraction              :  "
              f"{eta_ohmic / (eta_act + eta_ohmic) * 100:.1f} %")
        print()

        return result

    # ──────────────────────────────────────────────────────────────────────
    # 4. Publication-quality visualisation
    # ──────────────────────────────────────────────────────────────────────

    def plot_results(
        self,
        breakdown: VoltageBreakdown,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Generate a two-panel publication-quality figure.

        Panel A — Semi-log Tafel Plot
            ``ln(j)`` vs. ``η`` with experimental scatter, Tafel regression
            line, and shaded excluded region.

        Panel B — Full Polarisation Curve
            ``V_cell`` vs. ``j`` with the target operating point annotated.

        Parameters
        ----------
        breakdown : VoltageBreakdown
            Output from ``predict_voltage()``.
        save_path : str, optional
            If provided, saves the figure to this path (300 dpi PNG/PDF).

        Returns
        -------
        Figure
            The matplotlib Figure object for further customisation if needed.

        Raises
        ------
        RuntimeError
            If any prior pipeline step has not been completed.
        """
        if self.df is None or self.extraction is None or self._model is None:
            raise RuntimeError(
                "Run the full pipeline (generate_dataset → extract_tafel_params"
                " → predict_voltage) before calling plot_results()."
            )

        p = self.params
        ext = self.extraction
        pol_df: pd.DataFrame = self.build_polarisation_curve()

        # ── Style ──────────────────────────────────────────────────────────
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 120,
        })

        BLUE = "#1a6faf"
        ORANGE = "#e07b39"
        GREEN = "#2e8b57"
        RED = "#c0392b"
        GREY_LIGHT = "#f0f4f8"
        GREY_MID = "#adb5bd"

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 5.5),
            constrained_layout=True,
        )
        fig.patch.set_facecolor("white")

        # ══════════════════════════════════════════════════════════════════
        # PANEL A — Tafel Plot
        # ══════════════════════════════════════════════════════════════════
        ax_a: Axes = axes[0]
        ax_a.set_facecolor("white")

        mask_tafel = self.df["eta_V"] >= p.tafel_threshold
        df_tafel = self.df[mask_tafel]
        df_excl = self.df[~mask_tafel]

        # Shaded excluded zone
        ax_a.axvspan(
            p.eta_min, p.tafel_threshold,
            color=GREY_LIGHT, zorder=0,
            label=f"Excluded region (η < {p.tafel_threshold:.2f} V)"
        )
        ax_a.axvline(
            p.tafel_threshold, color=GREY_MID, linewidth=1.2,
            linestyle="--", zorder=1
        )

        # Experimental scatter — excluded points
        ax_a.scatter(
            df_excl["eta_V"], df_excl["ln_j"],
            color=GREY_MID, s=28, alpha=0.7, zorder=2,
            label="Experimental data (excluded)", marker="o",
        )

        # Experimental scatter — Tafel points
        ax_a.scatter(
            df_tafel["eta_V"], df_tafel["ln_j"],
            color=BLUE, s=32, alpha=0.85, zorder=3,
            label="Experimental data (Tafel region)", marker="o",
            edgecolors="white", linewidths=0.4,
        )

        # Regression fit line (extended slightly beyond data)
        eta_fit = np.linspace(p.tafel_threshold, p.eta_max, 300).reshape(-1, 1)
        ln_j_fit = self._model.predict(eta_fit)
        ax_a.plot(
            eta_fit, ln_j_fit,
            color=ORANGE, linewidth=2.2, zorder=4,
            label=(
                f"Linear Tafel fit  (R² = {ext.r2:.5f})\n"
                f"Slope = {ext.slope_ln:.2f} V⁻¹ | "
                f"j₀ = {ext.j0_extracted:.2e} A cm⁻²"
            ),
        )

        # True j₀ reference line
        ax_a.axhline(
            np.log(p.j0_true), color=GREEN, linewidth=1.2,
            linestyle=":", alpha=0.8, zorder=2,
            label=f"True ln(j₀) = {np.log(p.j0_true):.3f}"
        )

        ax_a.set_xlabel("Activation Overpotential  η  (V)")
        ax_a.set_ylabel("ln( j )   [j in A cm⁻²]")
        ax_a.set_title("(A)  Semi-log Tafel Plot")
        ax_a.legend(fontsize=8.5, loc="upper left")

        # Annotation box
        _box_text = (
            f"Tafel slope = {ext.tafel_slope_V * 1000:.1f} mV dec⁻¹\n"
            f"j₀ (extracted) = {ext.j0_extracted:.3e} A cm⁻²\n"
            f"j₀ (true)      = {p.j0_true:.3e} A cm⁻²"
        )
        ax_a.text(
            0.97, 0.05, _box_text,
            transform=ax_a.transAxes,
            fontsize=8.2, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9f0",
                      edgecolor=ORANGE, alpha=0.9),
        )

        ax_a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_a.tick_params(which="minor", length=3, color=GREY_MID)

        # ══════════════════════════════════════════════════════════════════
        # PANEL B — Full Polarisation Curve
        # ══════════════════════════════════════════════════════════════════
        ax_b: Axes = axes[1]
        ax_b.set_facecolor("white")

        # OCV reference
        ax_b.axhline(
            p.e_ocv, color=GREY_MID, linewidth=1.0,
            linestyle="--", alpha=0.7,
            label=f"E_ocv = {p.e_ocv:.2f} V"
        )

        # Polarisation curve
        ax_b.plot(
            pol_df["j"], pol_df["v_cell"],
            color=BLUE, linewidth=2.5, zorder=3,
            label="V_cell = E_ocv − η_act − η_ohmic",
        )

        # Shaded area under the curve (power region)
        ax_b.fill_between(
            pol_df["j"], pol_df["v_cell"], alpha=0.06,
            color=BLUE, zorder=1,
        )

        # Individual loss contributions (stacked reference)
        ax_b.plot(
            pol_df["j"], p.e_ocv - pol_df["eta_act"],
            color=ORANGE, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2,
            label="E_ocv − η_act  (activation only)"
        )

        # Target operating point
        j_t = breakdown.j_target
        v_t = breakdown.v_cell
        ax_b.scatter(
            [j_t], [v_t],
            color=RED, s=200, zorder=5, marker="*",
            label=(
                f"Operating point\n"
                f"j = {j_t:.2f} A cm⁻²,  V = {v_t:.3f} V\n"
                f"η_act = {breakdown.eta_act:.4f} V\n"
                f"η_ohm = {breakdown.eta_ohmic:.4f} V"
            ),
        )

        # Voltage loss arrows
        arrow_kw = dict(
            arrowstyle="<->", color=GREY_MID,
            lw=1.2, mutation_scale=10,
        )
        _eta_act_mid = p.e_ocv - breakdown.eta_act / 2
        _eta_ohm_mid = v_t + breakdown.eta_ohmic / 2

        ax_b.annotate(
            "", xy=(j_t + 0.07, p.e_ocv - breakdown.eta_act),
            xytext=(j_t + 0.07, p.e_ocv),
            arrowprops=arrow_kw,
        )
        ax_b.text(
            j_t + 0.09, _eta_act_mid,
            f"η_act\n{breakdown.eta_act:.3f} V",
            fontsize=7.5, color=ORANGE, va="center",
        )

        ax_b.annotate(
            "", xy=(j_t + 0.07, v_t),
            xytext=(j_t + 0.07, p.e_ocv - breakdown.eta_act),
            arrowprops=arrow_kw,
        )
        ax_b.text(
            j_t + 0.09, _eta_ohm_mid,
            f"η_ohm\n{breakdown.eta_ohmic:.3f} V",
            fontsize=7.5, color=BLUE, va="center",
        )

        ax_b.set_xlabel("Current Density  j  (A cm⁻²)")
        ax_b.set_ylabel("Cell Voltage  V_cell  (V)")
        ax_b.set_title("(B)  Full Fuel Cell Polarisation Curve")
        ax_b.set_xlim(left=0.0)
        ax_b.set_ylim(bottom=0.0)
        ax_b.legend(fontsize=8.5, loc="upper right")

        _box_b = (
            f"Efficiency: {breakdown.efficiency:.1f} %\n"
            f"R_int = {p.r_internal:.2f} Ω·cm²"
        )
        ax_b.text(
            0.03, 0.07, _box_b,
            transform=ax_b.transAxes,
            fontsize=8.2, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f8ff",
                      edgecolor=BLUE, alpha=0.9),
        )

        ax_b.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_b.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_b.tick_params(which="minor", length=3, color=GREY_MID)

        # ── Super-title ────────────────────────────────────────────────────
        fig.suptitle(
            "PEM Fuel Cell Diagnostics  ·  Tafel Analysis & Polarisation Curve",
            fontsize=14, fontweight="bold", y=1.05, color="#222222",
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight",
                        facecolor="white")
            print(f"  Figure saved → {save_path}")

        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────────────────────────

def run_diagnostics(
    target_current_density: float = 0.6,
    save_figure: Optional[str] = "/mnt/user-data/outputs/fuel_cell_polarisation.png",
    params: Optional[FuelCellParameters] = None,
) -> None:
    """
    Execute the full fuel cell diagnostic pipeline.

    Parameters
    ----------
    target_current_density : float
        Operating current density for voltage prediction (A cm⁻²).
    save_figure : str, optional
        File path to save the figure. Pass ``None`` to skip saving.
    params : FuelCellParameters, optional
        Override default physical parameters.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    print("\n" + "═" * 70)
    print("  PEM FUEL CELL DIAGNOSTICS — Full Modelling Pipeline")
    print("═" * 70 + "\n")

    model = FuelCellDiagnostics(params=params)

    # Step 1 — Synthetic dataset
    model.generate_dataset()

    # Step 2 — Tafel kinetics
    model.extract_tafel_params()

    # Step 3 — Voltage prediction
    breakdown = model.predict_voltage(j_target=target_current_density)

    # Step 4 — Plot
    print("=" * 70)
    print("  STEP 4 · GENERATING PUBLICATION-QUALITY FIGURE")
    print("=" * 70)
    model.plot_results(breakdown=breakdown, save_path=save_figure)

    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70 + "\n")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_diagnostics(
        target_current_density=0.6,
        save_figure= None,
    )
