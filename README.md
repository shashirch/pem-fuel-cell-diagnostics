PEM Fuel Cell Diagnostics & Kinetic Modelling
A production-grade Python tool for simulating proton exchange membrane fuel cell (PEMFC) electrochemical polarization data, performing Tafel linear regression, extracting fundamental kinetic parameters (such as exchange current density $j_0$), and predicting usable cell voltage under targeted operating conditions.
---
Key Features
Butler-Volmer & Tafel Kinetic Modelling: Simulates activation overpotential and fuel cell polarization characteristics.
Kinetic Parameter Extraction: Performs linear regression on $\ln(j)$ vs. $\eta$ to extract exchange current density ($j_0$) and Tafel slope.
Low-Overpotential Filtering: Excludes data below $0.06\text{ V}$ to prevent back-reaction kinetics from biasing Tafel estimates.
Usable Voltage Breakdown: Predicts cell voltage ($V_{\text{cell}}$) at targeted load conditions, decomposing activation ($\eta_{\text{act}}$) and ohmic ($\eta_{\text{ohmic}}$) loss fractions.
Statistical Uncertainty Quantification: Features statistical bootstrapping for estimating standard error and 95% confidence intervals on derived kinetics.
Publication-Quality Visuals: Generates dual-panel plots illustrating semi-log Tafel linearizations alongside usable polarization curves.
---
Technical Background
The framework relies on three core electrochemical principles:
Butler-Volmer / Tafel Kinetics:
At high activation overpotentials ($\eta > 0.06\text{ V}$), reverse cathodic reactions are negligible, simplifying Butler-Volmer to the Tafel equation:
$$\ln(j) = \ln(j_0) + \frac{\alpha n F}{R T} \cdot \eta$$
Slope: $b = \frac{\alpha n F}{R T}$
Intercept: $\ln(j_0)$
Ohmic Overpotential Loss:
$$\eta_{\text{ohmic}} = j \cdot R_{\text{internal}}$$
Usable Cell Voltage:
$$V_{\text{cell}} = E_{\text{ocv}} - \eta_{\text{act}}(j) - \eta_{\text{ohmic}}(j)$$
---
Project Structure
```text
pem-fuel-cell-diagnostics/
├── pem.py              # Main script containing simulation & diagnostic framework
├── README.md           # Project documentation
└── requirements.txt    # Required dependencies
