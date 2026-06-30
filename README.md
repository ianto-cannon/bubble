# Bubble & Pendant Drop Interface Calculator

An engineering tool to calculate the analytical interface shapes, volumes, and detachment thresholds of fluid bubbles and pendant drops on flat surfaces.

This toolkit implements the **Adams-Bashforth integration method** to solve the Young-Laplace equation across varying wetting states (pinned vs. spreading/moving contact lines), validated against classical literature experimental datasets.

---

## 🔬 Mathematical Background

The profile shapes are governed by the non-dimensionalized Young-Laplace equation under hydrostatic pressure. The interface arc-length parameter ($s$) is solved numerically using an explicit Adams-Bashforth integration routine according to:

$$\frac{d\phi}{ds} = \frac{2}{R_t} - \frac{z}{\lambda^2} - \frac{\sin\phi}{r}$$

Where:

* $R_t$ is the radius of curvature at the apex/top.
* $\lambda$ is the capillary length ($\lambda = \sqrt{\gamma / \Delta \rho g}$).
* $\phi$ is the tangent angle made with the horizontal plane.

---

## 📂 Project Structure

```text
├── bubble.py          # Core integration algorithms & profile loop re-orderers
├── run.py             # Main entry point to execute sweeps and profile tasks
├── plot.py            # Comprehensive Matplotlib script for manuscript-quality figures
├── simData/           # Output directory for generated integration text profiles
├── plots/             # Output folder for generated plots and diagrams (PDF format)
└── exptData/           # Experimental verification benchmarks (Demirkir24, Allred21, etc.)
```

---

## 🛠️ Installation & Requirements

Ensure you have a Python 3 environment with standard scientific computing libraries installed.

```bash
pip install numpy scipy matplotlib
```

> **Note:** The plotting script uses LaTeX rendering (`text.usetex: True`). Ensure you have a functioning LaTeX distribution installed on your system path (e.g., TeX Live, MiKTeX) to avoid pipeline plotting errors.

---

## 🚀 How to Run

### 1. Execute Simulations and Analysis

Run the primary execution pipeline to trigger profiling routines and process structural spatial data maps:

```bash
python run.py
```

### 2. Custom Scripting & Core Module Usage

Import the simulation module inside your scripts to manually compute properties for unique fluid interfaces:

```python
from bubble import AdamsBashforthProfile

# Parameters: Capillary Length, Apex Radius, file path destination
volume, radius, height, centroid, final_psi = AdamsBashforthProfile(
    capLen=1.0,
    RadTop=0.5,
    fname="simData/my_bubble_profile.txt"
)

print(f"Calculated Drop Volume: {volume:.4f} λ³")
```

---

## 📊 Outputs & Experimental Validation

Calculated analytical profiles are verified natively against established experimental benchmarks located inside the `/exptData` directory:

* Demirkir et al. (2024) — *Langmuir*
* Allred et al. (2021)
* Huang et al. (2025)

The output pipeline generates automated technical reports saved inside `/plots`, containing capillary curves, maximum volume vs. detachment limits, and shape evolution configurations.
