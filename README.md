# Bubble & Pendant Drop Interface Calculator

An engineering tool to calculate the analytical interface shapes, volumes, and detachment thresholds of fluid bubbles and pendant drops on flat surfaces.

This toolkit implements the **Adams-Bashforth integration method** to solve the Young-Laplace equation across varying wetting states (pinned vs. spreading/moving contact lines), validated against classical literature experimental datasets.

---

## Mathematical Background

The profile shapes are governed by the non-dimensionalized Young-Laplace equation under hydrostatic pressure. The interface arc-length parameter ($s$) is solved numerically using an explicit Adams-Bashforth integration routine according to:

$$\frac{d\phi}{ds} = \frac{2}{R_t} - \frac{z}{\lambda^2} - \frac{\sin\phi}{r}$$

Where:

* $R_t$ is the radius of curvature at the apex/top.
* $\lambda$ is the capillary length ($\lambda = \sqrt{\gamma / \Delta \rho g}$).
* $\phi$ is the tangent angle made with the horizontal plane.

---

## Project Structure

```text
├── bubble.py          # Core integration algorithms & profile loop re-orderers
├── run.py             # Main entry point to execute sweeps and profile tasks
├── plot.py            # Comprehensive Matplotlib script for manuscript-quality figures
├── simData/           # Output directory for generated integration text profiles
├── plots/             # Output folder for generated plots and diagrams (PDF format)
└── exptData/           # Experimental verification benchmarks (Demirkir24, Allred21, etc.)
```

---

## Installation & Requirements

Ensure you have a Python 3 environment with standard scientific computing libraries installed.

```bash
pip install numpy scipy matplotlib
```

> **Note:** The plotting script uses LaTeX rendering (`text.usetex: True`). Ensure you have a functioning LaTeX distribution installed on your system path (e.g., TeX Live, MiKTeX) to avoid pipeline plotting errors.

---

## How to Run

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

## Outputs & Experimental Validation

Calculated analytical profiles are verified natively against established experimental benchmarks located inside the `/exptData` directory:

| File | Reference |
|---|---|
| `demirkir24life.txt` | Demirkır, Ç., Wood, J. A., Lohse, D., and Krug, D. (2024). "Life beyond Fritz: On the Detachment of Electrolytic Bubbles." *Langmuir*, 40(39), 20474–20484. https://doi.org/10.1021/acs.langmuir.4c01963 |
| `allred21role.txt` | Allred, T. P., Weibel, J. A., and Garimella, S. V. (2021). "The Role of Dynamic Wetting Behavior during Bubble Growth and Departure from a Solid Surface." *Int. J. Heat Mass Transf.*, 172, 121167. https://doi.org/10.1016/j.ijheatmasstransfer.2021.121167 |
| `huang25effects.txt` | Huang, J. and Li, R. (2026). "Effects of Surface Wettability on Bubble Dynamics and Induced Liquid Flow: Finite-difference Analysis of Two-Phase Particle Image Velocimetry." *Phys. Rev. Fluids*, 11(2), 023603. https://doi.org/10.1103/jvxz-8mzv |
| `gunde01measurement.txt` | Gunde, R., Kumar, A., Lehnert-Batar, S., Mäder, R., and Windhab, E. J. (2001). "Measurement of the Surface and Interfacial Tension from Maximum Volume of a Pendant Drop." *J. Colloid Interface Sci.*, 244(1), 113–122. https://doi.org/10.1006/jcis.2001.7916 |
| `sasetty23stability.txt` | Sasetty, S. and Ward, T. (2023). "Stability and Critical Volume of a Suspended Pendant Drop in Air via Experiments and Eigenvalue Analysis." *Colloids Surf. A*, 666, 131346. https://doi.org/10.1016/j.colsurfa.2023.131346 |
| `LesageVolVsContRadSq.txt` | Lesage, F. J. and Marois, F. (2013). "Experimental and Numerical Analysis of Quasi-Static Bubble Size and Shape Characteristics at Detachment." *Int. J. Heat Mass Transf.*, 64, 53–69. https://doi.org/10.1016/j.ijheatmasstransfer.2013.04.019 |
| `MoriVolByContCubeVsContSqByCapSq.txt` | Mori, B. K. and Baines, W. D. (2001). "Bubble Departure from Cavities." *Int. J. Heat Mass Transf.*, 44(4), 771–783. https://doi.org/10.1016/s0017-9310(00)00133-2 |

The output pipeline generates automated technical reports saved inside `/plots`, containing capillary curves, maximum volume vs. detachment limits, and shape evolution configurations.
