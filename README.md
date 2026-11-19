# Bubble & Pendant Drop Interface Calculator

[![Faraday Discussions](https://img.shields.io/badge/Faraday%20Discussions-10.1039%2FD6FD00113K-blue)](https://doi.org/10.1039/D6FD00113K)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24342-b31b1b.svg)](https://arxiv.org/abs/2609.24342) 
[![Zenodo](https://img.shields.io/badge/Zenodo-20720640-1682C4)](https://doi.org/10.5281/zenodo.20720640)

Calculate **interface shapes, volumes, and detachment thresholds** for fluid bubbles and pendant drops on flat surfaces.

This toolkit solves the **Young–Laplace equation** using the **Adams–Bashforth integration method** across a range of wetting states, including pinned and spreading/moving contact lines. The results are validated against classical experimental datasets from the literature.

The method and results are described in:

> I. Cannon, S. C. Endres, L. Mädler, and M. Avila,
> *Bubble detachment from circular cavities and flat surfaces*,
> **Faraday Discussions** (2026).
> https://doi.org/10.1039/D6FD00113K

![Bubble growing](plots/hydrophobic.svg)

*Evolution of a growing bubble interface (right) and its trajectory in the contact angle - contact radius phase space (left), transitioning from a pinned contact line at the cavity edge (red) to a spreading contact line on the flat surface (blue).*

## Requirements

* GCC
* Python 3
* NumPy
* Matplotlib
* LaTeX

## How to Run

Compile the C program:

```bash
gcc -O2 -o run bubble.c -lm
```

Run the calculator:

```bash
./run
```

Install the required Python packages:

```bash
pip install numpy matplotlib
```

Generate the static plots:

```bash
python plot.py
```

Generate an animation of the bubble growing (`plots/growing.svg`):

```bash
python animate.py
```

Optional flags for `animate.py`:

* `--radius`: Target dimensionless cavity radius $r_c/\lambda$ (default: `1.0`).
* `--phi2`: Target dimensionless contact angle parameter $\phi^2/\pi^2$ (default: `0.49`).
* `--fill-rate`: Dimensionless volumetric fill rate $\mathrm{d}(V/\lambda^3)/\mathrm{d}t$ (default: `1.0`).
* `--single-shot`: Play the animation once instead of looping indefinitely.
* `--opaque`: Use an opaque background instead of transparent.
* `--no-morph`: Use discrete frame switching instead of smooth path morphing.

