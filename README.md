# Bubble & Pendant Drop Interface Calculator

[![Faraday Discussions](https://img.shields.io/badge/Faraday%20Discussions-10.1039%2FD6FD00113K-blue)](https://doi.org/10.1039/D6FD00113K)
[![arXiv](https://img.shields.io/badge/arXiv-2605.07504-b31b1b.svg)](https://arxiv.org/abs/2605.07504)
[![Zenodo](https://img.shields.io/badge/Zenodo-20720640-1682C4)](https://doi.org/10.5281/zenodo.20720640)
[![License](https://img.shields.io/github/license/ianto-cannon/bubble)](LICENSE)
![C](https://img.shields.io/badge/C-00599C?logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

Calculate **interface shapes, volumes, and detachment thresholds** for fluid bubbles and pendant drops on flat surfaces.

This toolkit solves the **Young–Laplace equation** using the **Adams–Bashforth integration method** across a range of wetting states, including pinned and spreading/moving contact lines. The results are validated against classical experimental datasets from the literature.

The method and results are described in:

> I. Cannon, S. C. Endres, L. Mädler, and M. Avila,
> *Bubble detachment from circular cavities and flat surfaces*,
> **Faraday Discussions** (2026).
> https://doi.org/10.1039/D6FD00113K

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

Generate the plots:

```bash
python plot.py
```
