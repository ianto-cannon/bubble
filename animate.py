#!/usr/bin/env python3
"""animatePort.py — animated SVG of bubble evolution, built to play in Safari.

Left panel   phi_0^2/pi^2 vs r_0/lambda — paper-style parameter-space plot.
Right panel  bubble interface profile.

Default: phi/pi = 0.7 and cavity radius = 1 lambda.

Run from the repository directory:
    python animateU_typical.py

Optional:
    python animateU_typical.py [repo_dir] [work_dir] [out.svg]
        [--phi2 0.49] [--radius 1] [--fill-rate 1]
        [--single-shot] [--opaque] [--no-morph]
"""

import os
import sys
import glob
import subprocess
import tempfile
import shutil
import re
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import xml.etree.ElementTree as ET


warnings.filterwarnings("ignore", message=".*tight_layout.*")

plt.rcParams.update({"text.usetex": True, "font.size": 14})
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["path.simplify"] = False
plt.rcParams["svg.hashsalt"] = "animsvg"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


# --------------- CLI ---------------

ap = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

ap.add_argument("repo", nargs="?", default=".")
ap.add_argument("work", nargs="?", default=None)
ap.add_argument("out", nargs="?", default="port7.svg")

ap.add_argument(
    "--fill-rate",
    type=float,
    default=1.0,
    help="dimensionless fill rate d(V/lambda^3)/dt [lambda^3/s]"
)

ap.add_argument(
    "--phi2",
    type=float,
    default=0.7 ** 2,
    help="target phi^2/pi^2 (default: 0.49, corresponding to phi/pi=0.7)"
)

ap.add_argument(
    "--radius",
    type=float,
    default=1.0,
    help="target cavity radius r_c/lambda"
)

ap.add_argument("--single-shot", action="store_true")

ap.add_argument(
    "--opaque",
    action="store_true",
    help="use an opaque white background (transparent by default)"
)

ap.add_argument("--transparent", action="store_true", help=argparse.SUPPRESS)

ap.add_argument(
    "--morph",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="animate bubble outline by morphing (default; --no-morph uses flipbook)"
)

args = ap.parse_args()

REPO, OUT = args.repo, args.out

FILL_RATE = args.fill_rate
if FILL_RATE <= 0:
    ap.error("--fill-rate must be positive")

REPEAT = "1" if args.single_shot else "indefinite"
TRANSPARENT = not args.opaque

WORK = args.work or tempfile.mkdtemp(prefix="animwork_")
os.makedirs(WORK, exist_ok=True)

SIM = os.path.join(REPO, "simData") + "/"
RUN = os.path.join(REPO, "run")


# --------------- Geometry and dataset selection ---------------

RC_TARGET = args.radius
PHI2_TARGET = args.phi2

if not 0.0 < RC_TARGET:
    ap.error("--radius must be positive")

if not 0.0 <= PHI2_TARGET <= 1.0:
    ap.error("--phi2 must be between 0 and 1")

if not os.path.isdir(SIM):
    ap.error(f"simulation data directory not found: {SIM}")

CAV = 0.3
KP, KS, M = 22, 118, 48

R_GUIDE, R_BAND_END = 3.219, 3.832
BLUE_REF = "BinAngMaxVol.txt"
PREC = 2


def find_nearest_dataset(
    pattern, column, target, scale=1.0, label="parameter"
):
    """Find the matching dataset by its first-row parameter value."""

    paths = sorted(glob.glob(os.path.join(SIM, pattern)))

    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {SIM!r}"
        )

    candidates = []

    for path in paths:
        try:
            first = np.loadtxt(path, max_rows=1)
            value = float(first[column]) * scale

            if np.isfinite(value):
                candidates.append((abs(value - target), value, path))

        except (OSError, ValueError, IndexError):
            continue

    if not candidates:
        raise ValueError(
            f"Could not read {label} from files matching {pattern!r}"
        )

    error, value, path = min(candidates, key=lambda item: item[0])

    print(
        f"{label}: requested {target:g}, selected {value:g} "
        f"({os.path.basename(path)}; difference {error:.4g})"
    )

    return path, value


# Select radius and phi^2/pi^2 independently.
# If pinning cannot reach the selected transition, animate pinning only.

rad_candidates = []

for path in sorted(glob.glob(os.path.join(SIM, "loop_rad*.txt"))):
    try:
        data = np.loadtxt(path)

        if data.ndim != 2 or data.shape[1] < 7 or len(data) < 2:
            continue

        radius = float(data[0, 0])
        phi_pin = 1.0 - data[:, 2] / np.pi

        rad_candidates.append((path, data, radius, phi_pin))

    except (OSError, ValueError, IndexError):
        continue


ang_candidates = []

for path in sorted(glob.glob(os.path.join(SIM, "loop_ang*.txt"))):
    try:
        data = np.loadtxt(path)

        if data.ndim != 2 or data.shape[1] < 7 or len(data) < 2:
            continue

        i_max = int(np.argmax(data[:, 6]))
        data = data[:i_max + 1]

        inside_angle = float(data[0, 2]) * 180.0 / np.pi
        phi2 = (1.0 - float(data[0, 2]) / np.pi) ** 2
        phi_w = 1.0 - float(data[-1, 2]) / np.pi

        if (
            np.isfinite(phi2)
            and np.isfinite(phi_w)
            and 0 <= phi2 <= 1
            and 0 <= phi_w <= 1
        ):
            ang_candidates.append(
                (path, data, inside_angle, phi2, phi_w)
            )

    except (OSError, ValueError, IndexError):
        continue


if not rad_candidates:
    raise FileNotFoundError(
        f"No readable loop_rad*.txt datasets in {SIM}"
    )

if not ang_candidates:
    raise FileNotFoundError(
        f"No readable loop_ang*.txt datasets in {SIM}"
    )


rad_path, pr, RC, phi_pin = min(
    rad_candidates,
    key=lambda item: abs(item[2] - RC_TARGET)
)

(
    ang_path,
    candidate_sp,
    CONTACT_ANGLE_DEG,
    PHI2_SELECTED,
    PHI_W
) = min(
    ang_candidates,
    key=lambda item: abs(item[3] - PHI2_TARGET)
)


pin_rows = np.flatnonzero(phi_pin <= PHI_W + 1e-12)
spread_rows = np.flatnonzero(candidate_sp[:, 0] >= RC)

if len(pin_rows) and len(spread_rows):

    jp = int(pin_rows[0])
    js = int(spread_rows[0])

    sp = candidate_sp
    HAS_SPREADING = True
    SPRD = os.path.basename(ang_path)

    print(
        f"spreading trajectory selected: {SPRD}; "
        f"transition phi_w={PHI_W:g}"
    )

else:

    jp = len(pr) - 1
    sp = np.empty((0, pr.shape[1]), dtype=float)
    js = 0

    HAS_SPREADING = False
    SPRD = None

    if not len(pin_rows):
        print(
            "WARNING: pinned trajectory does not reach the chosen "
            "contact angle; animating pinning only."
        )
    else:
        print(
            "WARNING: chosen spreading trajectory does not reach "
            "the selected cavity radius; animating pinning only."
        )


PRIN = os.path.basename(rad_path)

print(
    f"cavity radius r_c/lambda: requested {RC_TARGET:g}, "
    f"selected {RC:g} ({PRIN}; difference {abs(RC - RC_TARGET):.4g})"
)

print(
    f"phi^2/pi^2 target: {PHI2_TARGET:.8g}; "
    f"selected {PHI2_SELECTED:.8g} "
    f"(inside angle {CONTACT_ANGLE_DEG:.4g} deg)"
)

print(f"configuration: Rc={RC:g} lambda")
print(f"pinned: {PRIN}; rows used: {jp + 1}")

if HAS_SPREADING:
    print(f"spreading: {SPRD}; transition phi_w={PHI_W:g}")
else:
    print("spreading: skipped")


r0_p_all = pr[:jp + 1, 0] / pr[:jp + 1, 4]
ph_p_all = (1 - pr[:jp + 1, 2] / np.pi) ** 2

r0_s_all = sp[js:, 0] / sp[js:, 4]
ph_s_all = (1 - sp[js:, 2] / np.pi) ** 2


# --------------- Frame schedule — volume rate ---------------

print(f"pinning:   {jp + 1} rows available, requesting {KP}")

assert KP <= jp, (
    f"KP={KP} > jp={jp}: not enough distinct pinning rows — "
    "reduce KP or re-run sim with denser output"
)

if HAS_SPREADING:
    print(f"spreading: {len(sp) - js} rows available, requesting {KS}")

    assert KS <= len(sp) - js, (
        f"KS={KS} > len(sp)-js={len(sp) - js}: "
        "not enough distinct spreading rows"
    )


def volume_schedule(vol, n):
    """Choose rows at approximately uniform delta V, with strictly increasing V."""

    v = np.maximum.accumulate(np.asarray(vol, dtype=float))

    unique_v, first_rows = np.unique(v, return_index=True)

    if len(unique_v) < n:
        raise ValueError(
            f"cannot schedule {n} frames: only {len(unique_v)} "
            "distinct volume values are available"
        )

    targets = np.linspace(unique_v[0], unique_v[-1], n)

    pos = np.searchsorted(unique_v, targets)
    pos = np.clip(pos, 0, len(unique_v) - 1)

    n_dup = int(np.sum(np.diff(pos) == 0))

    if n_dup:
        for i in range(1, n):
            if pos[i] <= pos[i - 1]:
                pos[i] = pos[i - 1] + 1

        if pos[-1] >= len(unique_v):
            pos[-1] = len(unique_v) - 1

            for i in range(n - 2, -1, -1):
                if pos[i] >= pos[i + 1]:
                    pos[i] = pos[i + 1] - 1

        if pos[0] < 0:
            raise ValueError(
                f"cannot fit {n} strictly increasing volume samples; "
                "reduce KP/KS or use denser simulation output"
            )

        print(
            f"  nudge: resolved {n_dup} duplicate target(s) "
            "using distinct volumes"
        )

    idx = first_rows[pos]

    if np.any(np.diff(v[idx]) <= 0):
        raise AssertionError(
            "internal error: selected volumes are not increasing"
        )

    return idx


# Pinning: skip row 0 (degenerate initial state).
rows_p = volume_schedule(pr[:jp + 1, 6], KP + 1)[1:]

# Spreading: full range, starting at js.
rows_s = (
    volume_schedule(sp[js:, 6], KS) + js
    if HAS_SPREADING
    else np.array([], dtype=int)
)

if len(set(rows_p)) < KP:
    print(
        f"WARNING: pinning schedule has "
        f"{KP - len(set(rows_p))} duplicate rows"
    )

if HAS_SPREADING and len(set(rows_s)) < KS:
    print(
        f"WARNING: spreading schedule has "
        f"{KS - len(set(rows_s))} duplicate rows"
    )


# --------------- Frame profiles ---------------

def half_profile(row):
    lam_ = row[4]
    pf = os.path.join(WORK, "prof.txt")

    subprocess.run(
        [RUN, "1", f"{float(row[5]):.12g}", pf],
        check=True,
        capture_output=True
    )

    p = np.loadtxt(pf)

    V, x, z = p[:, 6], p[:, 0], p[:, 1]

    k = max(int(np.argmin(np.abs(row[6] - V))), 2)

    xf, zf, kk = x[k], z[k], k

    for j in (k - 1, k):

        lo, hi = sorted((V[j], V[j + 1]))

        if j >= 1 and lo <= row[6] <= hi and hi > lo:

            t = (row[6] - V[j]) / (V[j + 1] - V[j])

            xf = x[j] + t * (x[j + 1] - x[j])
            zf = z[j] + t * (z[j + 1] - z[j])

            kk = j
            break

    xs = np.concatenate([x[:kk + 1], [xf]])
    zs = np.concatenate([z[:kk + 1], [zf]])

    xs, ys = xs / lam_, (zs - zf) / lam_

    t = np.concatenate([[0], np.cumsum(np.hypot(
        np.diff(xs), np.diff(ys)
    ))])

    tt = np.linspace(0, t[-1], M)

    return np.interp(tt, t, xs), np.interp(tt, t, ys)


frames = []

for ph, rows, tab in (("p", rows_p, pr), ("s", rows_s, sp)):

    for i in rows:

        x, y = half_profile(tab[i])

        frames.append(dict(
            ph=ph,
            x=x,
            y=y,
            row=int(i),
            V=tab[i, 6] / tab[i, 4] ** 3,
            r0=tab[i, 0] / tab[i, 4],
            phi2=(1 - tab[i, 2] / np.pi) ** 2,
        ))

K = len(frames)


# --------------- Physical animation clock ---------------

vpin = np.maximum.accumulate(
    pr[:jp + 1, 6] / pr[:jp + 1, 4] ** 3
)

vspr = (
    np.maximum.accumulate(sp[js:, 6] / sp[js:, 4] ** 3)
    if HAS_SPREADING
    else np.array([], dtype=float)
)

pin_duration = (vpin[-1] - vpin[0]) / FILL_RATE

spr_duration = (
    (vspr[-1] - vspr[0]) / FILL_RATE
    if HAS_SPREADING
    else 0.0
)

TRANSITION_EPS = 1e-6 if HAS_SPREADING else 0.0
DURATION = pin_duration + spr_duration + TRANSITION_EPS

frame_times = []

for fr in frames:

    if fr["ph"] == "p":

        frame_times.append(
            (vpin[fr["row"]] - vpin[0]) / FILL_RATE
        )

    elif HAS_SPREADING:

        frame_times.append(
            pin_duration + TRANSITION_EPS
            + (vspr[fr["row"] - js] - vspr[0]) / FILL_RATE
        )

frame_times = np.asarray(frame_times)

if np.any(np.diff(frame_times) <= 0):
    raise ValueError(
        "Frame times are not strictly increasing; check simulation volumes"
    )

FRAME_KTS = [f"{t / DURATION:.9f}" for t in frame_times]

print(f"frames: {K}   duration: {DURATION:.2f}s")

print(
    f"fill rate: {FILL_RATE:g} lambda^3/s "
    f"(pinning {pin_duration:.2f}s, spreading {spr_duration:.2f}s)"
)


# --------------- Figure ---------------

fig = plt.figure(figsize=(14, 6))

gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1, 1],
    wspace=0.22
)

ax_phi = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])


# ---- RIGHT: bubble profile ----

xmax_p = max(f["x"].max() for f in frames)
ymax_p = max(f["y"].max() for f in frames)

xlim_b = xmax_p + 0.6

DEFAULT_CONFIGURATION = (
    np.isclose(PHI2_TARGET, 0.7 ** 2)
    and np.isclose(RC_TARGET, 1.0)
)

if DEFAULT_CONFIGURATION:

    right_xlim = (-2.0, 2.0)
    right_ylim = (-0.5, 2.5)

else:

    right_ylim = (-CAV - 0.3, ymax_p + 0.4)
    right_xlim = (-xlim_b, xlim_b)

right_y_range = right_ylim[1] - right_ylim[0]
right_x_range = right_xlim[1] - right_xlim[0]

common_aspect = right_y_range / right_x_range


for a, b, c, d_ in [
    (-xlim_b, -RC, 0, 0),
    (RC, xlim_b, 0, 0),
    (-RC, -RC, -CAV, 0),
    (RC, RC, -CAV, 0),
    (-RC, RC, -CAV, -CAV)
]:

    ax_b.plot(
        [a, b], [c, d_],
        "k-",
        lw=1.5,
        zorder=4,
        clip_on=False
    )

ax_b.set_xlim(right_xlim)
ax_b.set_ylim(right_ylim)
ax_b.set_aspect("equal", adjustable="box")

ax_b.set_xlabel(r"$r/\lambda$")
ax_b.set_ylabel(
    r"$\frac{ z }{\lambda}$",
    rotation=0,
    labelpad=15,
    size=22
)

ax_b.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True
)

bub_line, = ax_b.plot(
    [],
    [],
    lw=2.5,
    zorder=5,
    clip_on=False
)

bub_line.set_gid("bubble")


# ---- LEFT: parameter-space plot ----

ax_phi.set_xlim([0, 4])
ax_phi.set_ylim([0, 1.05])
ax_phi.set_yticks([0, 0.25, 0.5, 0.75, 1])

ax_phi.set_xlabel(r"$r_0/\lambda$")
ax_phi.set_ylabel(
    r"$\frac{\phi_0^2}{\pi^2}$",
    rotation=0,
    labelpad=10,
    size=22
)

ax_phi.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True
)

ax_phi.set_box_aspect(common_aspect)

ax_phi.axvspan(
    R_GUIDE,
    R_BAND_END,
    color="whitesmoke",
    zorder=0
)

xx = np.linspace(0, 4, 500)

# Keep the paper's very faint reference line.
ax_phi.plot(
    xx,
    xx / 3.5,
    c="k",
    lw=0.15,
    zorder=1
)


# ---- Decile trajectories ----

n_decile = 0

for fpath in sorted(glob.glob(SIM + "loop_ang*.txt")):

    try:
        dl = np.loadtxt(fpath)
    except Exception:
        continue

    if dl.ndim < 2 or len(dl) < 2:
        continue

    dl = dl.copy()

    indVol = int(np.argmax(dl[:, 6]))
    angl = 1 - dl[indVol, 2] / np.pi

    extremInd = int(np.argmax(dl[:, 0]))

    dl[0, 0] = 0

    if round(angl * 100) == 100:
        extremInd = 1
        dl[extremInd, 0] = 3.831698723

    y0 = (1 - dl[0, 2] / np.pi) ** 2
    ye = (1 - dl[extremInd, 2] / np.pi) ** 2

    x0 = dl[0, 0]
    xe = dl[extremInd, 0]

    # Omit near-horizontal grey reference segments.
    if abs(ye - y0) < 0.015:
        continue

    if round(angl * 100) % 20 != 10:
        continue

    n_decile += 1

    ax_phi.plot(
        [x0, xe],
        [y0, ye],
        c="grey",
        lw=0.7,
        zorder=2
    )

    if round(angl * 100) == 90:

        ax_phi.plot(
            [1.5, xe],
            [y0, ye],
            c="blue",
            zorder=2
        )

    for hei in range(5):

        target_h = (hei + 1) * dl[indVol, 1] / 5

        heiInd = int(
            np.argmin(
                np.abs(target_h - dl[:indVol + 1, 1])
            )
        )

        heiPrev = max(0, heiInd - 1)

        col = "grey" if hei < 4 else "b"

        ax_phi.annotate(
            "",
            xy=(
                dl[heiInd, 0],
                (1 - dl[heiInd, 2] / np.pi) ** 2
            ),
            xytext=(
                dl[heiPrev, 0],
                (1 - dl[heiPrev, 2] / np.pi) ** 2
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=0.7,
                mutation_scale=8
            ),
            zorder=3
        )

print(f"plotted {n_decile} decile arcs with arrowheads")


# ---- Blue reference curve ----

try:

    with open(SIM + BLUE_REF, encoding="utf-8") as f:
        df_bin = np.loadtxt(f)

    ax_phi.plot(
        df_bin[:, 0],
        (1 - df_bin[:, 2] / np.pi) ** 2,
        c="b",
        lw=0.9,
        zorder=3
    )

except FileNotFoundError:

    print(f"WARNING: {BLUE_REF} not found")


# ---- Static selected trajectory with arrows ----

def plot_static_path(x, y, color="grey"):

    ax_phi.plot(
        x,
        y,
        color=color,
        lw=0.9,
        zorder=3
    )

    if len(x) < 3:
        return

    for idx in np.linspace(
        1,
        len(x) - 1,
        min(7, len(x) - 1),
        dtype=int
    ):

        prev = max(
            0,
            idx - max(1, len(x) // 35)
        )

        ax_phi.annotate(
            "",
            xy=(x[idx], y[idx]),
            xytext=(x[prev], y[prev]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=0.8,
                mutation_scale=9
            ),
            zorder=4
        )


plot_static_path(r0_p_all, ph_p_all)

if HAS_SPREADING:
    plot_static_path(r0_s_all, ph_s_all)


# ---- Red detachment boundary ----

detachment_file = os.path.join(SIM, "BinRadMaxVol.txt")

if os.path.isfile(detachment_file):

    det = np.loadtxt(detachment_file)

    if det.ndim == 2 and len(det) > 100:

        min_len = 100

        min_ang = np.array([
            np.min(
                1 - det[i:i + min_len, 2] / np.pi
            )
            for i in range(
                0,
                len(det[:, 2]) - min_len,
                min_len
            )
        ])

        rad = np.array([
            det[i + min_len // 2, 0]
            for i in range(
                0,
                len(det[:, 2]) - min_len,
                min_len
            )
        ])

        side = min_ang >= 1 - 1e-3

        ax_phi.plot(
            rad[~side],
            min_ang[~side] ** 2,
            c="r",
            lw=1.2,
            zorder=5,
            clip_on=False
        )

        ax_phi.plot(
            rad[side],
            min_ang[side] ** 2,
            c="r",
            lw=1.2,
            ls="dashed",
            zorder=5,
            clip_on=False
        )

else:

    print(
        f"WARNING: {detachment_file} not found; "
        "red detachment line omitted"
    )


# ---- Animated marker ----

left_x_range = 4.0
left_y_range = 1.05

rx_data = 0.045
ry_data = rx_data * left_y_range / (
    left_x_range * common_aspect
)

dot = Ellipse(
    (RC, 0.0),
    width=2 * rx_data,
    height=2 * ry_data,
    facecolor="r",
    edgecolor="black",
    linewidth=1.0,
    zorder=5,
    clip_on=False
)

ax_phi.add_patch(dot)
dot.set_gid("magdot")

fig.tight_layout()


def update(k):

    fr = frames[k]

    X = np.concatenate([
        -fr["x"][::-1],
        fr["x"]
    ])

    Y = np.concatenate([
        fr["y"][::-1],
        fr["y"]
    ])

    bub_line.set_color(
        "r" if fr["ph"] == "p" else "b"
    )

    bub_line.set_data(X, Y)

    dot.center = (fr["r0"], fr["phi2"])

    dot.set_facecolor(
        "r" if fr["ph"] == "p" else "b"
    )


# --------------- Render frames ---------------

WORK_SVG = tempfile.mkdtemp(prefix="animsg_")

print(f"rendering {K} frame SVGs into {WORK_SVG}")

update(0)
fig.canvas.draw()

tight_bbox = fig.get_tightbbox(
    fig.canvas.get_renderer()
)

svg_files = []

for k in range(K):

    update(k)

    p = os.path.join(
        WORK_SVG,
        f"f{k:04d}.svg"
    )

    fig.savefig(
        p,
        format="svg",
        bbox_inches=tight_bbox,
        transparent=TRANSPARENT
    )

    svg_files.append(p)

    if (k + 1) % 20 == 0:
        print(f"  {k + 1}/{K}")


# --------------- Extract SVG paths ---------------

_NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def _fmt(m):

    v = round(float(m.group(0)), PREC)

    t = f"{v:.{PREC}f}".rstrip("0").rstrip(".")

    return "0" if t in ("", "-0") else t


def compact(d_str):
    return " ".join(
        _NUM_RE.sub(_fmt, d_str).split()
    )


def find_group(root, gid):

    for el in root.iter():

        if el.get("id") == gid:
            return el

    return None


def group_path(root, gid):

    g = find_group(root, gid)

    if g is None:
        return None

    for el in g.iter():

        if el.tag.endswith("path") and el.get("d"):

            at = dict(el.attrib)

            at.pop("clip-path", None)

            at["d"] = compact(at["d"])

            return at

    return None


def path_bbox_center(d_string):
    """Bounding-box centre of a path with absolute x,y coordinates."""

    nums = [
        float(x)
        for x in _NUM_RE.findall(d_string)
    ]

    if len(nums) < 2:
        return (0.0, 0.0)

    xs = nums[0::2]
    ys = nums[1::2]

    return (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2
    )


print("extracting frame paths …")

ex = []

for p in svg_files:

    root_k = ET.parse(p).getroot()

    ex.append({
        g: group_path(root_k, g)
        for g in ("bubble", "magdot")
    })


dot_centers = [
    path_bbox_center(ex[k]["magdot"]["d"])
    if ex[k]["magdot"]
    else (0.0, 0.0)
    for k in range(K)
]

bubble_ds = [
    ex[k]["bubble"]["d"]
    if ex[k]["bubble"]
    else None
    for k in range(K)
]


# --------------- Build the animation ---------------

DUR = f"{DURATION:.6f}s"
FREEZE = (REPEAT == "1")
ALPHA = pin_duration / DURATION


def add_visibility(el, k):

    if k == 0:

        vals = ["visible", "hidden"]
        kts = [0, frame_times[1] / DURATION]

    elif k == K - 1:

        vals = ["hidden", "visible"]
        kts = [0, frame_times[k] / DURATION]

    else:

        vals = ["hidden", "visible", "hidden"]

        kts = [
            0,
            frame_times[k] / DURATION,
            frame_times[k + 1] / DURATION
        ]

    an = ET.SubElement(
        el,
        f"{{{SVG_NS}}}animate"
    )

    an.set("attributeName", "visibility")
    an.set("values", ";".join(vals))

    an.set(
        "keyTimes",
        ";".join(f"{t:.9f}" for t in kts)
    )

    an.set("calcMode", "discrete")
    an.set("dur", DUR)
    an.set("begin", "0s")
    an.set("repeatCount", REPEAT)

    if FREEZE:
        an.set("fill", "freeze")


tree = ET.parse(svg_files[0])
root = tree.getroot()

for ch in list(root):

    if ch.tag.endswith("metadata"):
        root.remove(ch)


groups = {}

for gid in ("bubble", "magdot"):

    g = find_group(root, gid)

    if g is None:
        sys.exit(
            f"ERROR: group '{gid}' not found in the SVG"
        )

    for ch in list(g):
        g.remove(ch)

    groups[gid] = g


# ---- Verify morph viability ----

if args.morph:

    nL = [
        d.count("L")
        for d in bubble_ds
        if d
    ]

    if len(set(nL)) != 1:

        print(
            "WARNING: --morph disabled "
            f"(vertex counts differ: {sorted(set(nL))})"
        )

        args.morph = False

    else:

        print(
            f"  bubble morph: {nL[0]} vertices / frame"
        )


# ---- Bubble outline ----

if args.morph:

    attrs = dict(ex[-1]["bubble"])

    # Static SVG fallback = final frame.
    attrs["d"] = bubble_ds[-1]

    bub_el = ET.SubElement(
        groups["bubble"],
        f"{{{SVG_NS}}}path",
        attrib=attrs
    )

    # The first sampled pinning frame is held from t=0.
    d_vals = ";".join(
        [bubble_ds[0]] + bubble_ds[1:] + [bubble_ds[-1]]
    )

    kts = (
        [0.0]
        + [t / DURATION for t in frame_times[1:]]
        + [1.0]
    )

    an = ET.SubElement(
        bub_el,
        f"{{{SVG_NS}}}animate"
    )

    an.set("attributeName", "d")
    an.set("values", d_vals)

    an.set(
        "keyTimes",
        ";".join(f"{t:.9f}" for t in kts)
    )

    an.set("calcMode", "linear")
    an.set("dur", DUR)
    an.set("begin", "0s")
    an.set("repeatCount", REPEAT)

    if FREEZE:
        an.set("fill", "freeze")

    an2 = ET.SubElement(
        bub_el,
        f"{{{SVG_NS}}}animate"
    )

    an2.set("attributeName", "stroke")
    an2.set("values", "red;blue")
    an2.set("keyTimes", f"0;{ALPHA:.6f}")
    an2.set("calcMode", "discrete")
    an2.set("dur", DUR)
    an2.set("begin", "0s")
    an2.set("repeatCount", REPEAT)

    if FREEZE:
        an2.set("fill", "freeze")

else:

    for k in range(K):

        at = ex[k]["bubble"]

        if at is None:
            sys.exit(
                f"ERROR: frame {k} has no bubble path"
            )

        el = ET.SubElement(
            groups["bubble"],
            f"{{{SVG_NS}}}path",
            attrib=at
        )

        el.set(
            "visibility",
            "visible" if (k == K - 1) else "hidden"
        )

        add_visibility(el, k)


# ---- Dot: animateTransform on a single path ----

attrs = dict(ex[-1]["magdot"])
attrs.setdefault("fill", "#0000ff")

dot_el = ET.SubElement(
    groups["magdot"],
    f"{{{SVG_NS}}}path",
    attrib=attrs
)

cx_last, cy_last = dot_centers[-1]

dot_vals = [
    f"{(dot_centers[k][0] - cx_last):.{PREC}f},"
    f"{(dot_centers[k][1] - cy_last):.{PREC}f}"
    for k in range(K)
]

# Anchor the first sampled marker position at keyTime 0.
dot_vals = (
    [dot_vals[0]]
    + dot_vals[1:]
    + ["0,0"]
)

dot_kts = (
    [0.0]
    + [t / DURATION for t in frame_times[1:]]
    + [1.0]
)

at_el = ET.SubElement(
    dot_el,
    f"{{{SVG_NS}}}animateTransform"
)

at_el.set("attributeName", "transform")
at_el.set("type", "translate")
at_el.set("values", ";".join(dot_vals))

at_el.set(
    "keyTimes",
    ";".join(f"{t:.6f}" for t in dot_kts)
)

at_el.set("calcMode", "linear")
at_el.set("dur", DUR)
at_el.set("begin", "0s")
at_el.set("repeatCount", REPEAT)

if FREEZE:
    at_el.set("fill", "freeze")


fill_el = ET.SubElement(
    dot_el,
    f"{{{SVG_NS}}}animate"
)

fill_el.set("attributeName", "fill")
fill_el.set("values", "red;blue")
fill_el.set("keyTimes", f"0;{ALPHA:.6f}")
fill_el.set("calcMode", "discrete")
fill_el.set("dur", DUR)
fill_el.set("begin", "0s")
fill_el.set("repeatCount", REPEAT)

if FREEZE:
    fill_el.set("fill", "freeze")


# --------------- Write SVG ---------------

tree.write(
    OUT,
    encoding="utf-8",
    xml_declaration=True
)

kb = os.path.getsize(OUT) / 1024

mode_str = "morph" if args.morph else "flipbook"

print(
    f"wrote {OUT} ({kb:.0f} kB, {K} frames, "
    f"mode={mode_str}, dur={DURATION:.2f}s, "
    f"fill_rate={FILL_RATE:g} lambda^3/s)"
)

plt.close(fig)

shutil.rmtree(WORK_SVG, ignore_errors=True)

if args.work is None:
    shutil.rmtree(WORK, ignore_errors=True)
