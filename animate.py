#!/usr/bin/env python3
"""Animated SVG of bubble evolution.

Default: phi/pi=0.7, cavity radius=1, output=plots/growing.svg.
"""

import os
import sys
import glob
import subprocess
import tempfile
import shutil
import re
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path
import xml.etree.ElementTree as ET


# Matplotlib / SVG settings
plt.rcParams.update({"text.usetex": True, "font.size": 14})
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["path.simplify"] = False
plt.rcParams["svg.hashsalt"] = "animsvg"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


# Command-line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo", nargs="?", default=".")
parser.add_argument("work", nargs="?", default=None)
parser.add_argument("out", nargs="?", default="plots/growing.svg")
parser.add_argument("--fill-rate", type=float, default=1.0)
parser.add_argument("--phi2", type=float, default=0.7**2,
                    help="target phi^2/pi^2")
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--single-shot", action="store_true")
parser.add_argument("--opaque", action="store_true")
parser.add_argument("--morph", action=argparse.BooleanOptionalAction,
                    default=True)
args = parser.parse_args()

REPO, OUT = args.repo, args.out
FILL_RATE = args.fill_rate

if FILL_RATE <= 0:
    parser.error("--fill-rate must be positive")

if args.radius <= 0:
    parser.error("--radius must be positive")

if not 0 <= args.phi2 <= 1:
    parser.error("--phi2 must be between 0 and 1")

REPEAT = "1" if args.single_shot else "indefinite"
TRANSPARENT = not args.opaque

WORK = args.work or tempfile.mkdtemp(prefix="animwork_")
os.makedirs(WORK, exist_ok=True)

SIM = os.path.join(REPO, "simData")
RUN = os.path.join(REPO, "run")

if not os.path.isdir(SIM):
    parser.error(f"simulation data directory not found: {SIM}")

RC_TARGET = args.radius
PHI2_TARGET = args.phi2

CAV = 0.3
KP, KS, M = 22, 118, 48
R_GUIDE, R_BAND_END = 3.219, 3.832
BLUE_REF = "BinAngMaxVol.txt"
PREC = 2


# ----------------------------------------------------------------------
# Select the radius and contact-angle datasets
# ----------------------------------------------------------------------

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

        data = data[:np.argmax(data[:, 6]) + 1]

        inside_angle = float(data[0, 2]) * 180 / np.pi
        phi2 = (1 - float(data[0, 2]) / np.pi)**2
        phi_w = 1 - float(data[-1, 2]) / np.pi

        if np.isfinite(phi2) and np.isfinite(phi_w) and 0 <= phi2 <= 1:
            ang_candidates.append(
                (path, data, inside_angle, phi2, phi_w)
            )

    except (OSError, ValueError, IndexError):
        continue


if not rad_candidates:
    raise FileNotFoundError(f"No readable loop_rad*.txt datasets in {SIM}")

if not ang_candidates:
    raise FileNotFoundError(f"No readable loop_ang*.txt datasets in {SIM}")


rad_path, pr, RC, phi_pin = min(
    rad_candidates, key=lambda item: abs(item[2] - RC_TARGET)
)

ang_path, candidate_sp, CONTACT_ANGLE_DEG, PHI2_SELECTED, PHI_W = min(
    ang_candidates, key=lambda item: abs(item[3] - PHI2_TARGET)
)


pin_rows = np.flatnonzero(phi_pin <= PHI_W + 1e-12)
spread_rows = np.flatnonzero(candidate_sp[:, 0] >= RC)

if len(pin_rows) and len(spread_rows):
    jp = int(pin_rows[0])
    js = int(spread_rows[0])
    sp = candidate_sp
    HAS_SPREADING = True
else:
    jp = len(pr) - 1
    js = 0
    sp = np.empty((0, pr.shape[1]))
    HAS_SPREADING = False

    if not len(pin_rows):
        print("WARNING: pinned trajectory does not reach the selected angle; "
              "animating pinning only.")
    else:
        print("WARNING: spreading trajectory does not reach the selected "
              "radius; animating pinning only.")


print(f"cavity radius: requested {RC_TARGET:g}, selected {RC:g}")
print(f"phi^2/pi^2: requested {PHI2_TARGET:.6g}, selected {PHI2_SELECTED:.6g}")
print(f"inside angle: {CONTACT_ANGLE_DEG:.4g} degrees")
print(f"pinned dataset: {os.path.basename(rad_path)}")
print(f"spreading dataset: {os.path.basename(ang_path) if HAS_SPREADING else 'skipped'}")


r0_p_all = pr[:jp + 1, 0] / pr[:jp + 1, 4]
ph_p_all = (1 - pr[:jp + 1, 2] / np.pi)**2

r0_s_all = sp[js:, 0] / sp[js:, 4]
ph_s_all = (1 - sp[js:, 2] / np.pi)**2


# ----------------------------------------------------------------------
# Select frames at approximately uniform volume intervals
# ----------------------------------------------------------------------

def volume_schedule(volume, n):
    """Return indices sampled at uniform volume intervals."""

    volume = np.maximum.accumulate(np.asarray(volume, dtype=float))
    unique_v, first_rows = np.unique(volume, return_index=True)

    if len(unique_v) < n:
        raise ValueError(
            f"Cannot schedule {n} frames: only {len(unique_v)} "
            "distinct volume values are available."
        )

    targets = np.linspace(unique_v[0], unique_v[-1], n)
    pos = np.clip(np.searchsorted(unique_v, targets), 0, len(unique_v) - 1)

    duplicates = int(np.sum(np.diff(pos) == 0))

    if duplicates:
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
                "Not enough distinct volume samples; reduce KP/KS."
            )

        print(f"  adjusted {duplicates} duplicate volume targets")

    indices = first_rows[pos]

    if np.any(np.diff(volume[indices]) <= 0):
        raise ValueError("Selected frame volumes are not strictly increasing.")

    return indices


if KP > jp:
    raise ValueError(f"Not enough pinned rows for {KP} frames.")

if HAS_SPREADING and KS > len(sp) - js:
    raise ValueError(f"Not enough spreading rows for {KS} frames.")


rows_p = volume_schedule(pr[:jp + 1, 6], KP + 1)[1:]

rows_s = (
    volume_schedule(sp[js:, 6], KS) + js
    if HAS_SPREADING else np.array([], dtype=int)
)


# ----------------------------------------------------------------------
# Generate bubble profiles
# ----------------------------------------------------------------------

def half_profile(row):
    """Run the profile solver and interpolate the interface to the row volume."""

    lam = row[4]
    profile_file = os.path.join(WORK, "prof.txt")

    subprocess.run(
        [RUN, "1", f"{float(row[5]):.12g}", profile_file],
        check=True,
        capture_output=True
    )

    profile = np.loadtxt(profile_file)

    volume = profile[:, 6]
    x, z = profile[:, 0], profile[:, 1]

    k = max(int(np.argmin(np.abs(row[6] - volume))), 2)
    xf, zf, kk = x[k], z[k], k

    for j in (k - 1, k):
        lo, hi = sorted((volume[j], volume[j + 1]))

        if j >= 1 and lo <= row[6] <= hi and hi > lo:
            t = (row[6] - volume[j]) / (volume[j + 1] - volume[j])
            xf = x[j] + t * (x[j + 1] - x[j])
            zf = z[j] + t * (z[j + 1] - z[j])
            kk = j
            break

    xs = np.concatenate([x[:kk + 1], [xf]]) / lam
    ys = (np.concatenate([z[:kk + 1], [zf]]) - zf) / lam

    distance = np.concatenate([
        [0],
        np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))
    ])

    samples = np.linspace(0, distance[-1], M)

    return (
        np.interp(samples, distance, xs),
        np.interp(samples, distance, ys)
    )


frames = []

for phase, rows, table in (
    ("p", rows_p, pr),
    ("s", rows_s, sp)
):
    for i in rows:
        x, y = half_profile(table[i])

        frames.append({
            "ph": phase,
            "x": x,
            "y": y,
            "row": int(i),
            "V": table[i, 6] / table[i, 4]**3,
            "r0": table[i, 0] / table[i, 4],
            "phi2": (1 - table[i, 2] / np.pi)**2
        })


K = len(frames)


# ----------------------------------------------------------------------
# Animation timing: constant dimensionless volume filling rate
# ----------------------------------------------------------------------

vpin = np.maximum.accumulate(pr[:jp + 1, 6] / pr[:jp + 1, 4]**3)

vspr = (
    np.maximum.accumulate(sp[js:, 6] / sp[js:, 4]**3)
    if HAS_SPREADING else np.array([])
)

pin_duration = (vpin[-1] - vpin[0]) / FILL_RATE
spr_duration = (
    (vspr[-1] - vspr[0]) / FILL_RATE
    if HAS_SPREADING else 0.0
)

TRANSITION_EPS = 1e-6 if HAS_SPREADING else 0.0
DURATION = pin_duration + spr_duration + TRANSITION_EPS

frame_times = []

for fr in frames:
    if fr["ph"] == "p":
        frame_times.append(
            (vpin[fr["row"]] - vpin[0]) / FILL_RATE
        )
    else:
        frame_times.append(
            pin_duration + TRANSITION_EPS
            + (vspr[fr["row"] - js] - vspr[0]) / FILL_RATE
        )

frame_times = np.asarray(frame_times)

if np.any(np.diff(frame_times) <= 0):
    raise ValueError("Frame times are not strictly increasing.")

print(f"Frames: {K}; duration: {DURATION:.2f} s")
print(f"Fill rate: {FILL_RATE:g} lambda^3/s")


# ----------------------------------------------------------------------
# Figure: paper dimensions (10 x 3 inches)
# ----------------------------------------------------------------------

fig = plt.figure(figsize=(10, 3))
gs = fig.add_gridspec(1, 2, wspace=0.22)

ax_phi = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])


# RIGHT: bubble profile and cavity

xmax_p = max(f["x"].max() for f in frames)
ymax_p = max(f["y"].max() for f in frames)

DEFAULT_CONFIGURATION = (
    np.isclose(PHI2_TARGET, 0.7**2)
    and np.isclose(RC_TARGET, 1.0)
)

if DEFAULT_CONFIGURATION:
    right_xlim = (-2.0, 2.0)
    right_ylim = (-0.5, 2.5)
else:
    right_xlim = (-(xmax_p + 0.6), xmax_p + 0.6)
    right_ylim = (-CAV - 0.3, ymax_p + 0.4)

common_aspect = (
    (right_ylim[1] - right_ylim[0])
    / (right_xlim[1] - right_xlim[0])
)

# Substrate outside the cavity
ax_b.plot(
    [right_xlim[0], -RC], [0, 0],
    "k-", lw=1.5, zorder=4, clip_on=False
)
ax_b.plot(
    [RC, right_xlim[1]], [0, 0],
    "k-", lw=1.5, zorder=4, clip_on=False
)

# U-shaped cavity boundary
cavity = Path(
    [(-RC, 0), (-RC, -CAV), (RC, -CAV), (RC, 0)],
    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
)

ax_b.add_patch(PathPatch(
    cavity, fill=False, color="k", lw=1.5,
    zorder=4, clip_on=False
))

ax_b.set(
    xlim=right_xlim,
    ylim=right_ylim
)
ax_b.set_aspect("equal", adjustable="box")
ax_b.set_xlabel(r"$r/\lambda$")
ax_b.set_ylabel(
    r"$\frac{z}{\lambda}$",
    rotation=0, labelpad=10, size=22
)
ax_b.tick_params(which="both", direction="in", top=True, right=True)

bubble_line, = ax_b.plot([], [], lw=2.5, zorder=5, clip_on=False)
bubble_line.set_gid("bubble")


# LEFT: parameter-space plot

ax_phi.set_xlim(0, 4)
ax_phi.set_ylim(0, 1.05)
ax_phi.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax_phi.set_xlabel(r"$r_0/\lambda$")
ax_phi.set_ylabel(
    r"$\frac{\phi_0^2}{\pi^2}$",
    rotation=0, labelpad=10, size=22
)
ax_phi.tick_params(which="both", direction="in", top=True, right=True)
ax_phi.set_box_aspect(common_aspect)
ax_phi.axvspan(R_GUIDE, R_BAND_END, color="whitesmoke", zorder=0)

xx = np.linspace(0, 4, 500)
ax_phi.plot(xx, xx / 3.5, color="k", lw=0.15, zorder=1)


# Grey trajectories and directional arrows

n_decile = 0

for filename in sorted(glob.glob(os.path.join(SIM, "loop_ang*.txt"))):
    try:
        data = np.loadtxt(filename)
    except (OSError, ValueError):
        continue

    if data.ndim != 2 or len(data) < 2:
        continue

    data = data.copy()
    indVol = int(np.argmax(data[:, 6]))
    angle = 1 - data[indVol, 2] / np.pi
    end = int(np.argmax(data[:, 0]))

    data[0, 0] = 0

    if round(angle * 100) == 100:
        end = 1
        data[end, 0] = 3.831698723

    x0, xe = data[0, 0], data[end, 0]
    y0 = (1 - data[0, 2] / np.pi)**2
    ye = (1 - data[end, 2] / np.pi)**2

    if abs(ye - y0) < 0.015 or round(angle * 100) % 20 != 10:
        continue

    n_decile += 1
    ax_phi.plot([x0, xe], [y0, ye], color="grey", lw=0.7, zorder=2)

    if round(angle * 100) == 90:
        ax_phi.plot([1.5, xe], [y0, ye], color="blue", zorder=2)

    for hei in range(5):
        target_h = (hei + 1) * data[indVol, 1] / 5
        i = int(np.argmin(np.abs(target_h - data[:indVol + 1, 1])))
        prev = max(0, i - 1)

        ax_phi.annotate(
            "",
            xy=(data[i, 0], (1 - data[i, 2] / np.pi)**2),
            xytext=(data[prev, 0], (1 - data[prev, 2] / np.pi)**2),
            arrowprops={
                "arrowstyle": "->",
                "color": "grey" if hei < 4 else "b",
                "lw": 0.7,
                "mutation_scale": 8
            },
            zorder=3
        )

print(f"Plotted {n_decile} decile trajectories.")


# Blue reference curve

blue_file = os.path.join(SIM, BLUE_REF)

if os.path.isfile(blue_file):
    blue = np.loadtxt(blue_file)
    ax_phi.plot(
        blue[:, 0],
        (1 - blue[:, 2] / np.pi)**2,
        color="b", lw=0.9, zorder=3
    )
else:
    print(f"WARNING: {BLUE_REF} not found.")


# Selected trajectory: static grey line with arrows

def plot_static_path(x, y):
    ax_phi.plot(x, y, color="grey", lw=0.9, zorder=3)

    if len(x) < 3:
        return

    for i in np.linspace(1, len(x) - 1, min(7, len(x) - 1), dtype=int):
        prev = max(0, i - max(1, len(x) // 35))

        ax_phi.annotate(
            "",
            xy=(x[i], y[i]),
            xytext=(x[prev], y[prev]),
            arrowprops={
                "arrowstyle": "->",
                "color": "grey",
                "lw": 0.8,
                "mutation_scale": 9
            },
            zorder=4
        )


plot_static_path(r0_p_all, ph_p_all)

if HAS_SPREADING:
    plot_static_path(r0_s_all, ph_s_all)


# Red detachment boundary, using the paper's solid/dashed convention

detachment_file = os.path.join(SIM, "BinRadMaxVol.txt")

if os.path.isfile(detachment_file):
    det = np.loadtxt(detachment_file)

    if det.ndim == 2 and len(det) > 100:
        block = 100
        starts = range(0, len(det) - block, block)

        min_ang = np.array([
            np.min(1 - det[i:i + block, 2] / np.pi)
            for i in starts
        ])

        rad = np.array([
            det[i + block // 2, 0]
            for i in starts
        ])

        side = min_ang >= 1 - 1e-3

        ax_phi.plot(
            rad[~side], min_ang[~side]**2,
            color="r", lw=1.2, zorder=5, clip_on=False
        )
        ax_phi.plot(
            rad[side], min_ang[side]**2,
            color="r", lw=1.2, ls="dashed",
            zorder=5, clip_on=False
        )
else:
    print("WARNING: BinRadMaxVol.txt not found; detachment line omitted.")


# Animated marker

rx_data = 0.045
ry_data = rx_data * 1.05 / (4 * common_aspect)

dot = Ellipse(
    (RC, 0),
    width=2 * rx_data,
    height=2 * ry_data,
    facecolor="r",
    edgecolor="black",
    linewidth=1,
    zorder=5,
    clip_on=False
)

ax_phi.add_patch(dot)
dot.set_gid("magdot")

fig.tight_layout(pad=0)


def update(k):
    fr = frames[k]

    x = np.concatenate([-fr["x"][::-1], fr["x"]])
    y = np.concatenate([fr["y"][::-1], fr["y"]])

    color = "r" if fr["ph"] == "p" else "b"

    bubble_line.set_data(x, y)
    bubble_line.set_color(color)

    dot.center = (fr["r0"], fr["phi2"])
    dot.set_facecolor(color)


# ----------------------------------------------------------------------
# Render individual SVG frames
# ----------------------------------------------------------------------

WORK_SVG = tempfile.mkdtemp(prefix="animsg_")
print(f"Rendering {K} SVG frames into {WORK_SVG}")

update(0)
fig.canvas.draw()
tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())

svg_files = []

for k in range(K):
    update(k)

    filename = os.path.join(WORK_SVG, f"f{k:04d}.svg")

    fig.savefig(
        filename,
        format="svg",
        bbox_inches=tight_bbox,
        transparent=TRANSPARENT
    )

    svg_files.append(filename)

    if (k + 1) % 20 == 0:
        print(f"  {k + 1}/{K}")


# ----------------------------------------------------------------------
# Extract paths from the rendered SVG frames
# ----------------------------------------------------------------------

NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def compact(path_data):
    def fmt(match):
        value = round(float(match.group()), PREC)
        result = f"{value:.{PREC}f}".rstrip("0").rstrip(".")
        return "0" if result in ("", "-0") else result

    return " ".join(NUM_RE.sub(fmt, path_data).split())


def find_group(root, gid):
    return next(
        (el for el in root.iter() if el.get("id") == gid),
        None
    )


def group_path(root, gid):
    group = find_group(root, gid)

    if group is None:
        return None

    for element in group.iter():
        if element.tag.endswith("path") and element.get("d"):
            attrs = dict(element.attrib)
            attrs.pop("clip-path", None)
            attrs["d"] = compact(attrs["d"])
            return attrs

    return None


def path_bbox_center(path_data):
    values = [float(x) for x in NUM_RE.findall(path_data)]

    if len(values) < 2:
        return 0.0, 0.0

    xs, ys = values[0::2], values[1::2]

    return (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2
    )


print("Extracting SVG paths...")

extracted = []

for filename in svg_files:
    root = ET.parse(filename).getroot()
    extracted.append({
        gid: group_path(root, gid)
        for gid in ("bubble", "magdot")
    })


dot_centers = [
    path_bbox_center(frame["magdot"]["d"])
    if frame["magdot"] else (0.0, 0.0)
    for frame in extracted
]

bubble_paths = [
    frame["bubble"]["d"] if frame["bubble"] else None
    for frame in extracted
]


# ----------------------------------------------------------------------
# Assemble the animated SVG
# ----------------------------------------------------------------------

DUR = f"{DURATION:.6f}s"
FREEZE = REPEAT == "1"
ALPHA = pin_duration / DURATION


def add_visibility(element, k):
    if k == 0:
        values = ["visible", "hidden"]
        times = [0, frame_times[1] / DURATION]
    elif k == K - 1:
        values = ["hidden", "visible"]
        times = [0, frame_times[k] / DURATION]
    else:
        values = ["hidden", "visible", "hidden"]
        times = [
            0,
            frame_times[k] / DURATION,
            frame_times[k + 1] / DURATION
        ]

    animation = ET.SubElement(element, f"{{{SVG_NS}}}animate")
    animation.set("attributeName", "visibility")
    animation.set("values", ";".join(values))
    animation.set("keyTimes", ";".join(f"{t:.9f}" for t in times))
    animation.set("calcMode", "discrete")
    animation.set("dur", DUR)
    animation.set("begin", "0s")
    animation.set("repeatCount", REPEAT)

    if FREEZE:
        animation.set("fill", "freeze")


tree = ET.parse(svg_files[0])
root = tree.getroot()

for child in list(root):
    if child.tag.endswith("metadata"):
        root.remove(child)

groups = {}

for gid in ("bubble", "magdot"):
    group = find_group(root, gid)

    if group is None:
        sys.exit(f"ERROR: SVG group {gid!r} not found.")

    for child in list(group):
        group.remove(child)

    groups[gid] = group


# Check whether all bubble paths can be morphed

if args.morph:
    vertices = [path.count("L") for path in bubble_paths if path]

    if len(set(vertices)) != 1:
        print(f"WARNING: morph disabled; vertex counts differ: {sorted(set(vertices))}")
        args.morph = False
    else:
        print(f"Bubble morph: {vertices[0]} vertices per frame.")


# Bubble outline

if args.morph:
    attrs = dict(extracted[-1]["bubble"])
    attrs["d"] = bubble_paths[-1]

    bubble = ET.SubElement(
        groups["bubble"],
        f"{{{SVG_NS}}}path",
        attrib=attrs
    )

    # Static fallback is the final frame; animation begins with frame zero.
    values = ";".join([bubble_paths[0]] + bubble_paths[1:] + [bubble_paths[-1]])
    times = [0.0] + [t / DURATION for t in frame_times[1:]] + [1.0]

    animation = ET.SubElement(bubble, f"{{{SVG_NS}}}animate")
    animation.set("attributeName", "d")
    animation.set("values", values)
    animation.set("keyTimes", ";".join(f"{t:.9f}" for t in times))
    animation.set("calcMode", "linear")
    animation.set("dur", DUR)
    animation.set("begin", "0s")
    animation.set("repeatCount", REPEAT)

    if FREEZE:
        animation.set("fill", "freeze")

    stroke = ET.SubElement(bubble, f"{{{SVG_NS}}}animate")
    stroke.set("attributeName", "stroke")
    stroke.set("values", "red;blue")
    stroke.set("keyTimes", f"0;{ALPHA:.6f}")
    stroke.set("calcMode", "discrete")
    stroke.set("dur", DUR)
    stroke.set("begin", "0s")
    stroke.set("repeatCount", REPEAT)

    if FREEZE:
        stroke.set("fill", "freeze")

else:
    for k, frame in enumerate(extracted):
        attrs = frame["bubble"]

        if attrs is None:
            sys.exit(f"ERROR: no bubble path in frame {k}.")

        element = ET.SubElement(
            groups["bubble"],
            f"{{{SVG_NS}}}path",
            attrib=attrs
        )

        element.set("visibility", "visible" if k == K - 1 else "hidden")
        add_visibility(element, k)


# Animated marker: one path translated between frame positions

attrs = dict(extracted[-1]["magdot"])
attrs.setdefault("fill", "#0000ff")

dot_element = ET.SubElement(
    groups["magdot"],
    f"{{{SVG_NS}}}path",
    attrib=attrs
)

cx_last, cy_last = dot_centers[-1]

dot_values = [
    f"{x - cx_last:.{PREC}f},{y - cy_last:.{PREC}f}"
    for x, y in dot_centers
]

dot_values = [dot_values[0]] + dot_values[1:] + ["0,0"]
dot_times = [0.0] + [t / DURATION for t in frame_times[1:]] + [1.0]

translation = ET.SubElement(
    dot_element,
    f"{{{SVG_NS}}}animateTransform"
)

translation.set("attributeName", "transform")
translation.set("type", "translate")
translation.set("values", ";".join(dot_values))
translation.set("keyTimes", ";".join(f"{t:.6f}" for t in dot_times))
translation.set("calcMode", "linear")
translation.set("dur", DUR)
translation.set("begin", "0s")
translation.set("repeatCount", REPEAT)

if FREEZE:
    translation.set("fill", "freeze")


fill = ET.SubElement(dot_element, f"{{{SVG_NS}}}animate")
fill.set("attributeName", "fill")
fill.set("values", "red;blue")
fill.set("keyTimes", f"0;{ALPHA:.6f}")
fill.set("calcMode", "discrete")
fill.set("dur", DUR)
fill.set("begin", "0s")
fill.set("repeatCount", REPEAT)

if FREEZE:
    fill.set("fill", "freeze")


# Save
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

tree.write(OUT, encoding="utf-8", xml_declaration=True)

size_kb = os.path.getsize(OUT) / 1024
mode = "morph" if args.morph else "flipbook"

print(
    f"Wrote {OUT} ({size_kb:.0f} kB, {K} frames, "
    f"{mode}, {DURATION:.2f} s)"
)

plt.close(fig)
shutil.rmtree(WORK_SVG, ignore_errors=True)

if args.work is None:
    shutil.rmtree(WORK, ignore_errors=True)
