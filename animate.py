#!/usr/bin/env python3
"""Create an animated SVG of bubble evolution from simulation data."""
import os, sys, glob, subprocess, tempfile, shutil, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation, Affine2D
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
import xml.etree.ElementTree as ET


plt.rcParams.update({"text.usetex": True, "font.size": 14})
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["path.simplify"] = False
plt.rcParams["svg.hashsalt"] = "animsvg"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# --------------- CLI (everything optional) ---------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", nargs="?", default=".")
ap.add_argument("work", nargs="?", default=None)
ap.add_argument("out", nargs="?", default=os.path.join("plots", "growing.svg"))
ap.add_argument("--fill-rate", type=float, default=1.0,
                help="dimensionless fill rate d(V/lambda^3)/dt [lambda^3/s]")
ap.add_argument("--phi2", type=float, default=0.7 ** 2,
                help="target dimensionless y-axis value phi^2/pi^2 (default: 0.49, phi/pi=0.7)")
ap.add_argument("--radius", type=float, default=1.0,
                help="target cavity radius r_c/lambda")
ap.add_argument("--single-shot", action="store_true")
ap.add_argument("--opaque", action="store_true",
                help="use an opaque white background (transparent by default)")
ap.add_argument("--morph", action=argparse.BooleanOptionalAction, default=True,
                help="animate bubble outline `d` (default; use --no-morph for flipbook)")
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

# Geometry and dataset selection.
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

# Select radius and phi²/π² independently.
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
        if (np.isfinite(phi2) and np.isfinite(phi_w)
                and 0 <= phi2 <= 1 and 0 <= phi_w <= 1):
            ang_candidates.append((path, data, inside_angle, phi2, phi_w))
    except (OSError, ValueError, IndexError):
        continue

if not rad_candidates:
    raise FileNotFoundError(f"No readable loop_rad*.txt datasets in {SIM}")
if not ang_candidates:
    raise FileNotFoundError(f"No readable loop_ang*.txt datasets in {SIM}")

rad_path, pr, RC, phi_pin = min(
    rad_candidates, key=lambda item: abs(item[2] - RC_TARGET))
ang_path, candidate_sp, CONTACT_ANGLE_DEG, PHI2_SELECTED, PHI_W = min(
    ang_candidates, key=lambda item: abs(item[3] - PHI2_TARGET))

pin_rows = np.flatnonzero(phi_pin <= PHI_W + 1e-12)
spread_rows = np.flatnonzero(candidate_sp[:, 0] >= RC)

if len(pin_rows) and len(spread_rows):
    jp = int(pin_rows[0])
    js = int(spread_rows[0])
    sp = candidate_sp
    HAS_SPREADING = True
    SPRD = os.path.basename(ang_path)
    print(f"spreading trajectory selected: {SPRD}; transition phi_w={PHI_W:g}")
else:
    jp = len(pr) - 1
    sp = np.empty((0, pr.shape[1]), dtype=float)
    js = 0
    HAS_SPREADING = False
    SPRD = None
    if not len(pin_rows):
        print("WARNING: pinned trajectory does not reach the chosen contact angle; "
              "animating pinning only.")
    else:
        print("WARNING: chosen spreading trajectory does not reach the selected "
              "cavity radius; animating pinning only.")

PRIN = os.path.basename(rad_path)
print(f"cavity radius r_c/lambda: requested {RC_TARGET:g}, selected {RC:g} "
      f"({PRIN}; difference {abs(RC - RC_TARGET):.4g})")
print(f"phi^2/pi^2 target: {PHI2_TARGET:.8g}; selected {PHI2_SELECTED:.8g} "
      f"(inside angle {CONTACT_ANGLE_DEG:.4g} deg)")
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
    """Choose rows at approximately uniform ΔV, with strictly increasing V."""
    v = np.maximum.accumulate(np.asarray(vol, dtype=float))
    unique_v, first_rows = np.unique(v, return_index=True)

    if len(unique_v) < n:
        raise ValueError(
            f"cannot schedule {n} frames: only {len(unique_v)} distinct "
            "volume values are available"
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
        print(f"  nudge: resolved {n_dup} duplicate target(s) using distinct volumes")

    idx = first_rows[pos]
    if np.any(np.diff(v[idx]) <= 0):
        raise AssertionError("internal error: selected volumes are not increasing")
    return idx


# Pinning: skip row 0 (degenerate initial state).
rows_p = volume_schedule(pr[:jp + 1, 6], KP + 1)[1:]

# Spreading: full range, starting at js.
rows_s = (volume_schedule(sp[js:, 6], KS) + js
          if HAS_SPREADING else np.array([], dtype=int))



# --------------- Frame profiles ---------------
def half_profile(row):
    lam_ = row[4]
    pf = os.path.join(WORK, "prof.txt")
    subprocess.run(
        [RUN, "1", f"{float(row[5]):.12g}", pf],
        check=True, capture_output=True
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
    t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    tt = np.linspace(0, t[-1], M)
    return np.interp(tt, t, xs), np.interp(tt, t, ys)


frames = []
for ph, rows, tab in (("p", rows_p, pr), ("s", rows_s, sp)):
    for i in rows:
        x, y = half_profile(tab[i])
        frames.append(dict(
            ph=ph, x=x, y=y, row=int(i),
            V=tab[i, 6] / tab[i, 4] ** 3,
            r0=tab[i, 0] / tab[i, 4],
            phi2=(1 - tab[i, 2] / np.pi) ** 2,
        ))
K = len(frames)

# Physical animation clock: both phases use the same d(V/lambda^3)/dt.
vpin = np.maximum.accumulate(pr[:jp + 1, 6] / pr[:jp + 1, 4] ** 3)
vspr = (np.maximum.accumulate(sp[js:, 6] / sp[js:, 4] ** 3)
        if HAS_SPREADING else np.array([], dtype=float))

pin_duration = (vpin[-1] - vpin[0]) / FILL_RATE
spr_duration = ((vspr[-1] - vspr[0]) / FILL_RATE
                if HAS_SPREADING else 0.0)
TRANSITION_EPS = 1e-6 if HAS_SPREADING else 0.0
ANIMATION_DURATION = pin_duration + spr_duration + TRANSITION_EPS
FINAL_HOLD = 0.0
DURATION = ANIMATION_DURATION + FINAL_HOLD

frame_times = []
for fr in frames:
    if fr["ph"] == "p":
        frame_times.append((vpin[fr["row"]] - vpin[0]) / FILL_RATE)
    elif HAS_SPREADING:
        frame_times.append(
            pin_duration + TRANSITION_EPS
            + (vspr[fr["row"] - js] - vspr[0]) / FILL_RATE
        )
frame_times = np.asarray(frame_times)

if np.any(np.diff(frame_times) <= 0):
    raise ValueError("Frame times are not strictly increasing; check simulation volumes")

print(f"frames: {K}   animation: {ANIMATION_DURATION:.2f}s + "
      f"{FINAL_HOLD:.1f}s final hold")
print(f"fill rate: {FILL_RATE:g} lambda^3/s "
      f"(pinning {pin_duration:.2f}s, spreading {spr_duration:.2f}s)")


# --------------- Figure ---------------
# Two-panel figure using the paper's 10 x 3 inch proportions.
fig, (ax_phi, ax_b) = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(wspace=0.0)

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

# Cavity and substrate: three straight segments, with axes linewidth.
# Lines are clipped by the plotting axes.
cavity_lw = 1
ax_b.plot([-xlim_b, -RC], [0, 0], color="k", lw=cavity_lw,
          zorder=4, clip_on=True)
ax_b.plot([RC, xlim_b], [0, 0], color="k", lw=cavity_lw,
          zorder=4, clip_on=True)
ax_b.plot([-RC, -RC, RC, RC], [0, -CAV, -CAV, 0], color="k",
          lw=cavity_lw, zorder=4, clip_on=True)

ax_b.set_xlim(right_xlim)
ax_b.set_ylim(right_ylim)
ax_b.set_aspect("equal", adjustable="box")
ax_b.set_xlabel(r"$r/\lambda$")
ax_b.set_ylabel(r"$\frac{ z }{\lambda}$", rotation=0, labelpad=5, size=22)
ax_b.tick_params(which="both", direction="in", top=True, right=True)

bub_line, = ax_b.plot([], [], lw=1.5, zorder=5, clip_on=False)
bub_line.set_gid("bubble")

# phi_0 marker: arc of radius 0.15 centred on the right foot (lw=2, black),
# plus a black label offset at (+0.12, +0.05) - matching plot.py.  The label
# is rendered as a TextPath / PathPatch so it becomes a real SVG <path>.
ARC_RADIUS = 0.15
N_ARC = 51
LABEL_DX, LABEL_DY = 0.12, 0.05

arc_line, = ax_b.plot([], [], c="k", lw=2, zorder=5, clip_on=False)
arc_line.set_gid("phi0arc")

_PHI0_TP = TextPath((0, 0), r"$\phi_0$", size=14, usetex=True)
phi0_label = PathPatch(_PHI0_TP, facecolor="k", edgecolor="none",
                       lw=0, zorder=4, clip_on=False)
ax_b.add_patch(phi0_label)
phi0_label.set_gid("phi0label")


# ---- LEFT: φ₀²/π² vs r₀/λ ----
ax_phi.set_xlim([0, 4])
ax_phi.set_ylim([0, 1.05])
ax_phi.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax_phi.set_xlabel(r"$r_0/\lambda$")
ax_phi.set_ylabel(r"$\frac{\phi_0^2}{\pi^2}$", rotation=0,
                  labelpad=10, size=22)
ax_phi.tick_params(which="both", direction="in", top=True, right=True)
ax_phi.axvspan(R_GUIDE, R_BAND_END, color="whitesmoke", zorder=0)

xx = np.linspace(0, 4, 500)
ax_phi.plot(xx, xx / 3.5, c="k", lw=0.1, zorder=1)

# Plot decile reference trajectories.
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
    ax_phi.plot([x0, xe], [y0, ye], c="grey", lw=0.1, zorder=2)

    if round(angl * 100) == 90:
        ax_phi.plot([1.5, xe], [y0, ye], c="blue", zorder=2)

    for hei in range(5):
        target_h = (hei + 1) * dl[indVol, 1] / 5
        heiInd = int(np.argmin(np.abs(target_h - dl[:indVol + 1, 1])))
        heiPrev = max(0, heiInd - 1)
        col = "grey" if hei < 4 else "b"
        ax_phi.annotate(
            "",
            xy=(dl[heiInd, 0], (1 - dl[heiInd, 2] / np.pi) ** 2),
            xytext=(dl[heiPrev, 0], (1 - dl[heiPrev, 2] / np.pi) ** 2),
            arrowprops=dict(arrowstyle="->", color=col),
            zorder=3
        )

print(f"plotted {n_decile} decile arcs with arrowheads")

try:
    with open(SIM + BLUE_REF, encoding="utf-8") as f:
        df_bin = np.loadtxt(f)
    ax_phi.plot(df_bin[:, 0], (1 - df_bin[:, 2] / np.pi) ** 2,
                c="b", zorder=3)
except FileNotFoundError:
    print(f"WARNING: {BLUE_REF} not found")


ax_phi.plot(r0_p_all, ph_p_all, color="grey", zorder=3)
if HAS_SPREADING:
    ax_phi.plot(r0_s_all, ph_s_all, color="grey", zorder=3)


# Red detachment boundary from BinRadMaxVol.txt.
detachment_file = os.path.join(SIM, "BinRadMaxVol.txt")
if os.path.isfile(detachment_file):
    det = np.loadtxt(detachment_file)
    if det.ndim == 2 and len(det) > 100:
        min_len = 100
        min_ang = np.array([
            np.min(1 - det[i:i + min_len, 2] / np.pi)
            for i in range(0, len(det[:, 2]) - min_len, min_len)
        ])
        rad = np.array([
            det[i + min_len // 2, 0]
            for i in range(0, len(det[:, 2]) - min_len, min_len)
        ])
        side = min_ang >= 1 - 1e-3

        ax_phi.plot(rad[~side], min_ang[~side] ** 2, c="r",
                    zorder=5, clip_on=False)
        ax_phi.plot(rad[side], min_ang[side] ** 2, c="r",
                    ls="dashed", zorder=5, clip_on=False)
else:
    print(f"WARNING: {detachment_file} not found; red detachment line omitted")



# Moving marker: an ax.annotate arrow in the same style as the arrowheads in
# plot.py (arrowstyle "->", default linewidth and mutation_scale).  It is always
# rendered pointing along +x in display space and centred on the data point;
# the per-frame heading (direction of increasing volume) is applied
# afterwards as an animated SVG rotation.
ARROW_LEN_PT = 6.0   # shaft length; the open head sits at its tip
arrow_tr = ax_phi.transData + ScaledTranslation(
    ARROW_LEN_PT / 2 / 72.0, 0, fig.dpi_scale_trans)
dot = ax_phi.annotate(
    "",
    xy=(RC, 0), xycoords=arrow_tr,                  # head: point + L/2
    xytext=(-ARROW_LEN_PT, 0), textcoords="offset points",  # tail: head - L
    annotation_clip=False,
    zorder=5,
    arrowprops=dict(
        arrowstyle="->", color="r",   # as plot.py: default lw and mutation_scale
        shrinkA=0, shrinkB=0,         # keep the head centred on the data point
    ),
)
dot.arrow_patch.set_gid("magdot")


# phi_0 arc: span the actual contact angle from sim data.  φ₀ = π√(phi2),
# since phi2 = (1 - tab[i,2]/π)² and plot.py uses φ₀/π = 1 - tab[i,2]/π.
def phi0_arc(fr):
    xAn, yAn = fr["x"][-1], fr["y"][-1]
    phi_max = np.pi * np.sqrt(fr["phi2"])
    phis = np.linspace(0.0, phi_max, N_ARC)
    return xAn + ARC_RADIUS * np.cos(phis), yAn + ARC_RADIUS * np.sin(phis)


# Force the aspect adjustment now so ax_b.get_position() reflects the
# final position; then compute the fixed points-to-data scale for the label.
fig.canvas.draw()
fig_w, fig_h = fig.get_size_inches()
ax_pos = ax_b.get_position()
_SX = (right_xlim[1] - right_xlim[0]) / (ax_pos.width * fig_w * 72)
_SY = (right_ylim[1] - right_ylim[0]) / (ax_pos.height * fig_h * 72)
_EXT_X0 = _PHI0_TP.get_extents().x0
_EXT_Y0 = _PHI0_TP.get_extents().y0


def _phi0_label_transform(foot_x, foot_y):
    """Position the label's bottom-left at (foot + LABEL_DX, foot + LABEL_DY),
    matching plot.py's ha='left', va='bottom' placement."""
    tx = foot_x + LABEL_DX - _EXT_X0 * _SX
    ty = foot_y + LABEL_DY - _EXT_Y0 * _SY
    return Affine2D().scale(_SX, _SY).translate(tx, ty) + ax_b.transData


# Axes positions are adjusted explicitly to retain the tight paper-style gap.
def update(k):
    fr = frames[k]
    X = np.concatenate([-fr["x"][::-1], fr["x"]])
    Y = np.concatenate([fr["y"][::-1], fr["y"]])
    bub_line.set_color("r" if fr["ph"] == "p" else "b")
    bub_line.set_data(X, Y)

    foot_x, foot_y = fr["x"][-1], fr["y"][-1]
    phi0_label.set_transform(_phi0_label_transform(foot_x, foot_y))

    Xarc, Yarc = phi0_arc(fr)
    arc_line.set_data(Xarc, Yarc)

    dot.xy = (fr["r0"], fr["phi2"])
    dot.arrow_patch.set_color("r" if fr["ph"] == "p" else "b")

# --------------- Render frames ---------------
WORK_SVG = tempfile.mkdtemp(prefix="animsg_")
print(f"rendering {K} frame SVGs into {WORK_SVG}")

update(0)
fig.canvas.draw()
tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())


# Heading of the marker per frame: direction of increasing volume along the
# (r0/lambda, phi0^2/pi^2) trajectory, measured in display space so the
# plot's axis scaling is respected.  Returned as SVG rotation angles (deg).
def frame_heading_svg_deg():
    thetas, prev = [], 0.0
    for fr in frames:
        tab = pr if fr["ph"] == "p" else sp
        lo, hi = (0, jp) if fr["ph"] == "p" else (js, len(sp) - 1)
        a, b = max(lo, fr["row"] - 3), min(hi, fr["row"] + 3)
        if tab[b, 6] / tab[b, 4] ** 3 < tab[a, 6] / tab[a, 4] ** 3:
            a, b = b, a  # make sure we point towards larger volume
        pa, pb = (ax_phi.transData.transform(
            (tab[i, 0] / tab[i, 4], (1 - tab[i, 2] / np.pi) ** 2))
            for i in (a, b))
        dx, dy = pb - pa
        prev = float(np.arctan2(dy, dx)) if np.hypot(dx, dy) > 0 else prev
        thetas.append(prev)
    # display y points up, SVG y points down -> negate; unwrap for smooth tween
    return -np.degrees(np.unwrap(thetas))


heading_deg = frame_heading_svg_deg()

svg_files = []
for k in range(K):
    update(k)
    p = os.path.join(WORK_SVG, f"f{k:04d}.svg")
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
_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _fmt(m):
    v = round(float(m.group(0)), PREC)
    t = f"{v:.{PREC}f}".rstrip("0").rstrip(".")
    return "0" if t in ("", "-0") else t


def compact(d_str):
    return " ".join(_NUM_RE.sub(_fmt, d_str).split())


def find_group(root, gid):
    for el in root.iter():
        if el.get("id") == gid:
            return el
    return None


def group_path(root, gid, merge=False):
    """Attributes of the first path in group `gid`.  With merge=True the `d`
    of every path in the group is joined into one multi-subpath string
    (an "->" arrow is drawn as separate shaft and head paths)."""
    g = find_group(root, gid)
    if g is None:
        return None

    out = None
    for el in g.iter():
        if el.tag.endswith("path") and el.get("d"):
            if out is None:
                out = dict(el.attrib)
                out.pop("clip-path", None)
                out["d"] = compact(out["d"])
                if not merge:
                    return out
            else:
                out["d"] += " " + compact(el.get("d"))

    return out


def path_bbox_center(d_string):
    """Bounding-box centre of a path with absolute x,y coordinates."""
    nums = [float(x) for x in _NUM_RE.findall(d_string)]
    if len(nums) < 2:
        return 0.0, 0.0

    xs, ys = nums[0::2], nums[1::2]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


print("extracting frame paths …")
ex = []
for p in svg_files:
    root_k = ET.parse(p).getroot()
    ex.append({g: group_path(root_k, g, merge=(g in ("magdot", "phi0label")))
               for g in ("bubble", "magdot", "phi0arc", "phi0label")})

dot_centers = [
    path_bbox_center(ex[k]["magdot"]["d"])
    if ex[k]["magdot"] else (0.0, 0.0)
    for k in range(K)
]
bubble_ds = [
    ex[k]["bubble"]["d"] if ex[k]["bubble"] else None
    for k in range(K)
]
arc_ds = [
    ex[k]["phi0arc"]["d"] if ex[k]["phi0arc"] else None
    for k in range(K)
]
label_centers = [
    path_bbox_center(ex[k]["phi0label"]["d"])
    if ex[k]["phi0label"] else (0.0, 0.0)
    for k in range(K)
]


# --------------- Build the animation ---------------
DUR = f"{DURATION:.6f}s"
ANIM_END = ANIMATION_DURATION / DURATION
FREEZE = (REPEAT == "1")
ALPHA = pin_duration / DURATION


def add_visibility(el, k):
    if k == 0:
        vals, kts = ["visible", "hidden"], [0, frame_times[1] / DURATION]
    elif k == K - 1:
        vals, kts = ["hidden", "visible"], [0, frame_times[k] / DURATION]
    else:
        vals, kts = [
            "hidden", "visible", "hidden"
        ], [
            0, frame_times[k] / DURATION, frame_times[k + 1] / DURATION
        ]

    an = ET.SubElement(el, f"{{{SVG_NS}}}animate")
    an.set("attributeName", "visibility")
    an.set("values", ";".join(vals))
    an.set("keyTimes", ";".join(f"{t:.9f}" for t in kts))
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
for gid in ("bubble", "magdot", "phi0arc", "phi0label"):
    g = find_group(root, gid)
    if g is None:
        sys.exit(f"ERROR: group '{gid}' not found in the SVG")

    for ch in list(g):
        g.remove(ch)

    groups[gid] = g


# ---- Verify morph viability before committing ----
if args.morph:
    nL = [d.count("L") for d in bubble_ds if d]

    if len(set(nL)) != 1:
        print(f"WARNING: --morph disabled (vertex counts differ: {sorted(set(nL))})")
        args.morph = False
    else:
        print(f"  bubble morph: {nL[0]} vertices / frame")


# --- Bubble outline ---
if args.morph:
    attrs = dict(ex[-1]["bubble"])
    attrs["d"] = bubble_ds[-1]
    bub_el = ET.SubElement(
        groups["bubble"], f"{{{SVG_NS}}}path", attrib=attrs
    )

    # Hold the first sampled pinning shape from t=0.
    d_vals = ";".join(
        [bubble_ds[0]] + bubble_ds[1:] + [bubble_ds[-1], bubble_ds[-1]]
    )
    kts = (
        [0.0]
        + [t / DURATION for t in frame_times[1:]]
        + [ANIM_END, 1.0]
    )

    an = ET.SubElement(bub_el, f"{{{SVG_NS}}}animate")
    an.set("attributeName", "d")
    an.set("values", d_vals)
    an.set("keyTimes", ";".join(f"{t:.9f}" for t in kts))
    an.set("calcMode", "linear")
    an.set("dur", DUR)
    an.set("begin", "0s")
    an.set("repeatCount", REPEAT)

    if FREEZE:
        an.set("fill", "freeze")

    an2 = ET.SubElement(bub_el, f"{{{SVG_NS}}}animate")
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
            sys.exit(f"ERROR: frame {k} has no bubble path")

        el = ET.SubElement(
            groups["bubble"], f"{{{SVG_NS}}}path", attrib=at
        )
        el.set("visibility", "visible" if k == K - 1 else "hidden")
        add_visibility(el, k)


# --- phi0 arc: morph with the bubble when vertex counts match ---
if all(arc_ds):
    if args.morph and len(set(d.count("L") for d in arc_ds)) == 1:
        attrs_arc = dict(ex[-1]["phi0arc"])
        attrs_arc["d"] = arc_ds[-1]
        arc_el = ET.SubElement(
            groups["phi0arc"], f"{{{SVG_NS}}}path", attrib=attrs_arc
        )

        d_vals_arc = ";".join(
            [arc_ds[0]] + arc_ds[1:] + [arc_ds[-1], arc_ds[-1]]
        )
        an = ET.SubElement(arc_el, f"{{{SVG_NS}}}animate")
        an.set("attributeName", "d")
        an.set("values", d_vals_arc)
        an.set("keyTimes", ";".join(f"{t:.9f}" for t in kts))
        an.set("calcMode", "linear")
        an.set("dur", DUR)
        an.set("begin", "0s")
        an.set("repeatCount", REPEAT)

        if FREEZE:
            an.set("fill", "freeze")
    else:
        for k in range(K):
            if ex[k]["phi0arc"] is None:
                continue
            el = ET.SubElement(
                groups["phi0arc"], f"{{{SVG_NS}}}path",
                attrib=dict(ex[k]["phi0arc"])
            )
            el.set("visibility", "visible" if k == K - 1 else "hidden")
            add_visibility(el, k)


# --- phi0 label: translate animation on a fixed shape ---
if all(ex[k]["phi0label"] for k in range(K)):
    if args.morph:
        attrs_lbl = dict(ex[-1]["phi0label"])
        lbl_move = ET.SubElement(groups["phi0label"], f"{{{SVG_NS}}}g")
        ET.SubElement(lbl_move, f"{{{SVG_NS}}}path", attrib=attrs_lbl)

        lx_last, ly_last = label_centers[-1]
        lbl_vals = [
            f"{(label_centers[k][0] - lx_last):.{PREC}f},"
            f"{(label_centers[k][1] - ly_last):.{PREC}f}"
            for k in range(K)
        ]
        lbl_vals = lbl_vals[:1] + lbl_vals[1:] + ["0,0", "0,0"]
        lbl_kts = (
            [0.0]
            + [t / DURATION for t in frame_times[1:]]
            + [ANIM_END, 1.0]
        )

        at = ET.SubElement(lbl_move, f"{{{SVG_NS}}}animateTransform")
        at.set("attributeName", "transform")
        at.set("type", "translate")
        at.set("values", ";".join(lbl_vals))
        at.set("keyTimes", ";".join(f"{t:.6f}" for t in lbl_kts))
        at.set("calcMode", "linear")
        at.set("dur", DUR)
        at.set("begin", "0s")
        at.set("repeatCount", REPEAT)

        if FREEZE:
            at.set("fill", "freeze")
    else:
        for k in range(K):
            if ex[k]["phi0label"] is None:
                continue
            el = ET.SubElement(
                groups["phi0label"], f"{{{SVG_NS}}}path",
                attrib=dict(ex[k]["phi0label"])
            )
            el.set("visibility", "visible" if k == K - 1 else "hidden")
            add_visibility(el, k)


# --- Dot: animateTransform on a single path ---
attrs = dict(ex[-1]["magdot"])
# matplotlib puts stroke/fill in `style`, which would override an animated
# `stroke` attribute; move them to presentation attributes.  The "->" arrow
# is stroke-only, so the colour change animates `stroke`.
style = {}
for item in attrs.pop("style", "").split(";"):
    if ":" in item:
        key, val = item.split(":", 1)
        style[key.strip()] = val.strip()
style.pop("fill", None)
style.pop("stroke", None)
if style:
    attrs["style"] = ";".join(f"{k}:{v}" for k, v in style.items())
attrs["fill"] = "none"
attrs["stroke"] = "blue" if HAS_SPREADING else "red"

cx_last, cy_last = dot_centers[-1]

# <g translate-anim> <g rotate-anim, base = final heading> <path/> </g></g>
# Rotation (about the final-frame centre) is applied inside the translation,
# so the arrow turns about its own centre while it travels.  The base
# transform keeps the final frame correct in renderers that ignore SMIL.
dot_move = ET.SubElement(groups["magdot"], f"{{{SVG_NS}}}g")
dot_turn = ET.SubElement(dot_move, f"{{{SVG_NS}}}g")
dot_turn.set(
    "transform",
    f"rotate({heading_deg[-1]:.2f} {cx_last:.{PREC}f} {cy_last:.{PREC}f})")
dot_el = ET.SubElement(dot_turn, f"{{{SVG_NS}}}path", attrib=attrs)

dot_vals = [
    f"{(dot_centers[k][0] - cx_last):.{PREC}f},"
    f"{(dot_centers[k][1] - cy_last):.{PREC}f}"
    for k in range(K)
]

# Anchor the first sampled marker position at keyTime 0.
dot_vals = dot_vals[:1] + dot_vals[1:] + ["0,0", "0,0"]
dot_kts = (
    [0.0]
    + [t / DURATION for t in frame_times[1:]]
    + [ANIM_END, 1.0]
)

at_el = ET.SubElement(
    dot_move, f"{{{SVG_NS}}}animateTransform"
)
at_el.set("attributeName", "transform")
at_el.set("type", "translate")
at_el.set("values", ";".join(dot_vals))
at_el.set("keyTimes", ";".join(f"{t:.6f}" for t in dot_kts))
at_el.set("calcMode", "linear")
at_el.set("dur", DUR)
at_el.set("begin", "0s")
at_el.set("repeatCount", REPEAT)

if FREEZE:
    at_el.set("fill", "freeze")

rot_vals = [f"{a:.2f} {cx_last:.{PREC}f} {cy_last:.{PREC}f}"
            for a in heading_deg]
rot_vals = rot_vals + [rot_vals[-1], rot_vals[-1]]
rot_el = ET.SubElement(dot_turn, f"{{{SVG_NS}}}animateTransform")
rot_el.set("attributeName", "transform")
rot_el.set("type", "rotate")
rot_el.set("values", ";".join(rot_vals))
rot_el.set("keyTimes", ";".join(f"{t:.6f}" for t in dot_kts))
rot_el.set("calcMode", "linear")
rot_el.set("dur", DUR)
rot_el.set("begin", "0s")
rot_el.set("repeatCount", REPEAT)

if FREEZE:
    rot_el.set("fill", "freeze")

fill_el = ET.SubElement(dot_el, f"{{{SVG_NS}}}animate")
fill_el.set("attributeName", "stroke")
fill_el.set("values", "red;blue")
fill_el.set("keyTimes", f"0;{ALPHA:.6f}")
fill_el.set("calcMode", "discrete")
fill_el.set("dur", DUR)
fill_el.set("begin", "0s")
fill_el.set("repeatCount", REPEAT)

if FREEZE:
    fill_el.set("fill", "freeze")


os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
tree.write(OUT, encoding="utf-8", xml_declaration=True)

kb = os.path.getsize(OUT) / 1024
mode_str = "morph" if args.morph else "flipbook"

print(
    f"wrote {OUT}  ({kb:.0f} kB,  {K} frames,  mode={mode_str},  "
    f"dur={DURATION:.2f}s, fill_rate={FILL_RATE:g} lambda^3/s)"
)

plt.close(fig)
shutil.rmtree(WORK_SVG, ignore_errors=True)

if args.work is None:
    shutil.rmtree(WORK, ignore_errors=True)
