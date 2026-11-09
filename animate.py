#!/usr/bin/env python3
"""animatePort.py — animated SVG of bubble evolution, built to play in Safari.

Left panel   φ₀²/π² vs r₀/λ — plot.py's axA[1] verbatim (grey arcs,
             decile highlights, blue reference, arrowheads)
Right panel  interface profile — plot_profiles style

Run with no arguments from the repo directory:

    python animateU.py              ->  ./port7.svg

Optional:  python animateU.py [repo_dir] [work_dir] [out.svg]
                              [--fill-rate 1] [--single-shot] [--transparent]
                              [--morph]

How the animation works (and why it is portable)
------------------------------------------------
Frames are sampled at uniform Δ(volume) so the animation fills at a constant
volume rate.  Duplicate row indices (which would produce frame-to-frame
stalls) are nudged, with the last index clamped so the schedule never
overruns the source array; a warning is printed with the count.

* Trails — one path per phase, progressive reveal via `stroke-dashoffset`.
  The dash VALUE at each keyTime is the arc-length fraction up to that frame,
  so the tip stays exactly on the dot even though keyTimes are uniform in
  volume, not in arc.  Well-supported everywhere SMIL is.
* Dot — one path (frame K-1) translated by `animateTransform`; discrete
  `fill` tween at the pinning→spreading transition.
* Bubble outline — discrete flipbook by default (Safari-safe).  With --morph,
  a linear `d` animation is used instead; auto-falls back to flipbook if the
  frame vertex counts differ.  WebKit SMIL `d` support is the open question —
  test on the target Safari before committing to --morph.

Timing: keyTimes follow the dimensionless volume increase at a constant
fill rate shared by pinning and spreading.  Every animate element holds its
last value until loop end, then snaps back.

Fallback: renderers that ignore SMIL (Quick Look, Keynote, some Markdown
viewers) see the final frame with fully-revealed trails.
"""
import os, sys, glob, subprocess, tempfile, shutil, re, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore", message=".*tight_layout.*")

plt.rcParams["svg.fonttype"] = "path"      # text -> outlines, no font deps
plt.rcParams["path.simplify"] = False
plt.rcParams["svg.hashsalt"] = "animsvg"   # deterministic ids

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# --------------- CLI (everything optional) ---------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("repo", nargs="?", default=".")
ap.add_argument("work", nargs="?", default=None)
ap.add_argument("out",  nargs="?", default="port7.svg")
ap.add_argument("--fill-rate", type=float, default=1.0,
                help="dimensionless fill rate d(V/lambda^3)/dt [lambda^3/s]")
ap.add_argument("--single-shot", action="store_true")
ap.add_argument("--transparent", action="store_true")
ap.add_argument("--morph", action="store_true",
                help="animate bubble outline `d` (test on target Safari)")
args = ap.parse_args()

REPO, OUT = args.repo, args.out
FILL_RATE = args.fill_rate
if FILL_RATE <= 0:
    ap.error("--fill-rate must be positive")
REPEAT = "1" if args.single_shot else "indefinite"
TRANSPARENT = args.transparent
WORK = args.work or tempfile.mkdtemp(prefix="animwork_")
os.makedirs(WORK, exist_ok=True)

SIM = os.path.join(REPO, "simData") + "/"
RUN = os.path.join(REPO, "run")

RC, PHI_W, CAV = 0.5, 0.5, 0.3
KP, KS, M     = 22, 118, 48
PRIN, SPRD    = "loop_rad00005.txt", "loop_ang00025.txt"
R_GUIDE, R_BAND_END = 3.219, 3.832
BLUE_REF = "BinAngMaxVol.txt"
PREC = 2                       # decimals kept in path data

# --------------- Load main files ---------------
pr = np.loadtxt(SIM + PRIN)
assert abs(pr[0, 0] - RC) < 1e-3, "pinned file is not r0 = rc"
phi_p = 1 - pr[:, 2] / np.pi
jp = int(np.argmax(phi_p <= PHI_W + 1e-12))

sp = np.loadtxt(SIM + SPRD)
iM = int(np.argmax(sp[:, 6]))
sp = sp[: iM + 1]
assert abs((1 - sp[-1, 2] / np.pi) - PHI_W) < 1e-3, "spreading file is not phi_w"
js = int(np.argmax(sp[:, 0] >= RC))

r0_p_all = pr[: jp + 1, 0] / pr[: jp + 1, 4]
ph_p_all = (1 - pr[: jp + 1, 2] / np.pi) ** 2
r0_s_all = sp[js:, 0] / sp[js:, 4]
ph_s_all = (1 - sp[js:, 2] / np.pi) ** 2

# --------------- Frame schedule — volume rate ---------------
print(f"pinning:   {jp + 1} rows available, requesting {KP}")
print(f"spreading: {len(sp) - js} rows available, requesting {KS}")
assert KP <= jp, \
    f"KP={KP} > jp={jp}: not enough distinct pinning rows — reduce KP or re-run sim with denser output"
assert KS <= len(sp) - js, \
    f"KS={KS} > len(sp)-js={len(sp) - js}: not enough distinct spreading rows"

def volume_schedule(vol, n):
    """Choose rows at approximately uniform ΔV, with strictly increasing V."""
    v = np.maximum.accumulate(np.asarray(vol, dtype=float))

    # Keep only the first row for each distinct cumulative volume.  Nudging
    # indices in the original array can otherwise select different rows on a
    # volume plateau, giving them identical physical timestamps.
    unique_v, first_rows = np.unique(v, return_index=True)
    if len(unique_v) < n:
        raise ValueError(
            f"cannot schedule {n} frames: only {len(unique_v)} distinct "
            f"volume values are available")

    targets = np.linspace(unique_v[0], unique_v[-1], n)
    pos = np.searchsorted(unique_v, targets)
    pos = np.clip(pos, 0, len(unique_v) - 1)

    # If a large volume jump maps adjacent targets to the same support point,
    # move forward through distinct-volume support points, not raw rows.
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
                "reduce KP/KS or use denser simulation output")
        print(f"  nudge: resolved {n_dup} duplicate target(s) using distinct volumes")

    idx = first_rows[pos]
    if np.any(np.diff(v[idx]) <= 0):
        raise AssertionError("internal error: selected volumes are not increasing")
    return idx

# Pinning: skip row 0 (degenerate initial state) — matches the original
# `linspace(0, 1, KP+1)[1:]` behaviour: KP frames from KP+1 uniform-ΔV samples.
rows_p = volume_schedule(pr[: jp + 1, 6], KP + 1)[1:]

# Spreading: full range, starting at js.
rows_s = volume_schedule(sp[js:, 6], KS) + js

if len(set(rows_p)) < KP:
    print(f"WARNING: pinning schedule has {KP - len(set(rows_p))} duplicate rows")
if len(set(rows_s)) < KS:
    print(f"WARNING: spreading schedule has {KS - len(set(rows_s))} duplicate rows")

# --------------- Frame profiles ---------------
def half_profile(row):
    lam_ = row[4]
    pf = os.path.join(WORK, "prof.txt")
    subprocess.run([RUN, "1", f"{float(row[5]):.12g}", pf],
                   check=True, capture_output=True)
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
    xs = np.concatenate([x[: kk + 1], [xf]])
    zs = np.concatenate([z[: kk + 1], [zf]])
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
            V   = tab[i, 6] / tab[i, 4] ** 3,
            r0  = tab[i, 0] / tab[i, 4],
            phi2= (1 - tab[i, 2] / np.pi) ** 2,
        ))
K = len(frames)

# Physical animation clock: both phases use the same d(V/lambda^3)/dt.
# Use monotone volume histories to avoid small numerical reversals.
vpin = np.maximum.accumulate(pr[:jp + 1, 6] / pr[:jp + 1, 4] ** 3)
vspr = np.maximum.accumulate(sp[js:, 6] / sp[js:, 4] ** 3)
pin_duration = (vpin[-1] - vpin[0]) / FILL_RATE
spr_duration = (vspr[-1] - vspr[0]) / FILL_RATE
TRANSITION_EPS = 1e-6  # separates the two states at the instantaneous phase change
DURATION = pin_duration + spr_duration + TRANSITION_EPS

frame_times = []
for fr in frames:
    if fr["ph"] == "p":
        frame_times.append((vpin[fr["row"]] - vpin[0]) / FILL_RATE)
    else:
        frame_times.append(pin_duration + TRANSITION_EPS +
                           (vspr[fr["row"] - js] - vspr[0]) / FILL_RATE)
frame_times = np.asarray(frame_times)
if np.any(np.diff(frame_times) <= 0):
    raise ValueError("Frame times are not strictly increasing; check simulation volumes")
FRAME_KTS = [f"{t / DURATION:.9f}" for t in frame_times]
print(f"frames: {K}   duration: {DURATION:.2f}s")
print(f"fill rate: {FILL_RATE:g} lambda^3/s "
      f"(pinning {pin_duration:.2f}s, spreading {spr_duration:.2f}s)")

# --------------- Arc-length fractions along each trail ---------------
# keyTimes are uniform in VOLUME, so the dash VALUE must be an arc fraction
# (not a time fraction) to keep the dash tip on the dot.
def arc_frac(x, y):
    seg = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return arc / arc[-1] if arc[-1] > 0 else arc

pin_arc = arc_frac(r0_p_all, ph_p_all)
spr_arc = arc_frac(r0_s_all, ph_s_all)

# --------------- Figure ---------------
fig = plt.figure(figsize=(14, 6))
gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.22)
ax_phi = fig.add_subplot(gs[0])
ax_b   = fig.add_subplot(gs[1])

# ---- RIGHT: bubble profile (plot_profiles style) ----
xmax_p = max(f["x"].max() for f in frames)
ymax_p = max(f["y"].max() for f in frames)
xlim_b = xmax_p + 0.6
right_ylim = (-CAV - 0.3, ymax_p + 0.4)
right_xlim = (-xlim_b, xlim_b)
right_y_range = right_ylim[1] - right_ylim[0]
right_x_range = right_xlim[1] - right_xlim[0]
common_aspect = right_y_range / right_x_range

for a, b, c, d_ in [(-xlim_b, -RC, 0, 0), (RC, xlim_b, 0, 0),
                    (-RC, -RC, -CAV, 0), (RC, RC, -CAV, 0),
                    (-RC, RC, -CAV, -CAV)]:
    ax_b.plot([a, b], [c, d_], 'k-', lw=1.5, zorder=4, clip_on=False)
ax_b.set_xlim(right_xlim)
ax_b.set_ylim(right_ylim)
ax_b.set_aspect('equal', adjustable='box')
ax_b.set_xlabel(r'$r/\lambda$', fontsize=13)
ax_b.set_ylabel(r'$z/\lambda$', rotation=0, labelpad=16, fontsize=13)
ax_b.tick_params(which='both', direction='in', top=True, right=True)

bub_line, = ax_b.plot([], [], lw=2.5, zorder=5, clip_on=False)
bub_line.set_gid("bubble")

# ---- LEFT: φ₀²/π² vs r₀/λ, axA[1] style, aspect locked to right ----
ax_phi.set_xlim([0, 4])
ax_phi.set_ylim([0, 1.05])
ax_phi.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax_phi.set_xlabel(r'$r_0/\lambda$', fontsize=13)
ax_phi.set_ylabel(r'$\dfrac{\phi_0^2}{\pi^2}$', rotation=0,
                  labelpad=10, size=18)
ax_phi.tick_params(which='both', direction='in', top=True, right=True)
ax_phi.set_box_aspect(common_aspect)
ax_phi.axvspan(R_GUIDE, R_BAND_END, color='whitesmoke', zorder=0)

xx = np.linspace(0, 4, 500)
ax_phi.plot(xx, xx / 3.5, c='k', lw=0.15, zorder=1)
line, = ax_phi.plot(xx, xx / 3.219, c='k', lw=0.15, zorder=1)
line.set_dashes([50, 30])

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
    ax_phi.plot([x0, xe], [y0, ye], c='grey', lw=0.1, zorder=1)
    if round(angl * 100) % 20 != 10:
        continue
    n_decile += 1
    ax_phi.plot([x0, xe], [y0, ye], c='grey', lw=0.7, zorder=2)
    if round(angl * 100) == 90:
        ax_phi.plot([1.5, xe], [y0, ye], c='blue', zorder=2)
    for hei in range(5):
        target_h = (hei + 1) * dl[indVol, 1] / 5
        heiInd = int(np.argmin(np.abs(target_h - dl[:indVol + 1, 1])))
        heiPrev = max(0, heiInd - 1)
        col = 'grey' if hei < 4 else 'b'
        ax_phi.annotate('',
            xy=(dl[heiInd, 0], (1 - dl[heiInd, 2] / np.pi) ** 2),
            xytext=(dl[heiPrev, 0], (1 - dl[heiPrev, 2] / np.pi) ** 2),
            arrowprops=dict(arrowstyle='->', color=col,
                            lw=0.7, mutation_scale=8),
            zorder=3)
print(f"plotted {n_decile} decile arcs with arrowheads")

try:
    with open(SIM + BLUE_REF, encoding='utf-8') as f:
        df_bin = np.loadtxt(f)
    ax_phi.plot(df_bin[:, 0], (1 - df_bin[:, 2] / np.pi) ** 2,
                c='b', lw=0.9, zorder=3)
except FileNotFoundError:
    print(f"WARNING: {BLUE_REF} not found")

pin_trail, = ax_phi.plot([], [], 'r-', lw=2.0, zorder=4)
pin_trail.set_gid("pintrail")
spr_trail, = ax_phi.plot([], [], 'b-', lw=2.0, zorder=4)
spr_trail.set_gid("sprtrail")

left_x_range = 4.0
left_y_range = 1.05
rx_data = 0.045
ry_data = rx_data * left_y_range / (left_x_range * common_aspect)
dot = Ellipse((RC, 0.0), width=2 * rx_data, height=2 * ry_data,
              facecolor='r', edgecolor='black', linewidth=1.0,
              zorder=5, clip_on=False)
ax_phi.add_patch(dot)
dot.set_gid("magdot")

fig.tight_layout()

def update(k):
    fr = frames[k]
    X = np.concatenate([-fr["x"][::-1], fr["x"]])
    Y = np.concatenate([ fr["y"][::-1], fr["y"]])
    bub_line.set_color('r' if fr["ph"] == "p" else 'b')
    bub_line.set_data(X, Y)
    if fr["ph"] == "p":
        i = min(fr["row"], jp)
        a = min(frames[k - 1]["row"], jp) if k > 0 else 0
        pin_trail.set_data(r0_p_all[a:i + 1], ph_p_all[a:i + 1])
        spr_trail.set_data([], [])
    else:
        j = max(0, fr["row"] - js)
        prev = frames[k - 1]
        a = max(0, prev["row"] - js) if prev["ph"] == "s" else 0
        pin_trail.set_data([], [])
        spr_trail.set_data(r0_s_all[a:j + 1], ph_s_all[a:j + 1])
    dot.center = (fr["r0"], fr["phi2"])
    dot.set_facecolor('r' if fr["ph"] == "p" else 'b')

# --------------- Render frames ---------------
WORK_SVG = tempfile.mkdtemp(prefix="animsg_")
print(f"rendering {K} frame SVGs into {WORK_SVG}")
update(0)
fig.canvas.draw()
tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())

svg_files = []
for k in range(K):
    update(k)
    p = os.path.join(WORK_SVG, f"f{k:04d}.svg")
    fig.savefig(p, format='svg', bbox_inches=tight_bbox,
                transparent=TRANSPARENT)
    svg_files.append(p)
    if (k + 1) % 20 == 0:
        print(f"  {k + 1}/{K}")

print("rendering full trails for dash animation …")
update(0)
pin_trail.set_data(r0_p_all, ph_p_all)
spr_trail.set_data(r0_s_all, ph_s_all)
FULL_TRAILS = os.path.join(WORK_SVG, "full_trails.svg")
fig.savefig(FULL_TRAILS, format='svg', bbox_inches=tight_bbox,
            transparent=TRANSPARENT)

# --------------- Extract ---------------
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
    nums = [float(x) for x in _NUM_RE.findall(d_string)]
    if len(nums) < 2:
        return (0.0, 0.0)
    xs, ys = nums[0::2], nums[1::2]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

def polyline_length(d_string):
    """Sum of segment lengths for an M/L-only path (Line2D output)."""
    nums = [float(x) for x in _NUM_RE.findall(d_string)]
    if len(nums) < 4:
        return 0.0
    pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    return sum(((pts[i][0] - pts[i - 1][0]) ** 2
              + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
               for i in range(1, len(pts)))

print("extracting frame paths …")
ex = []
for p in svg_files:
    root_k = ET.parse(p).getroot()
    ex.append({g: group_path(root_k, g) for g in ("bubble", "magdot")})

print("extracting full trail paths …")
full_root = ET.parse(FULL_TRAILS).getroot()
pin_full = group_path(full_root, "pintrail")
spr_full = group_path(full_root, "sprtrail")
pin_L = polyline_length(pin_full["d"]) if pin_full else 0.0
spr_L = polyline_length(spr_full["d"]) if spr_full else 0.0
print(f"  trail lengths: pin={pin_L:.2f}, spr={spr_L:.2f}")

dot_centers = [path_bbox_center(ex[k]["magdot"]["d"]) if ex[k]["magdot"] else (0.0, 0.0)
               for k in range(K)]
bubble_ds   = [ex[k]["bubble"]["d"] if ex[k]["bubble"] else None for k in range(K)]

# --------------- Build the animation ---------------
DUR = f"{DURATION:.6f}s"
FREEZE = (REPEAT == "1")
ALPHA = pin_duration / DURATION    # pinning-phase end (fraction of loop)

def add_visibility(el, k):
    if k == 0:
        vals, kts = ["visible", "hidden"], [0, frame_times[1] / DURATION]
    elif k == K - 1:
        vals, kts = ["hidden", "visible"], [0, frame_times[k] / DURATION]
    else:
        vals, kts = ["hidden", "visible", "hidden"], [0, frame_times[k] / DURATION, frame_times[k + 1] / DURATION]
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
for gid in ("bubble", "magdot", "pintrail", "sprtrail"):
    g = find_group(root, gid)
    if g is None:
        sys.exit(f"ERROR: group '{gid}' not found in the SVG")
    for ch in list(g):
        g.remove(ch)
    groups[gid] = g

# ---- verify morph viability before committing ----
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
    attrs["d"] = bubble_ds[-1]                     # static fallback = last frame
    bub_el = ET.SubElement(groups["bubble"], f"{{{SVG_NS}}}path", attrib=attrs)
    # SMIL keyTimes must start at 0 and end at 1.  The first sampled
    # pinning frame is after t=0 (row 0 is deliberately skipped), so hold
    # that first sampled shape from the start of the animation.
    d_vals = ";".join([bubble_ds[0]] + bubble_ds[1:] + [bubble_ds[-1]])
    kts    = [0.0] + [t / DURATION for t in frame_times[1:]] + [1.0]
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
        el = ET.SubElement(groups["bubble"], f"{{{SVG_NS}}}path", attrib=at)
        el.set("visibility", "visible" if (k == K - 1) else "hidden")
        add_visibility(el, k)

# --- Dot (animateTransform on a single path) ---
attrs = dict(ex[-1]["magdot"])
attrs.setdefault("fill", "#0000ff")
dot_el = ET.SubElement(groups["magdot"], f"{{{SVG_NS}}}path", attrib=attrs)

cx_last, cy_last = dot_centers[-1]
dot_vals = [f"{(dot_centers[k][0] - cx_last):.{PREC}f},"
            f"{(dot_centers[k][1] - cy_last):.{PREC}f}" for k in range(K)]
# As above, anchor the first sampled marker position at keyTime 0.
dot_vals = [dot_vals[0]] + dot_vals[1:] + ["0,0"]
dot_kts = [0.0] + [t / DURATION for t in frame_times[1:]] + [1.0]

at_el = ET.SubElement(dot_el, f"{{{SVG_NS}}}animateTransform")
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

fill_el = ET.SubElement(dot_el, f"{{{SVG_NS}}}animate")
fill_el.set("attributeName", "fill")
fill_el.set("values", "red;blue")
fill_el.set("keyTimes", f"0;{ALPHA:.6f}")
fill_el.set("calcMode", "discrete")
fill_el.set("dur", DUR)
fill_el.set("begin", "0s")
fill_el.set("repeatCount", REPEAT)
if FREEZE:
    fill_el.set("fill", "freeze")

# --- Trails (dash-reveal, tip synced to dot) ---
if pin_full is not None and pin_L > 0.01:
    pin_ks   = [k for k in range(K) if frames[k]["ph"] == "p"]
    pin_vals = [f"{pin_L * (1 - pin_arc[frames[k]['row']]):.{PREC}f}"
                for k in pin_ks]
    # Pinning's first sampled frame is after t=0; hold its initial trail
    # state from t=0 so the SMIL keyTimes list starts at zero.
    pin_kts  = ["0"] + [FRAME_KTS[k] for k in pin_ks[1:]] + ["1"]
    pin_vals = [pin_vals[0]] + pin_vals[1:] + ["0"]
    attrs = dict(pin_full)
    attrs["stroke-dasharray"]  = f"{pin_L:.{PREC}f}"
    attrs["stroke-dashoffset"] = "0"
    el = ET.SubElement(groups["pintrail"], f"{{{SVG_NS}}}path", attrib=attrs)
    an = ET.SubElement(el, f"{{{SVG_NS}}}animate")
    an.set("attributeName", "stroke-dashoffset")
    an.set("values", ";".join(pin_vals))
    an.set("keyTimes", ";".join(pin_kts))
    an.set("calcMode", "linear")
    an.set("dur", DUR)
    an.set("begin", "0s")
    an.set("repeatCount", REPEAT)
    if FREEZE:
        an.set("fill", "freeze")

if spr_full is not None and spr_L > 0.01:
    spr_ks   = [k for k in range(K) if frames[k]["ph"] == "s"]
    # keyTime 0 with value spr_L keeps the spreading trail hidden during the
    # entire pinning phase; the first real spreading frame is at k = KP.
    spr_kts  = ["0"] + [FRAME_KTS[k] for k in spr_ks] + ["1"]
    spr_vals = [f"{spr_L:.{PREC}f}"] + [
        f"{spr_L * (1 - spr_arc[frames[k]['row'] - js]):.{PREC}f}"
        for k in spr_ks] + ["0"]
    attrs = dict(spr_full)
    attrs["stroke-dasharray"]  = f"{spr_L:.{PREC}f}"
    attrs["stroke-dashoffset"] = "0"
    el = ET.SubElement(groups["sprtrail"], f"{{{SVG_NS}}}path", attrib=attrs)
    an = ET.SubElement(el, f"{{{SVG_NS}}}animate")
    an.set("attributeName", "stroke-dashoffset")
    an.set("values", ";".join(spr_vals))
    an.set("keyTimes", ";".join(spr_kts))
    an.set("calcMode", "linear")
    an.set("dur", DUR)
    an.set("begin", "0s")
    an.set("repeatCount", REPEAT)
    if FREEZE:
        an.set("fill", "freeze")

tree.write(OUT, encoding="utf-8", xml_declaration=True)
kb = os.path.getsize(OUT) / 1024
mode_str = "morph" if args.morph else "flipbook"
print(f"wrote {OUT}  ({kb:.0f} kB,  {K} frames,  mode={mode_str},  "
      f"dur={DURATION:.2f}s, fill_rate={FILL_RATE:g} lambda^3/s)")

plt.close(fig)
shutil.rmtree(WORK_SVG, ignore_errors=True)
if args.work is None:
    shutil.rmtree(WORK, ignore_errors=True)
