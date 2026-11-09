#!/usr/bin/env python3
"""make_animated_svg.py — animated SVG, plot.py-faithful left panel.

Left panel   φ₀²/π² vs r₀/λ — plot.py's axA[1] verbatim (grey arcs,
             decile highlights, magenta, blue reference, arrowheads)
Right panel  interface profile — plot_profiles style

Cross-browser note
------------------
The moving dot's position is injected as `<animate attributeName="d">`
on its <path> (a Circle patch renders as a path in matplotlib's SVG
backend, so this is the same SMIL mechanism as the bubble outline).
`animateTransform type="translate"` on `<path>` works in Blink/WebKit
for the common case, but attribute-d animation is what WebKit has
historically supported most reliably.  A transform fallback with
`additive="sum"` is used if `d` turns out to be constant (defensive).

Usage:
    python make_animated_svg.py <repo_dir> <work_dir> <out.svg>
                                [--fps 15] [--single-shot]
"""
import os, sys, glob, subprocess, tempfile, shutil, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# --------------- CLI ---------------
if len(sys.argv) < 4:
    sys.exit(__doc__)
REPO, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FPS, REPEAT = 15.0, "indefinite"
for i, a in enumerate(sys.argv):
    if a == "--fps":
        FPS = float(sys.argv[i + 1])
    elif a == "--single-shot":
        REPEAT = "1"

SIM = os.path.join(REPO, "simData") + "/"
RUN = os.path.join(REPO, "run")

RC, PHI_W, CAV = 1.5, 0.9, 0.3
KP, KS, M     = 22, 118, 48
PRIN, SPRD    = "loop_rad00015.txt", "loop_ang00005.txt"
R_GUIDE, R_BAND_END = 3.219, 3.832
BLUE_REF = "BinAngMaxVol.txt"

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

# --------------- Frame schedule ---------------
def monotone01(a):
    a = np.maximum.accumulate(a)
    return (a - a[0]) / (a[-1] - a[0])

hp = -pr[: jp + 1, 1] / pr[: jp + 1, 4]
rows_p = np.clip(np.searchsorted(monotone01(hp),
                                  np.linspace(0, 1, KP + 1)[1:]), 1, jp)
rows_p[-1] = jp

d = sp[js:]
lam = d[:, 4]
v13 = (d[:, 6] / lam ** 3) ** (1 / 3)
hh  = -d[:, 1] / lam
r0d = d[:, 0] / lam
ds  = np.sqrt(np.diff(v13) ** 2 + np.diff(hh) ** 2 + np.diff(r0d) ** 2)
s   = np.concatenate([[0.0], np.cumsum(ds)])
s  /= s[-1]
rows_s = np.clip(np.searchsorted(s, np.linspace(0, 1, KS)),
                 0, len(d) - 1) + js
rows_s[0], rows_s[-1] = js, len(sp) - 1

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
print(f"frames: {K}   duration: {K / FPS:.2f}s at {FPS} fps")

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
common_aspect = right_y_range / right_x_range     # height / width

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
ax_phi.set_box_aspect(common_aspect)              # ← match right's box shape
ax_phi.axvspan(R_GUIDE, R_BAND_END, color='whitesmoke', zorder=0)

xx = np.linspace(0, 4, 500)
ax_phi.plot(xx, xx / 3.5, c='k', lw=0.15, zorder=1)
line, = ax_phi.plot(xx, xx / 3.219, c='k', lw=0.15, zorder=1)
line.set_dashes([50, 30])

# All loop_ang*.txt trajectories — plot.py axA[1] verbatim
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
    #    ax_phi.plot([0.5, 0.5], [y0, 1], c='red', zorder=2)
        ax_phi.plot([1.5, xe], [y0, ye], c='blue',zorder=2)

    for hei in range(5):
        target_h = (hei + 1) * dl[indVol, 1] / 5
        heiInd = int(np.argmin(np.abs(target_h - dl[:indVol + 1, 1])))
        heiPrev = max(0, heiInd - 1)
        col = 'grey' if hei < 4 else 'b'
        ax_phi.annotate(
            '',
            xy=(dl[heiInd, 0], (1 - dl[heiInd, 2] / np.pi) ** 2),
            xytext=(dl[heiPrev, 0], (1 - dl[heiPrev, 2] / np.pi) ** 2),
            arrowprops=dict(arrowstyle='->', color=col,
                            lw=0.7, mutation_scale=8),
            zorder=3,
        )
print(f"plotted {n_decile} decile arcs with arrowheads")

# Blue reference curve (BinAngMaxVol.txt)
try:
    with open(SIM + BLUE_REF, encoding='utf-8') as f:
        df_bin = np.loadtxt(f)
    ax_phi.plot(df_bin[:, 0], (1 - df_bin[:, 2] / np.pi) ** 2,
                c='b', lw=0.9, zorder=3)
except FileNotFoundError:
    print(f"WARNING: {BLUE_REF} not found")


# Animated trails (paths whose d morphs per frame)
pin_trail, = ax_phi.plot([], [], 'r-', lw=2.0, zorder=4)
pin_trail.set_gid("pintrail")
spr_trail, = ax_phi.plot([], [], 'b-', lw=2.0, zorder=4)
spr_trail.set_gid("sprtrail")

# Dot: Ellipse in data units with radii chosen so it appears circular on the
# aspect-constrained left axes.  Red at init (frame 0 is pinned).
left_x_range = 4.0
left_y_range = 1.05
rx_data = 0.045
ry_data = rx_data * left_y_range / (left_x_range * common_aspect)
dot = Ellipse((RC, 0.0), width=2 * rx_data, height=2 * ry_data,
              facecolor='r', edgecolor='black', linewidth=1.0,
              zorder=5, clip_on=False)
ax_phi.add_patch(dot)
dot.set_gid("magdot")

#fig.suptitle(r'bubble evolution:  $r_c = 0.5\lambda$,  '
#             r'$\phi_w = 0.9\pi$', fontsize=14)
fig.tight_layout()

# --------------- Update ---------------
def update(k):
    fr = frames[k]

    X = np.concatenate([-fr["x"][::-1], fr["x"]])
    Y = np.concatenate([ fr["y"][::-1], fr["y"]])
    bub_line.set_color('r' if fr["ph"] == "p" else 'b')
    bub_line.set_data(X, Y)

    if fr["ph"] == "p":
        i = min(fr["row"], jp)
        pin_trail.set_data(r0_p_all[:i + 1], ph_p_all[:i + 1])
        spr_trail.set_data([], [])
    else:
        pin_trail.set_data(r0_p_all, ph_p_all)
        j = max(0, fr["row"] - js)
        spr_trail.set_data(r0_s_all[:j + 1], ph_s_all[:j + 1])

    dot.center = (fr["r0"], fr["phi2"])
    dot.set_facecolor('r' if fr["ph"] == "p" else 'b')

# --------------- Render each frame ---------------
WORK_SVG = tempfile.mkdtemp(prefix="animsg_")
print(f"rendering {K} frame SVGs into {WORK_SVG}")
update(0)
fig.canvas.draw()
tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())

svg_files = []
for k in range(K):
    update(k)
    p = os.path.join(WORK_SVG, f"f{k:04d}.svg")
    fig.savefig(p, format='svg', bbox_inches=tight_bbox, transparent=True)
    svg_files.append(p)
    if (k + 1) % 20 == 0:
        print(f"  {k + 1}/{K}")

# --------------- Extract ---------------
def _find(root, gid, tag):
    for el in root.iter():
        if el.get('id') == gid:
            if el.tag.endswith(tag):
                return el
            for ch in el.iter():
                if ch.tag.endswith(tag):
                    return ch
    return None

_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def path_centroid(d_str):
    if not d_str:
        return (0.0, 0.0)
    nums = [float(v) for v in _NUM_RE.findall(d_str)]
    if len(nums) < 4 or len(nums) % 2:
        return (0.0, 0.0)
    xs, ys = nums[0::2], nums[1::2]
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))

def extract(svg_path):
    root = ET.parse(svg_path).getroot()
    out = {}
    for gid in ("bubble", "pintrail", "sprtrail"):
        el = _find(root, gid, "path")
        out[gid] = (el.get('d') or "M 0,0") if el is not None else "M 0,0"
    el = _find(root, "magdot", "path")
    out["magdot_d"] = el.get('d', "") if el is not None else ""
    return out

print("extracting paths and dot positions …")
ex = [extract(p) for p in svg_files]
bubble_ds = [e["bubble"]    for e in ex]
pin_ds    = [e["pintrail"]  for e in ex]
spr_ds    = [e["sprtrail"]  for e in ex]
dot_ds    = [e["magdot_d"]  for e in ex]

dot_x = []; dot_y = []
for d_str in dot_ds:
    cx, cy = path_centroid(d_str)
    dot_x.append(cx); dot_y.append(cy)
cx0, cy0 = dot_x[0], dot_y[0]
dot_deltas = [f"{dot_x[k]-cx0:.2f},{dot_y[k]-cy0:.2f}" for k in range(K)]

d_varies = len(set(dot_ds)) > 1
print(f"dot d varies across frames: {d_varies}  "
      f"(deltas range {max(abs(x-cx0) for x in dot_x):.1f}, "
      f"{max(abs(y-cy0) for y in dot_y):.1f} px)")

# --------------- SMIL injection ---------------
T      = K / FPS
DUR    = f"{T:.2f}s"
keyT   = [k / (K - 1) for k in range(K)]
colors = ['red'] * KP + ['blue'] * (K - KP)
fills  = ['#ff0000'] * KP + ['#0000ff'] * (K - KP)

def _style_to_attrs(el, keys=('stroke', 'fill', 'stroke-width',
                               'stroke-linecap', 'stroke-linejoin',
                               'fill-opacity', 'stroke-opacity')):
    s = el.get('style', '')
    if not s:
        return
    kept = []
    for pair in s.split(';'):
        if ':' not in pair:
            kept.append(pair)
            continue
        k, v = pair.split(':', 1)
        if k.strip() in keys:
            el.set(k.strip(), v.strip())
        else:
            kept.append(pair)
    if kept:
        el.set('style', ';'.join(kept))
    else:
        el.attrib.pop('style', None)

def find_target(root, gid):
    for tag in ('path', 'use', 'circle', 'ellipse', 'g'):
        el = _find(root, gid, tag)
        if el is not None:
            return el
    return None

def add_animate(parent, attr, values, keyTimes, dur, repeat, calcMode):
    sub = ET.SubElement(parent, f"{{{SVG_NS}}}animate")
    sub.set("attributeName", attr)
    sub.set("attributeType", "XML")
    sub.set("values", ";".join(str(v) for v in values))
    sub.set("keyTimes", ";".join(f"{k:.6f}" for k in keyTimes))
    sub.set("dur", dur)
    sub.set("begin", "0s")
    sub.set("repeatCount", repeat)
    sub.set("calcMode", calcMode)
    return sub

def add_animate_transform(parent, ttype, values, keyTimes, dur,
                          repeat, calcMode, additive=None):
    sub = ET.SubElement(parent, f"{{{SVG_NS}}}animateTransform")
    sub.set("attributeName", "transform")
    sub.set("attributeType", "XML")
    sub.set("type", ttype)
    sub.set("values", ";".join(values))
    sub.set("keyTimes", ";".join(f"{k:.6f}" for k in keyTimes))
    sub.set("dur", dur)
    sub.set("begin", "0s")
    sub.set("repeatCount", repeat)
    sub.set("calcMode", calcMode)
    if additive:
        sub.set("additive", additive)
    return sub

base_svg = svg_files[0]
tree = ET.parse(base_svg)
root = tree.getroot()

# --- bubble outline + stroke ---
el = find_target(root, "bubble")
if el is None: sys.exit("ERROR: bubble not found")
_style_to_attrs(el)
add_animate(el, "d", bubble_ds, keyT, DUR, REPEAT, "linear")
add_animate(el, "stroke", colors, keyT, DUR, REPEAT, "discrete")

# --- moving dot: prefer `d` animation (portable), fall back to transform ---
el = find_target(root, "magdot")
if el is None: sys.exit("ERROR: magdot not found")
_style_to_attrs(el)
if d_varies:
    add_animate(el, "d", dot_ds, keyT, DUR, REPEAT, "linear")
else:
    add_animate_transform(el, "translate", dot_deltas, keyT,
                          DUR, REPEAT, "linear", additive="sum")

# --- dot colour: red then blue, discrete at unpinning ---
add_animate(el, "fill", fills, keyT, DUR, REPEAT, "discrete")

# --- trails ---
el = find_target(root, "pintrail")
if el is None: sys.exit("ERROR: pintrail not found")
add_animate(el, "d", pin_ds, keyT, DUR, REPEAT, "discrete")

el = find_target(root, "sprtrail")
if el is None: sys.exit("ERROR: sprtrail not found")
add_animate(el, "d", spr_ds, keyT, DUR, REPEAT, "discrete")

tree.write(OUT, encoding='utf-8', xml_declaration=True)
kb = os.path.getsize(OUT) / 1024
print(f"wrote {OUT}  ({kb:.0f} kB,  {K} keyframes,  dur={T:.2f}s,  "
      f"dot={'d-anim' if d_varies else 'transform-anim'})")

plt.close(fig)
shutil.rmtree(WORK_SVG, ignore_errors=True)
