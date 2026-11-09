#!/usr/bin/env python3
"""make_animated_svg.py — post-process matplotlib SVGs to add SMIL animation.

Two-stage pipeline:
  1) matplotlib renders each frame to a static SVG (set_gid markers on the
     artists we want to animate: bubble outline, pinned trail, spreading
     trail, moving dot).
  2) This script parses every frame SVG, extracts the changing attributes
     (path d, dot centroid, stroke), and injects them as <animate> /
     <animateTransform> elements into the first frame SVG.

Safari compatibility note
-------------------------
The moving dot is drawn as a matplotlib Circle patch (which renders as a
<path>) rather than a plot(..., 'o') marker (which renders as a <use>).
Safari is known not to honour SMIL animation of <use x/y>, so the marker
version plays in Chrome but not Safari.  The Circle-based version
animates transform="translate(...)" on a <path>, which Safari supports.

Usage:
    python make_animated_svg.py <repo_dir> <work_dir> <out.svg>
                                [--fps 15] [--single-shot]
"""
import os, sys, glob, subprocess, tempfile, shutil, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# ---------------------------------------------------------------- CLI
if len(sys.argv) < 4:
    sys.exit(__doc__)
REPO, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = 15.0
REPEAT = "indefinite"
for i, a in enumerate(sys.argv):
    if a == "--fps":
        FPS = float(sys.argv[i + 1])
    elif a == "--single-shot":
        REPEAT = "1"

SIM = os.path.join(REPO, "simData") + "/"
RUN = os.path.join(REPO, "run")

RC, PHI_W, CAV = 0.5, 0.9, 0.3
KP, KS, M     = 22, 118, 48
PRIN, SPRD    = "loop_rad00005.txt", "loop_ang00005.txt"
R_GUIDE       = 3.219
R_BAND_END    = 3.832

# ---------------------------------------------------------------- data
pr = np.loadtxt(SIM + PRIN)
assert abs(pr[0, 0] - RC) < 1e-3, "pinned file is not r0 = rc"
phi_p = 1 - pr[:, 2] / np.pi
jp = int(np.argmax(phi_p <= PHI_W + 1e-12))

sp = np.loadtxt(SIM + SPRD)
sp = sp[: int(np.argmax(sp[:, 6])) + 1]
assert abs((1 - sp[-1, 2] / np.pi) - PHI_W) < 1e-3, "spreading file is not phi_w"
js = int(np.argmax(sp[:, 0] >= RC))

r0_p_all = pr[: jp + 1, 0] / pr[: jp + 1, 4]
ph_p_all = 1 - pr[: jp + 1, 2] / np.pi
r0_s_all = sp[js:, 0] / sp[js:, 4]
ph_s_all = 1 - sp[js:, 2] / np.pi

bg_curves = []
for fpath in sorted(glob.glob(SIM + "loop_*.txt")):
    if os.path.basename(fpath) in (PRIN, SPRD):
        continue
    try:
        d = np.loadtxt(fpath)
    except Exception:
        continue
    if d.ndim < 2 or len(d) < 2:
        continue
    lam = d[:, 4]
    bg_curves.append((d[:, 0] / lam, 1 - d[:, 2] / np.pi))
print(f"loaded {len(bg_curves)} background curves from simData/loop_*.txt")

# ---------------------------------------------------------------- schedule
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
            phi = 1 - tab[i, 2] / np.pi,
        ))
K = len(frames)
print(f"frames: {K}   duration: {K / FPS:.2f}s at {FPS} fps")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(13, 10))
gs  = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.35], hspace=0.42)
ax_b   = fig.add_subplot(gs[0])
ax_phi = fig.add_subplot(gs[1])

# --- bubble panel (z vs x, plot_profiles style) --------------------------
xmax_p = max(f["x"].max() for f in frames)
ymax_p = max(f["y"].max() for f in frames)
xlim   = max(4.0, xmax_p + 0.5)
for a, b, c, d_ in [(-xlim, -RC, 0, 0), (RC, xlim, 0, 0),
                    (-RC, -RC, -CAV, 0), (RC, RC, -CAV, 0),
                    (-RC, RC, -CAV, -CAV)]:
    ax_b.plot([a, b], [c, d_], 'k-', lw=1.5, zorder=4, clip_on=False)
ax_b.set_xlim([-xlim, xlim])
ax_b.set_ylim([-CAV - 0.3, ymax_p + 0.4])
ax_b.set_aspect('equal', adjustable='box')
ax_b.set_xlabel(r'$r/\lambda$', fontsize=13)
ax_b.set_ylabel(r'$z/\lambda$', rotation=0, labelpad=16, fontsize=13)
ax_b.tick_params(which='both', direction='in', top=True, right=True)

bub_line, = ax_b.plot([], [], lw=2.5, zorder=5, clip_on=False)
bub_line.set_gid("bubble")

# --- phi_0 vs r_0 panel (figA style) ------------------------------------
ax_phi.set_xlim([0, 4])
ax_phi.set_ylim([0, 1.05])
ax_phi.set_xlabel(r'$r_0/\lambda$', fontsize=13)
ax_phi.set_ylabel(r'$\phi_0/\pi$', rotation=0, labelpad=18, fontsize=13)
ax_phi.tick_params(which='both', direction='in', top=True, right=True)
ax_phi.axvspan(R_GUIDE, R_BAND_END, color='whitesmoke', zorder=0)
ax_phi.plot([RC, RC], [PHI_W, 1.0], 'k:', lw=0.5, zorder=1)
ax_phi.plot([RC, 4.0], [PHI_W, PHI_W], 'k:', lw=0.5, zorder=1)
for r0, ph in bg_curves:
    ax_phi.plot(r0, ph, '-', color='grey', lw=0.35, alpha=0.35, zorder=1)
ax_phi.plot(r0_p_all, ph_p_all, 'r-', lw=1.0, alpha=0.25, zorder=2)
ax_phi.plot(r0_s_all, ph_s_all, 'b-', lw=1.0, alpha=0.25, zorder=2)

pin_trail, = ax_phi.plot([], [], 'r-', lw=2.4, zorder=4)
pin_trail.set_gid("pintrail")
spr_trail, = ax_phi.plot([], [], 'b-', lw=2.4, zorder=4)
spr_trail.set_gid("sprtrail")

# Moving dot as a Circle patch.  This renders as a <path> in the SVG (not
# a <use>), which allows Safari to animate it via <animateTransform>.
# radius is in data units; ~0.05 data units ≈ 14pt on this panel.
dot = Circle((0.0, 0.0), radius=0.05, facecolor='magenta',
             edgecolor='black', linewidth=1.0, zorder=6, clip_on=False)
ax_phi.add_patch(dot)
dot.set_gid("magdot")

fig.suptitle(r'bubble evolution:  $r_c = 0.5\lambda$,  $\phi_w = 0.9\pi$',
             fontsize=14)
fig.tight_layout()

# ---------------------------------------------------------------- update
def update(k):
    fr = frames[k]
    X = np.concatenate([-fr["x"][::-1], fr["x"]])
    Y = np.concatenate([ fr["y"][::-1], fr["y"]])
    bub_line.set_color('r' if fr["ph"] == "p" else 'b')
    bub_line.set_data(X, Y)
    if fr["ph"] == "p":
        i = min(fr["row"], jp)
        pin_trail.set_data(r0_p_all[:i + 1], ph_p_all[:i + 1])
        spr_trail.set_data([r0_s_all[0]], [ph_s_all[0]])   # 1-point placeholder
    else:
        pin_trail.set_data(r0_p_all, ph_p_all)
        j = max(0, fr["row"] - js)
        spr_trail.set_data(r0_s_all[:j + 1], ph_s_all[:j + 1])
    dot.center = (fr["r0"], fr["phi"])

# ---------------------------------------------------------------- render
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

# ---------------------------------------------------------------- extract
def _find(root, gid, tag):
    for el in root.iter():
        if el.get('id') == gid:
            if el.tag.endswith(tag):
                return el
            for ch in el.iter():
                if ch.tag.endswith(tag):
                    return ch
    return None

# Matches positive/negative ints, decimals, and scientific notation.
_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def path_centroid(d):
    """Centre of the bounding box of all coordinate pairs in an SVG path d."""
    if not d:
        return (0.0, 0.0)
    nums = [float(v) for v in _NUM_RE.findall(d)]
    if len(nums) < 4:
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

print("extracting path and dot data …")
ex = [extract(p) for p in svg_files]
bubble_ds = [e["bubble"]   for e in ex]
pin_ds    = [e["pintrail"] for e in ex]
spr_ds    = [e["sprtrail"] for e in ex]

dot_x = []
dot_y = []
for e in ex:
    cx, cy = path_centroid(e["magdot_d"])
    dot_x.append(cx)
    dot_y.append(cy)
cx0, cy0 = dot_x[0], dot_y[0]
dot_deltas = [f"{dot_x[k] - cx0:.2f},{dot_y[k] - cy0:.2f}" for k in range(K)]
print(f"dot centroid at frame 0: ({cx0:.1f}, {cy0:.1f})  "
      f"max delta: "
      f"({max(abs(x - cx0) for x in dot_x):.1f}, "
      f"{max(abs(y - cy0) for y in dot_y):.1f})")

# ---------------------------------------------------------------- inject
T      = K / FPS
keyT   = [k / (K - 1) for k in range(K)]
colors = ['red'] * KP + ['blue'] * (K - KP)

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

def add_animate_transform(parent, ttype, values, keyTimes, dur, repeat, calcMode):
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
    return sub

base_svg = svg_files[0]
tree = ET.parse(base_svg)
root = tree.getroot()

# --- bubble outline + stroke ---
el = find_target(root, "bubble")
if el is None:
    sys.exit("ERROR: could not find id='bubble' in base SVG")
_style_to_attrs(el)
add_animate(el, "d", bubble_ds, keyT, f"{T}s", REPEAT, "linear")
add_animate(el, "stroke", colors, keyT, f"{T}s", REPEAT, "discrete")

# --- moving dot: <animateTransform type="translate"> on the Circle <path> ---
el = find_target(root, "magdot")
if el is None:
    sys.exit("ERROR: could not find id='magdot' in base SVG")
add_animate_transform(el, "translate", dot_deltas, keyT,
                      f"{T}s", REPEAT, "linear")

# --- pinned trail ---
el = find_target(root, "pintrail")
if el is None:
    sys.exit("ERROR: could not find id='pintrail' in base SVG")
add_animate(el, "d", pin_ds, keyT, f"{T}s", REPEAT, "discrete")

# --- spreading trail ---
el = find_target(root, "sprtrail")
if el is None:
    sys.exit("ERROR: could not find id='sprtrail' in base SVG")
add_animate(el, "d", spr_ds, keyT, f"{T}s", REPEAT, "discrete")

tree.write(OUT, encoding='utf-8', xml_declaration=True)

kb = os.path.getsize(OUT) / 1024
print(f"wrote {OUT}  ({kb:.0f} kB,  {K} keyframes,  dur={T:.2f}s,  "
      f"repeat={REPEAT})")

plt.close(fig)
shutil.rmtree(WORK_SVG, ignore_errors=True)
