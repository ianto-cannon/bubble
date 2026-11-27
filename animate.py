#!/usr/bin/env python3
"""Create an animated SVG of bubble evolution from simulation data."""
import argparse
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from simdata import bin_files, load_bin

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
ap.add_argument("-o", "--out", default=os.path.join("plots", "growing.svg"),
        help="output SVG path (default: plots/growing.svg)")
ap.add_argument("-f", "--fill-rate", type=float, default=1.0,
        help="dimensionless fill rate d(V/lambda^3)/dt [lambda^3/s]")
ap.add_argument("-p", "--phi", type=float, default=0,
        help="target dimensionless contact angle parameter phi/pi (default: 0)")
ap.add_argument("-r", "--radius", type=float, default=1,
        help="target cavity radius r_c/lambda (default: 1)")
ap.add_argument("-s", "--single-shot", action="store_true",
        help="play the animation once instead of looping indefinitely")
ap.add_argument("-y", "--transparent", action="store_true",
        help="use a transparent background (opaque by default)")
ap.add_argument("-m", "--morph", action=argparse.BooleanOptionalAction, default=True,
        help="animate bubble outline `d` (default; use --no-morph for flipbook)")
args = ap.parse_args()

OUT = args.out
FILL_RATE = args.fill_rate
if FILL_RATE <= 0:
  ap.error("--fill-rate must be positive")
REPEAT = "1" if args.single_shot else "indefinite"
TRANSPARENT = args.transparent
WORK = tempfile.mkdtemp(prefix="anim_")
PROF_DIR = os.path.join(WORK, "prof")
SVG_DIR = os.path.join(WORK, "svg")
os.makedirs(PROF_DIR)
os.makedirs(SVG_DIR)

SIM = "simData/"

# Geometry and dataset selection.
RC_TARGET = args.radius
PHI_TARGET = args.phi
if not 0.0 < RC_TARGET:
  ap.error("--radius must be positive")
if not 0.0 <= PHI_TARGET <= 1.0:
  ap.error("--phi must be between 0 and 1")
PHI2_TARGET = PHI_TARGET ** 2
if not os.path.isdir(SIM):
  ap.error(f"simulation data directory not found: {SIM}")

CAV = 0.3
KP, KS, M = 22, 118, 48
R_GUIDE, R_BAND_END = 3.219, 3.832
BLUE_REF = "BinAngMaxVol.txt"
PREC = 2

def load_candidates(kind, extract):
  """Load `ang` or `rad` datasets, keeping those that satisfy `extract`."""
  results = []
  for path in bin_files(SIM, kind):
    try:
      data = load_bin(path)
      if len(data) < 2:
        continue
      result = extract(path, data)
      if result is not None:
        results.append(result)
    except (OSError, ValueError, IndexError):
      continue
  return results

def rad_extract(path, data):
  return (path, data, float(data[0, 0]), 1.0 - data[:, 2] / np.pi)

def ang_extract(path, data):
  data = data[:int(np.argmax(data[:, 6])) + 1]
  phi2 = (1.0 - float(data[0, 2]) / np.pi) ** 2
  phi_w = 1.0 - float(data[-1, 2]) / np.pi
  if np.isfinite(phi2) and np.isfinite(phi_w) and 0 <= phi2 <= 1 and 0 <= phi_w <= 1:
    return (path, data, phi2, phi_w)
  return None


rad_candidates = load_candidates("rad", rad_extract)
ang_candidates = load_candidates("ang", ang_extract)
if not rad_candidates:
  raise FileNotFoundError(f"No readable rad*.bin datasets in {SIM}")
if not ang_candidates:
  raise FileNotFoundError(f"No readable ang*.bin datasets in {SIM}")

rad_path, pr, RC, phi_pin = min(
  rad_candidates, key=lambda item: abs(item[2] - RC_TARGET))
ang_path, candidate_sp, PHI2_SELECTED, PHI_W = min(
  ang_candidates, key=lambda item: abs(item[2] - PHI2_TARGET))

pin_rows = np.flatnonzero(phi_pin <= PHI_W + 1e-12)
spread_rows = np.flatnonzero(candidate_sp[:, 0] >= RC)

if len(pin_rows) and len(spread_rows):
  jp, js = int(pin_rows[0]), int(spread_rows[0])
  sp = candidate_sp
  HAS_SPREADING = True
  SPRD = os.path.basename(ang_path)
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
print(f"phi/pi target: {PHI_TARGET:.8g}; selected {np.sqrt(PHI2_SELECTED):.8g} "
   f"(inside angle {candidate_sp[0, 2] * 180.0 / np.pi:.4g} deg)")
print(f"configuration: Rc={RC:g} lambda; pinned {PRIN} rows {jp + 1}; "
   + (f"spreading {SPRD} (transition phi_w={PHI_W:g})" if HAS_SPREADING
      else "spreading skipped"))

r0_p_all = pr[:jp + 1, 0] / pr[:jp + 1, 4]
ph_p_all = (1 - pr[:jp + 1, 2] / np.pi) ** 2
r0_s_all = sp[js:, 0] / sp[js:, 4]
ph_s_all = (1 - sp[js:, 2] / np.pi) ** 2

# --------------- Frame schedule — volume rate ---------------
print(f"pinning:   {jp + 1} rows available, requesting {KP}")
assert KP <= jp, (f"KP={KP} > jp={jp}: not enough distinct pinning rows — "
  "reduce KP or re-run sim with denser output")
if HAS_SPREADING:
  print(f"spreading: {len(sp) - js} rows available, requesting {KS}")
  assert KS <= len(sp) - js, (f"KS={KS} > len(sp)-js={len(sp) - js}: "
    "not enough distinct spreading rows")

def volume_schedule(vol, n):
  """Choose rows at approximately uniform ΔV, with strictly increasing V."""
  v = np.maximum.accumulate(np.asarray(vol, dtype=float))
  unique_v, first_rows = np.unique(v, return_index=True)
  if len(unique_v) < n:
    raise ValueError(f"cannot schedule {n} frames: only {len(unique_v)} distinct "
      "volume values are available")
  pos = np.clip(np.searchsorted(unique_v, np.linspace(unique_v[0], unique_v[-1], n)),
                0, len(unique_v) - 1)
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
      raise ValueError(f"cannot fit {n} strictly increasing volume samples; "
        "reduce KP/KS or use denser simulation output")
    print(f"  nudge: resolved {n_dup} duplicate target(s) using distinct volumes")
  idx = first_rows[pos]
  if np.any(np.diff(v[idx]) <= 0):
    raise AssertionError("internal error: selected volumes are not increasing")
  return idx


rows_p = volume_schedule(pr[:jp + 1, 6], KP + 1)[1:]  # skip degenerate row 0
rows_s = (volume_schedule(sp[js:, 6], KS) + js
     if HAS_SPREADING else np.array([], dtype=int))

# --------------- Frame profiles ---------------
def half_profile(row):
  lam_ = row[4]
  pf = os.path.join(PROF_DIR, "prof.txt")
  subprocess.run(["./run", "1", f"{float(row[5]):.12g}", pf],
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
    frames.append(dict(ph=ph, x=x, y=y, row=int(i),
      r0=tab[i, 0] / tab[i, 4], phi2=(1 - tab[i, 2] / np.pi) ** 2))
K = len(frames)

# Physical animation clock: both phases use the same d(V/lambda^3)/dt.
vpin = np.maximum.accumulate(pr[:jp + 1, 6] / pr[:jp + 1, 4] ** 3)
vspr = (np.maximum.accumulate(sp[js:, 6] / sp[js:, 4] ** 3)
    if HAS_SPREADING else np.array([], dtype=float))
pin_duration = (vpin[-1] - vpin[0]) / FILL_RATE
spr_duration = (vspr[-1] - vspr[0]) / FILL_RATE if HAS_SPREADING else 0.0
TRANSITION_EPS = 1e-6 if HAS_SPREADING else 0.0
DURATION = pin_duration + spr_duration + TRANSITION_EPS

frame_times = np.asarray([
  (vpin[fr["row"]] - vpin[0]) / FILL_RATE if fr["ph"] == "p" else
  pin_duration + TRANSITION_EPS + (vspr[fr["row"] - js] - vspr[0]) / FILL_RATE
  for fr in frames])
if np.any(np.diff(frame_times) <= 0):
  raise ValueError("Frame times are not strictly increasing; check simulation volumes")

print(f"frames: {K}   animation: {DURATION:.2f}s")
print(f"fill rate: {FILL_RATE:g} lambda^3/s "
   f"(pinning {pin_duration:.2f}s, spreading {spr_duration:.2f}s)")


# --------------- Figure ---------------
fig, (ax_phi, ax_b) = plt.subplots(1, 2, figsize=(10, 3))
#fig.subplots_adjust(wspace=0.15)

# ---- RIGHT: bubble profile ----
xmax_p = max(f["x"].max() for f in frames)
ymax_p = max(f["y"].max() for f in frames)
xlim_b = xmax_p + 0.6
if False:
  right_xlim, right_ylim = (-3.0, 3.0), (-0.5, 3)
else:
  right_ylim = (-CAV - 0.3, ymax_p + 0.4)
  right_xlim = (-xlim_b, xlim_b)

# Substrate + cavity: one path per visible stroke (left, right, well).
ax_b.plot([-100, -RC], [0, 0], color="k", lw=1, zorder=4, clip_on=True)
ax_b.plot([RC, 100], [0, 0], color="k", lw=1, zorder=4, clip_on=True)
ax_b.plot([-RC, -RC, RC, RC], [0, -CAV, -CAV, 0],
          color="k", lw=1, zorder=4, clip_on=True)

ax_b.set_xlim(right_xlim)
ax_b.set_ylim(right_ylim)
ax_b.set_aspect("equal", adjustable="datalim")
ax_b.set_xlabel(r"$r/\lambda$")
ax_b.set_ylabel(r"$\frac{ z }{\lambda}$", rotation=0, labelpad=5, size=22)
ax_b.tick_params(which="both", direction="in", top=True, right=True)

bub_line, = ax_b.plot([], [], lw=1.5, zorder=5, clip_on=False)
bub_line.set_gid("bubble")

ARC_RADIUS, N_ARC = 0.15, 51
LABEL_DX, LABEL_DY = 0.12, 0.05
arc_line, = ax_b.plot([], [], c="k", lw=2, zorder=5, clip_on=False)
arc_line.set_gid("phi0arc")

_PHI0_TP = TextPath((0, 0), r"$\phi_0$", size=14, usetex=True)
phi0_label = PathPatch(_PHI0_TP, facecolor="k", edgecolor="none", lw=0,
            zorder=4, clip_on=False)
ax_b.add_patch(phi0_label)
phi0_label.set_gid("phi0label")

# ---- LEFT: φ₀²/π² vs r₀/λ ----
ax_phi.set_xlim([0, 4])
ax_phi.set_ylim([0, 1.05])
ax_phi.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax_phi.set_xlabel(r"$r_0/\lambda$")
ax_phi.set_ylabel(r"$\frac{\phi_0^2}{\pi^2}$", rotation=0, labelpad=10, size=22)
ax_phi.tick_params(which="both", direction="in", top=True, right=True)
ax_phi.axvspan(R_GUIDE, R_BAND_END, color="whitesmoke", zorder=0)

xx = np.linspace(0, 4, 500)
ax_phi.plot(xx, xx / 3.5, c="k", lw=0.1, zorder=1)

# Plot decile reference trajectories.
n_decile = 0
for fpath in bin_files(SIM, "ang"):
  try:
    dl = load_bin(fpath)
  except (OSError, ValueError):
    continue
  if len(dl) < 2:
    continue
  indVol = int(np.argmax(dl[:, 6]))
  angl = 1 - dl[indVol, 2] / np.pi
  extremInd = int(np.argmax(dl[:, 0]))
  dl[0, 0] = 0
  if round(angl * 100) == 100:
    extremInd = 1
    dl[extremInd, 0] = 3.831698723
  y0 = (1 - dl[0, 2] / np.pi) ** 2
  ye = (1 - dl[extremInd, 2] / np.pi) ** 2
  x0, xe = dl[0, 0], dl[extremInd, 0]
  # Omit near-horizontal grey reference segments.
  if abs(ye - y0) < 0.015 or round(angl * 100) % 20 != 10:
    continue
  n_decile += 1
  ax_phi.plot([x0, xe], [y0, ye], c="grey", lw=0.1, zorder=2)
  if round(angl * 100) == 90:
    ax_phi.plot([1.5, xe], [y0, ye], c="blue", zorder=2)
  for hei in range(5):
    target_h = (hei + 1) * dl[indVol, 1] / 5
    heiInd = int(np.argmin(np.abs(target_h - dl[:indVol + 1, 1])))
    heiPrev = max(0, heiInd - 1)
    ax_phi.annotate("",
      xy=(dl[heiInd, 0], (1 - dl[heiInd, 2] / np.pi) ** 2),
      xytext=(dl[heiPrev, 0], (1 - dl[heiPrev, 2] / np.pi) ** 2),
      arrowprops=dict(arrowstyle="->", color="grey" if hei < 4 else "b"),
      zorder=3)

print(f"plotted {n_decile} decile arcs with arrowheads")

try:
  with open(SIM + BLUE_REF, encoding="utf-8") as f:
    df_bin = np.loadtxt(f)
  ax_phi.plot(df_bin[:, 0], (1 - df_bin[:, 2] / np.pi) ** 2, c="b", zorder=3)
except FileNotFoundError:
  print(f"WARNING: {BLUE_REF} not found")

# Keep grey path trajectories above the blue and red reference lines.
ax_phi.plot(r0_p_all, ph_p_all, color="grey", zorder=7)
if HAS_SPREADING:
  ax_phi.plot(r0_s_all, ph_s_all, color="grey", zorder=7)

# Red detachment boundary from BinRadMaxVol.txt.
detachment_file = os.path.join(SIM, "BinRadMaxVol.txt")
if os.path.isfile(detachment_file):
  det = np.loadtxt(detachment_file)
  if det.ndim == 2 and len(det) > 100:
    min_len = 100
    min_ang = np.array([np.min(1 - det[i:i + min_len, 2] / np.pi)
              for i in range(0, len(det[:, 2]) - min_len, min_len)])
    rad = np.array([det[i + min_len // 2, 0]
            for i in range(0, len(det[:, 2]) - min_len, min_len)])
    side = min_ang >= 1 - 1e-3
    ax_phi.plot(rad[~side], min_ang[~side] ** 2, c="r", zorder=5, clip_on=False)
    ax_phi.plot(rad[side], min_ang[side] ** 2, c="r", ls="dashed",
          zorder=5, clip_on=False)
else:
  print(f"WARNING: {detachment_file} not found; red detachment line omitted")

# Moving "V" arrowhead: two subpaths (tip->left, tip->right) morphed as one `d`.
HEAD_LEN_PT = 15.0
HEAD_HALF_ANGLE = np.deg2rad(25.0)
PX_PER_PT = fig.dpi / 72.0
arrow_patch = PathPatch(
  Path(np.zeros((4, 2)), [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]),
  facecolor="none", edgecolor="r", lw=1.0, zorder=10, clip_on=False)
ax_phi.add_patch(arrow_patch)
arrow_patch.set_gid("arrowhead")

def phi0_arc(fr):
  xAn, yAn = fr["x"][-1], fr["y"][-1]
  phis = np.linspace(0.0, np.pi * np.sqrt(fr["phi2"]), N_ARC)
  return xAn + ARC_RADIUS * np.cos(phis), yAn + ARC_RADIUS * np.sin(phis)

def compute_headings_display():
  """Per-frame heading in display space (same-phase neighbours only)."""
  P = [ax_phi.transData.transform((fr["r0"], fr["phi2"])) for fr in frames]
  headings, prev = [], 0.0
  for k in range(K):
    lo = max(0, k - 1)
    hi = min(K - 1, k + 1)
    if lo < k and frames[lo]["ph"] != frames[k]["ph"]:
      lo = k
    if hi > k and frames[hi]["ph"] != frames[k]["ph"]:
      hi = k
    if lo == hi:
      headings.append(prev)
      continue
    dx, dy = P[hi][0] - P[lo][0], P[hi][1] - P[lo][1]
    if np.hypot(dx, dy) > 0:
      prev = float(np.arctan2(dy, dx))
    headings.append(prev)
  return headings

headings_disp = compute_headings_display()

# Force aspect adjustment now, then compute fixed points->data scale for label.
fig.canvas.draw()
fig_w, fig_h = fig.get_size_inches()
ax_pos = ax_b.get_position()
_SX = (right_xlim[1] - right_xlim[0]) / (ax_pos.width * fig_w * 72)
_SY = (right_ylim[1] - right_ylim[0]) / (ax_pos.height * fig_h * 72)
_ext = _PHI0_TP.get_extents()

def _phi0_label_transform(foot_x, foot_y):
  """Position the label's bottom-left at (foot + LABEL_DX, foot + LABEL_DY)."""
  tx = foot_x + LABEL_DX - _ext.x0 * _SX
  ty = foot_y + LABEL_DY - _ext.y0 * _SY
  return Affine2D().scale(_SX, _SY).translate(tx, ty) + ax_b.transData

def update(k):
  fr = frames[k]
  bub_line.set_color("r" if fr["ph"] == "p" else "b")
  bub_line.set_data(np.r_[-fr["x"][::-1], fr["x"]],
                    np.r_[fr["y"][::-1], fr["y"]])
  phi0_label.set_transform(_phi0_label_transform(fr["x"][-1], fr["y"][-1]))
  arc_line.set_data(*phi0_arc(fr))

  # Arrowhead: tip at data point, V opening backwards along the heading.
  tip_data = np.array([fr["r0"], fr["phi2"]])
  tip_disp = ax_phi.transData.transform(tip_data)
  theta = headings_disp[k]
  back = np.array([-np.cos(theta), -np.sin(theta)])
  ca, sa = np.cos(HEAD_HALF_ANGLE), np.sin(HEAD_HALF_ANGLE)
  left_dir = np.array([back[0]*ca - back[1]*sa, back[0]*sa + back[1]*ca])
  right_dir = np.array([back[0]*ca + back[1]*sa, -back[0]*sa + back[1]*ca])
  L_px = HEAD_LEN_PT * PX_PER_PT
  inv = ax_phi.transData.inverted()
  left_data = inv.transform(tip_disp + L_px * left_dir)
  right_data = inv.transform(tip_disp + L_px * right_dir)
  arrow_patch.set_path(Path(np.array([tip_data, left_data, tip_data, right_data]),
    [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]))
  arrow_patch.set_edgecolor("r" if fr["ph"] == "p" else "b")

# --------------- Render frames ---------------
print(f"rendering {K} frame SVGs into {SVG_DIR}")
update(0)
fig.canvas.draw()
tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())

svg_files = []
for k in range(K):
  update(k)
  p = os.path.join(SVG_DIR, f"f{k:04d}.svg")
  fig.savefig(p, format="svg", bbox_inches=tight_bbox, transparent=TRANSPARENT)
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
  return next((el for el in root.iter() if el.get("id") == gid), None)

def group_path(root, gid, merge=False):
  """Attributes of the first path in group `gid`.  With merge=True the `d`
  of every path in the group is joined into one multi-subpath string."""
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
  nums = [float(x) for x in _NUM_RE.findall(d_string)]
  if len(nums) < 2:
    return 0.0, 0.0
  xs, ys = nums[0::2], nums[1::2]
  return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

GROUPS = ("bubble", "arrowhead", "phi0arc", "phi0label")
ex = [{g: group_path(ET.parse(p).getroot(), g) for g in GROUPS} for p in svg_files]
bubble_ds = [ex[k]["bubble"]["d"] if ex[k]["bubble"] else None for k in range(K)]
arc_ds = [ex[k]["phi0arc"]["d"] if ex[k]["phi0arc"] else None for k in range(K)]
label_centers = [path_bbox_center(ex[k]["phi0label"]["d"])
         if ex[k]["phi0label"] else (0.0, 0.0) for k in range(K)]

# --------------- Build the animation ---------------
DUR = f"{DURATION:.6f}s"
FREEZE = (REPEAT == "1")
ALPHA = pin_duration / DURATION

def mk_animate(parent, name, values, key_times, calc_mode):
  an = ET.SubElement(parent, f"{{{SVG_NS}}}animate")
  an.set("attributeName", name)
  an.set("values", values)
  an.set("keyTimes", key_times)
  an.set("calcMode", calc_mode)
  an.set("dur", DUR)
  an.set("begin", "0s")
  an.set("repeatCount", REPEAT)
  if FREEZE:
    an.set("fill", "freeze")
  return an

def add_visibility(el, k):
  if k == 0:
    vals, kts = "visible;hidden", f"0;{frame_times[1] / DURATION:.9f}"
  elif k == K - 1:
    vals, kts = "hidden;visible", f"0;{frame_times[k] / DURATION:.9f}"
  else:
    vals = "hidden;visible;hidden"
    kts = (f"0;{frame_times[k] / DURATION:.9f};"
           f"{frame_times[k + 1] / DURATION:.9f}")
  mk_animate(el, "visibility", vals, kts, "discrete")

def d_values(ds):
  return ";".join([ds[0]] + ds[1:] + [ds[-1], ds[-1]])

def emit_flipbook(gid, at_list, strict=False):
  for k in range(K):
    at = at_list[k]
    if at is None:
      if strict:
        raise SystemExit(f"ERROR: frame {k} has no {gid} path")
      continue
    el = ET.SubElement(groups[gid], f"{{{SVG_NS}}}path", attrib=dict(at))
    el.set("visibility", "visible" if k == K - 1 else "hidden")
    add_visibility(el, k)

def emit_d(gid, ds, transform=None):
  attrs = dict(ex[-1][gid])
  if transform:
    attrs = transform(attrs)
  attrs["d"] = ds[-1]
  el = ET.SubElement(groups[gid], f"{{{SVG_NS}}}path", attrib=attrs)
  mk_animate(el, "d", d_values(ds), kts_str, "linear")
  return el

def _arrow_attrs(attrs):
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
  return attrs

tree = ET.parse(svg_files[0])
root = tree.getroot()
for ch in list(root):
  if ch.tag.endswith("metadata"):
    root.remove(ch)

groups = {}
for gid in GROUPS:
  g = find_group(root, gid)
  if g is None:
    raise SystemExit(f"ERROR: group '{gid}' not found in the SVG")
  for ch in list(g):
    g.remove(ch)
  groups[gid] = g

# Common keyTimes for `d`/translate animations.
kts_str = ";".join(f"{t:.9f}" for t in
  [0.0] + [t / DURATION for t in frame_times[1:]] + [1.0, 1.0])

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
  mk_animate(emit_d("bubble", bubble_ds), "stroke", "red;blue",
             f"0;{ALPHA:.6f}", "discrete")
else:
  emit_flipbook("bubble", bubble_ds, strict=True)

# --- phi0 arc: morph with the bubble when vertex counts match ---
if all(arc_ds):
  if args.morph and len({d.count("L") for d in arc_ds}) == 1:
    emit_d("phi0arc", arc_ds)
  else:
    emit_flipbook("phi0arc", arc_ds)

# --- phi0 label: translate animation on a fixed shape ---
label_at = [ex[k]["phi0label"] for k in range(K)]
if all(label_at):
  if args.morph:
    lbl_move = ET.SubElement(groups["phi0label"], f"{{{SVG_NS}}}g")
    ET.SubElement(lbl_move, f"{{{SVG_NS}}}path", attrib=dict(ex[-1]["phi0label"]))
    lx_last, ly_last = label_centers[-1]
    lbl_vals = [f"{x - lx_last:.{PREC}f},{y - ly_last:.{PREC}f}"
                for x, y in label_centers]
    lbl_vals = lbl_vals[:1] + lbl_vals[1:] + ["0,0", "0,0"]
    at = ET.SubElement(lbl_move, f"{{{SVG_NS}}}animateTransform")
    at.set("attributeName", "transform")
    at.set("type", "translate")
    at.set("values", ";".join(lbl_vals))
    at.set("keyTimes", kts_str)
    at.set("calcMode", "linear")
    at.set("dur", DUR)
    at.set("begin", "0s")
    at.set("repeatCount", REPEAT)
    if FREEZE:
      at.set("fill", "freeze")
  else:
    emit_flipbook("phi0label", label_at)

# --- Arrowhead ---
arrow_ds = [ex[k]["arrowhead"]["d"] if ex[k]["arrowhead"] else None for k in range(K)]
if not all(arrow_ds):
  raise SystemExit("ERROR: some frames have no arrowhead path")

arrow_counts = {d.count("L") for d in arrow_ds}
if args.morph and len(arrow_counts) != 1:
  print(f"WARNING: arrowhead --morph disabled (L counts differ: {sorted(arrow_counts)})")

if args.morph and len(arrow_counts) == 1:
  print(f"  arrowhead morph: {next(iter(arrow_counts))} line segments / frame")
  mk_animate(emit_d("arrowhead", arrow_ds, transform=_arrow_attrs),
             "stroke", "red;blue", f"0;{ALPHA:.6f}", "discrete")
else:
  emit_flipbook("arrowhead", arrow_ds)

# --------------- Light / Dark Mode CSS Injection ---------------
SKIP_GROUPS = {"bubble", "arrowhead", "phi0arc", "phi0label"}
DARK_BG = "#0d1117"
DARK_FG = "#ffffff"
DARK_BAND = "#21262d"
DARK_REF = "#8b949e"

def add_css_class(el, cls_name):
  existing = el.get("class", "").split()
  if cls_name not in existing:
    existing.append(cls_name)
  el.set("class", " ".join(existing))

def classify_for_theme(el, in_skip=False):
  if el.get("id") in SKIP_GROUPS:
    in_skip = True
  style_str = " ".join(el.attrib.values()).lower()
  if "fill: #ffffff" in style_str or "fill:#ffffff" in style_str:
    add_css_class(el, "plot-bg")
  if "#f5f5f5" in style_str or "whitesmoke" in style_str:
    add_css_class(el, "plot-band")
  if not in_skip:
    if "stroke: #000000" in style_str or "stroke:#000000" in style_str:
      add_css_class(el, "plot-fg-stroke")
    if "fill: #000000" in style_str or "fill:#000000" in style_str:
      add_css_class(el, "plot-fg-fill")
    if "stroke: #808080" in style_str or "stroke:#808080" in style_str:
      add_css_class(el, "plot-ref-stroke")
    if "fill: #808080" in style_str or "fill:#808080" in style_str:
      add_css_class(el, "plot-ref-fill")
  for child in el:
    classify_for_theme(child, in_skip)

classify_for_theme(root)

# Tag animated phi0 arc/label so they also flip to light text in dark mode.
for gid in ("phi0arc", "phi0label"):
  g = find_group(root, gid)
  if g is not None:
    for el in g.iter():
      if el.tag.endswith("path"):
        st = el.get("style", "")
        if "stroke: #000000" in st:
          add_css_class(el, "plot-fg-stroke")
        if "fill: #000000" in st:
          add_css_class(el, "plot-fg-fill")

defs = root.find(f"{{{SVG_NS}}}defs")
if defs is None:
  defs = ET.Element(f"{{{SVG_NS}}}defs")
  root.insert(0, defs)
style_el = defs.find(f"{{{SVG_NS}}}style")
if style_el is None:
  style_el = ET.SubElement(defs, f"{{{SVG_NS}}}style", {"type": "text/css"})
  style_el.text = ""

style_el.text = (style_el.text or "") + f"""
@media (prefers-color-scheme: dark) {{
 text, tspan, use {{ fill: {DARK_FG} !important; }}
 g[id^="text_"] path, g[id^="text_"] use,
 #phi0label path, #phi0label use {{
   fill: {DARK_FG} !important;
   stroke: none !important;
 }}
 path[fill="#000000"], path[fill="#000"],
 path[style*="fill: #000000"], path[style*="fill:#000000"],
 use[fill="#000000"], use[fill="#000"] {{
   fill: {DARK_FG} !important;
   stroke: {DARK_FG} !important;
 }}
 .plot-bg {{ fill: {DARK_BG} !important; }}
 .plot-band {{ fill: {DARK_BAND} !important; stroke: {DARK_BAND} !important; }}
 .plot-fg-stroke {{ stroke: {DARK_FG} !important; }}
 .plot-fg-fill {{ fill: {DARK_FG} !important; }}
 .plot-ref-stroke {{ stroke: {DARK_REF} !important; }}
 .plot-ref-fill {{ fill: {DARK_REF} !important; }}
}}
"""

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
tree.write(OUT, encoding="utf-8", xml_declaration=True)

print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1024:.0f} kB,  {K} frames,  "
   f"mode={'morph' if args.morph else 'flipbook'},  "
   f"dur={DURATION:.2f}s,  fill_rate={FILL_RATE:g} lambda^3/s)")

plt.close(fig)
shutil.rmtree(WORK, ignore_errors=True)
