#Ianto Cannon 2026 Feb 4. Find the maximum volume for each contact angle.
import numpy as np
from plot import plot_drop_height_vs_rad, plot_graphical_abstract
from bubble import AdamsBashforthProfile, reorder_drop_height_vs_vol
for b in range(0):
  RadTop = .1*b+.1
  print(b,'RadTop',RadTop)
  AdamsBashforthProfile(1, RadTop, fname=f'data/bub{b:05}.txt')
RadTops = np.concatenate([
    np.logspace(-2, 2,            1000),
    ])
for b in range(len(RadTops)):
  RadTop = 1/RadTops[b]
  print(b,RadTop)
  AdamsBashforthProfile(1, RadTop, angleSave=.001, radSave=0.01)
reorder_drop_height_vs_vol(nam='ang')
reorder_drop_height_vs_vol(nam='rad')
plot_drop_height_vs_rad(nam='loop_rad loop_ang')
