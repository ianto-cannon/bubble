#Ianto Cannon 2025 Fab 13. 
#Functions for calculating the interface shape of bubbles on surfaces
#Adapted from code written by Stefan Endres
import numpy as np

def AdamsBashforthProfile(capLen, RadTop, contactAng=-1, fname=None, angleSave=0, radSave=0):
#compute analytical interface shape according to eq 1 of Demirkir2024Langmuir
#Input the Bond number Bo, and the radius of curvature at bubble top; RadTop
#Return the volume of the bubble, radius of the contact patch, height of bubble 
#and height of the centre of mass.
  ds = min( 1e-4*abs(RadTop), 1e-4*abs(capLen), 1e-4 )
  psi=0
  r=0
  z=0
  Volume=0
  centroid=0
  area=0
  dPsiPrev=0
  if fname: adams_txt = open(fname, "w") 
  for i in range(int(1e6)):
    dr = ds * np.cos(psi)
    dz = ds * np.sin(psi)
    if i==0:
      drPrev = dr
      dzPrev = dz
    r += 1.5*dr - 0.5*drPrev
    z += 1.5*dz - 0.5*dzPrev
    drPrev = dr
    dzPrev = dz
    dPsi = ds * (2/RadTop - z/capLen**2 - np.sin(psi)/r)
    psi += 1.5*dPsi-.5*dPsiPrev
    Volume += np.pi*r**2*dz
    centroid += z*np.pi*r**2*dz
    area+= 2*np.pi*r*ds
    if fname: 
      if not i%100 or dPsi>1e-2: 
        print(r, -z, psi, dPsi, capLen, RadTop, Volume, area, centroid/Volume, file=adams_txt)
    if angleSave and i:
      angBin = int(np.floor(psi/np.pi / angleSave))
      angBinPrev = int(np.floor((psi-dPsi)/np.pi / angleSave))
      if angBin != angBinPrev:
        angFname=f'data/ang{max(angBin,angBinPrev):05}.txt'
        ang_txt = open(angFname, "a") 
        print(r, -z, psi, dPsi, capLen, RadTop, Volume, area, centroid/Volume, file=ang_txt)
    if radSave and i:
      nam='rad'
      radBin = int(np.floor(r / radSave))
      radBinPrev = int(np.floor( (r-dr) / radSave))
      if radBin != radBinPrev:
        angFname=f'data/'+nam+f'{max(radBin,radBinPrev):05}.txt'
        ang_txt = open(angFname, "a") 
        print(r, -z, psi, dPsi, capLen, RadTop, Volume, area, centroid/Volume, file=ang_txt)
    if psi<0: break
    if psi>np.pi: break
    if dPsi>0 and dPsiPrev<0: break
    dPsiPrev=dPsi
  print('saved',fname,'capLen',capLen,f'RadTop{RadTop:10g}','i',i)
  centroid /= Volume
  return Volume, r, z, centroid, psi

def reorder_drop_height_vs_vol(nam=''):
  import os
  from scipy.spatial import cKDTree
  folName = 'data/'
  for fname in sorted(os.listdir(folName)):
    if 'loop' in fname or 'txt' not in fname or nam not in fname: continue
    df = np.loadtxt(folName + fname)
    if df.ndim < 2: continue
    N = df.shape[0]
    dfLoop = np.zeros_like(df)
    dfLoop[:, 4] = 1
    # Feature space
    with np.errstate(invalid='ignore', divide='ignore'):
      x = df[:, 1] / df[:, 4]
      y = np.cbrt(df[:, 6]) / df[:, 4]
      vol = df[:, 6] / df[:, 4]**3
    points = np.column_stack((x, y))
    # Build KD-tree
    tree = cKDTree(points)
    used = np.zeros(N, dtype=bool)
    # Initialize
    #dfLoop[0] = df[0]
    #used[0] = True
    for j in range(1, N):
      x_prev = dfLoop[j-1, 1] / dfLoop[j-1, 4]
      y_prev = np.cbrt(dfLoop[j-1, 6]) / dfLoop[j-1, 4]
      vol_prev = dfLoop[j-1, 6] / dfLoop[j-1, 4]**3
      # Query multiple nearest neighbors (important!)
      dists, idxs = tree.query([x_prev, y_prev], k=20)
      # Ensure arrays
      idxs = np.atleast_1d(idxs)
      # Filter valid candidates
      best = None
      best_dist = np.inf
      for i in idxs:
        if used[i] or np.isnan(x[i]): continue
        dis = (x[i] - x_prev)**2 + (y[i] - y_prev)**2
        # volume penalty
        if vol[i] < vol_prev: dis += 1
        if dis < best_dist:
          best = i
          best_dist = dis
      # Fallback if all k neighbors invalid
      if best is None:
        remaining = np.where(~used)[0]
        best = remaining[0]
      dfLoop[j] = df[best]
      used[best] = True
    print('save', folName + 'loop_' + fname)
    np.savetxt(folName + 'loop_' + fname, dfLoop)
