import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcdefaults()
plt.rcParams.update({"text.usetex": True, 'font.size': 14})

# ------------------------------------------------------------
# Global settings and helper
# ------------------------------------------------------------
inFol = 'simData/'
outFol = 'plots/'
cav = 0.3

def colRB(cont, r):
  if r < 1:
    colVal = r ** 0.5
    if 'rad' in cont:
      return (colVal, 0, 0)
    else:
      return (0, 0, colVal)
  else:
    colVal = 1 - 1 / r
    if 'rad' in cont:
      return (1, colVal, colVal)
    else:
      return (colVal, colVal, 1)

# ------------------------------------------------------------
# Function 1: profiles and height vs volume
# ------------------------------------------------------------
def plot_profiles_and_height_vs_vol(nam='rad ang bub'):
  figProf, axProf = plt.subplots(2, sharex=True)
  figHei, axHei = plt.subplots(1, 2, sharey=True)

  for cont in nam.split():
    axInd = 0
    if 'ang' in cont:
      axInd = 1
    elif 'rad' in cont or 'bub' in cont:
      axInd = 0
    else:
      continue

    for fname in reversed(sorted(os.listdir(inFol))):
      if 'prof' in fname or 'txt' not in fname or cont not in fname:
        continue
      with open(inFol + fname, encoding='utf-8') as f:
        df = np.loadtxt(f)
      if df.ndim < 2:
        continue

      # Normalise
      df[:, 0] /= df[:, 4]
      df[:, 1] /= df[:, 4]
      df[:, 5] /= df[:, 4]
      df[:, 6] /= df[:, 4] ** 3
      df[:, 7] /= df[:, 4] ** 2
      df[:, 8] /= df[:, 4]
      df[:, 4] /= df[:, 4]

      indVol = np.argmax(df[:, 6])
      angl = 1 - df[indVol, 2] / np.pi

      # ----- Height vs volume (light grey + black curves) -----
      axHei[axInd].plot(df[:indVol + 1, 6], -df[:indVol + 1, 1], c='lightgrey', lw=0.5)
      if 'ang' in cont and round(angl * 100) % 20 == 0 and angl > 50 / 180:
        axHei[axInd].text(df[indVol, 6], -df[indVol, 1] + 0.05,
                  rf"${angl:.1f}$", va='bottom', ha='center')
      if 'rad' in cont and round(df[0, 0] * 10) % 10 == 5 and df[0, 0] < 3.2 and df[0, 0] > 0.05:
        axHei[axInd].text(df[indVol, 6], -df[indVol, 1] + 0.05,
                  rf"${df[0, 0]:.1f}$", va='bottom', ha='center')
      if 'bub' in cont and df[0, 5] > 0.5 and df[0, 5] <= 1:
        axHei[axInd].text(df[indVol, 6], -df[indVol, 1] + 0.05,
                  rf"${df[0, 5]:.1f}$", va='center', ha='left')

      # ----- Skip drawing profiles for some cases -----
      if 'rad' in cont and round(df[0, 0] * 10) % 10 != 5:
        continue
      if 'ang' in cont and round(angl * 100) % 20 != 0:
        continue
      
      axHei[axInd].plot(df[:indVol + 1, 6], -df[:indVol + 1, 1],
                c='k', zorder=3)

      # Colour bar inset
      axRt = axProf[axInd].inset_axes(
        (18 / 21.5, 2.6 / 3, (21.1 - 18) / 21.5, 0.2 / 3)
      )
      axRt.set_xscale('log')
      axRt.set_xlabel('$R_h/\\lambda$')
      axRt.set_xlim([0.1, 10])
      axRt.set_yticks([])
      axRt.tick_params(which='both', direction='in', top=True, right=True)
      for ri in range(21):
        Rt = 10 ** ((ri - 10) / 10)
        axRt.plot((Rt, Rt), (0, 1), lw=6, c=colRB(cont, Rt), zorder=-1)

      # Horizontal spacing
      if 'ang' in cont:
        spac = (0.5, 2.3, 5.1, 9.5, 17)[round(4 - df[indVol, 2] * 5 / np.pi)]
      if 'rad' in cont:
        spac = (1, 4.5, 10, 17.5)[round(df[0, 0] - 0.5)]
        axProf[axInd].plot((spac - df[0, 0], spac + df[0, 0]), (0, 0),
                   c='w', clip_on=False, zorder=3)
        axProf[axInd].plot((spac - df[0, 0], spac + df[0, 0]), (-cav, -cav),
                   c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac - df[0, 0], spac - df[0, 0]), (-cav, 0),
                   c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac + df[0, 0], spac + df[0, 0]), (-cav, 0),
                   c='k', clip_on=False, zorder=3, lw=1)

      drawCoord = (6 < spac < 11)

      # Load and draw profiles for each height level
      for hei in reversed(range(5)):
        heiInd = np.argmin(abs((hei + 1) * df[indVol, 1] / 5 - df[:indVol + 1, 1]))
        fPath = os.path.join(inFol, f'prof{hei:05}' + fname)
        if not os.path.exists(fPath):
          print("run", './run', '1', str(df[heiInd, 5]), fPath)
          subprocess.run(['./run', '1', str(df[heiInd, 5]), fPath])
        with open(fPath, encoding='utf-8') as f:
          prof = np.loadtxt(f)
        footInd = np.argmin(abs(df[heiInd, 6] - prof[:, 6]))

        axHei[axInd].plot(prof[footInd, 6], -prof[footInd, 1],
                  'o', ms=5, c=colRB(cont, df[heiInd, 5]),
                  clip_on=False, zorder=4)

        xProf = np.concatenate((-prof[:footInd, 0][::-1], prof[:footInd, 0]))
        xProf = xProf + spac
        yProf = np.concatenate((prof[:footInd, 1][::-1] - prof[footInd, 1],
                    prof[:footInd, 1] - prof[footInd, 1]))
        axProf[axInd].plot(xProf, yProf,
                   c=colRB(cont, df[heiInd, 5]),
                   clip_on=False, zorder=4)

        if not drawCoord or hei < 4:
          continue

        # Annotations: phi0, r0, s, phi, g
        xAn = xProf[-1]
        yAn = yProf[-1]
        Xarr, Yarr = [], []
        for phi in range(51):
          X = xAn + 0.17 * np.cos(phi * np.pi / 50)
          Y = yAn + 0.17 * np.sin(phi * np.pi / 50)
          for i in range(len(xProf)):
            if xProf[i] > X:
              break
          if yProf[i] > Y:
            break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr, Yarr, c='k', zorder=5, lw=2)

        if 'rad' in cont:
          axProf[axInd].text(xAn + 0.1, yAn + 0.1, "$\\phi_0$",
                     ha='left', va='bottom', c='k')
          axProf[axInd].plot((spac, xProf[-1]), (-cav, -cav),
                     c='k', clip_on=False, zorder=4, lw=2)
          axProf[axInd].text((spac + xProf[-1]) / 2, 0.1 - cav,
                     "$r_0$", ha='center', va='bottom', c='k')
        if 'ang' in cont:
          axProf[axInd].text(xAn + 0.1, yAn + 0.1, "$\\phi_0$",
                     ha='left', va='bottom', c='k')
          axProf[axInd].plot((spac, xProf[-1]), (0, 0),
                     c='k', clip_on=False, zorder=5, lw=2)
          axProf[axInd].text((spac + xProf[-1]) / 2, -0.1,
                     "$r_0$", ha='center', va='top', c='k')

        h = int(0.5 * len(xProf))
        t = int(0.8 * len(xProf))
        axProf[axInd].plot(xProf[h:t], yProf[h:t], c='k', zorder=5, lw=2)
        theta = np.arctan2(yProf[t + 1] - yProf[t], xProf[t + 1] - xProf[t]) - np.pi / 2
        tri = RegularPolygon((xProf[t], yProf[t]), 3,
                   radius=0.1, orientation=theta,
                   color='k', zorder=5)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(xProf[t] + 0.1, yProf[t], '$s$',
                   va='center', ha='left', c='k', zorder=4)

        t = int(0.63 * len(xProf))
        xAn = xProf[t]
        yAn = yProf[t]
        axProf[axInd].plot((xAn, xAn + 0.3), (yAn, yAn),
                   color='k', zorder=4, lw=2)
        Xarr, Yarr = [], []
        for phi in range(51):
          X = xAn + 0.17 * np.cos(phi * np.pi / 50)
          Y = yAn + 0.17 * np.sin(phi * np.pi / 50)
          for i in range(len(xProf)):
            if xProf[i] > X:
              break
          if yProf[i] > Y:
            break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr, Yarr, c='k', zorder=4)
        axProf[axInd].text(xAn + 0.1, yAn + 0.1, "$\\phi$",
                   ha='left', va='bottom', c='k', zorder=4)

        gravX, gravTailY, gravHeadY = 21, 1.6, 0.8
        axProf[axInd].plot([gravX, gravX], [gravTailY, gravHeadY], c='k')
        tri = RegularPolygon((gravX, gravHeadY), 3,
                   radius=0.1, orientation=np.pi,
                   color='k', zorder=4)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(gravX + 0.1, (gravHeadY + gravTailY) / 2,
                   '$g$', va='center', ha='left', c='k')

    # ----- Axis formatting for this cont -----
    axHei[axInd].tick_params(which='both', direction='in', top=True, right=True)
    axHei[axInd].set_xlabel('$V/\\lambda^3$')
    axHei[0].set_ylabel('$\\frac{h}{\\lambda}$', rotation=0, size=22, labelpad=10)

    axProf[axInd].tick_params(axis='y', which='both', direction='in', right=True)
    axProf[axInd].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axProf[axInd].set_ylabel('$\\frac{ z }{\\lambda}$', rotation=0, size=22, labelpad=15)
    axProf[axInd].set_ylim([0, 3])
    axProf[axInd].set_xlim([0, 21.5])
    axProf[axInd].set_aspect('equal', adjustable='box')
    axHei[axInd].set_ylim([0, 3])
    axHei[axInd].set_xlim([0, 20])

    if 'ang' in cont:
      axProf[axInd].text(5e-3, 0.99, '$\\mathrm{(b)~spreading}$',
                 transform=axProf[axInd].transAxes, va='top', ha='left')
      axHei[axInd].text(5e-3, 0.99, '$\\mathrm{(b)~spreading}$',
                transform=axHei[axInd].transAxes, va='top', ha='left')
    if 'rad' in cont:
      axProf[axInd].text(5e-3, 0.99, '$\\mathrm{(a)~pinned}$',
                 transform=axProf[axInd].transAxes, va='top', ha='left')
      axHei[axInd].text(5e-3, 0.99, '$\\mathrm{(a)~pinned}$',
                transform=axHei[axInd].transAxes, va='top', ha='left')

  # ----- Save figures -----
  figProf.set_figwidth(12)
  figProf.subplots_adjust(hspace=-0.05)
  outName = outFol + 'pin.pdf'
  print('saving ', outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)

  figHei.set_figwidth(10)
  figHei.set_figheight(3)
  figHei.tight_layout(pad=0.7)
  outName = outFol + 'heightVsVol_' + cont + '.pdf'   # uses last cont
  print('saving ', outName)
  figHei.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)


# ------------------------------------------------------------
# Function 2: volume vs radius and angle vs radius
# ------------------------------------------------------------
def plot_volume_and_angle():
  figV, axV = plt.subplots(2, 2, sharex='col', sharey='row')
  figA, axA = plt.subplots(1, 2, sharey=True)
  for ax in axV.flat:
    ax.tick_params(which='both', direction='in', top=True, right=True)
  for ax in axA:
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlim([0, 4])
    ax.set_xlabel('$r_0/\\lambda$')
  
  with open(inFol + 'BinRadMaxVol.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axV[0,0].plot(df[:,1], df[:, 7], c='r', clip_on=False)
  axV[1,0].plot(df[:,1], df[:, 6], c='r', label='$R_h/\\lambda$')
  axA[0].plot(df[:,1], 1 - df[:,3]/np.pi, c='r', clip_on=False, zorder=3)
  axI = inset_axes(axV[0,0], width="40%", height="50%", loc='upper left')
  axI.yaxis.set_label_position("right")
  axI.yaxis.tick_right()
  axI.tick_params(which='both', direction='in', top=True, left=True, right=True)
  axI.set_xscale('log')
  axI.set_yscale('log')
  axI.set_xlim([6e-2, 1.2])
  axI.set_ylim([0.3, 8])
  axI.set_yticks([0.5, 1, 2, 5])
  axI.set_yticklabels(['$0.5$', '$1$', '$2$', '$5$'])
  axI.plot(df[:,1], df[:, 7], c='r')

  xx = np.linspace(0, 4)
  axV[0,0].plot(xx, 2 * np.pi * xx, linestyle='dotted', c='k')
  axI.plot(xx, 2 * np.pi * xx, linestyle='dotted', c='k')
  axA[0].plot(xx, (xx / 3.5) ** 0.5, linestyle='dotted', c='k', zorder=3)

  axV[1,0].set_xlabel('$r_0/\\lambda$')
  axA[0].set_ylim([0, 1.05])
  axA[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
  axV[1,0].axvspan(3.219, 3.832, color='lightgrey')
  axV[0,0].axvspan(3.219, 3.832, color='lightgrey')
  axI.text(0.03, 0.97, '$\\mathrm{(a)~pinned}$', transform=axI.transAxes, va='top', ha='left')
  axV[1,0].text(0.02, 0.97, '$\\mathrm{(c)~pinned}$', transform=axV[1,0].transAxes, va='top', ha='left')
  axV[0,0].set_ylabel('$\\frac{V_\\mathrm{max}}{\\lambda^3}$', size=22, rotation=0, labelpad=15)
  axA[0].axvspan(3.219, 3.832, color='lightgrey')
  axA[0].set_ylabel('$\\frac{\\phi_0}{\\pi}$', size=22, rotation=0, labelpad=10)
  axA[0].text(0.02, 0.99, '$\\mathrm{(a)~pinned}$', transform=axA[0].transAxes, va='top', ha='left')
  axV[1,0].set_xlim([0, 4])

  with open(inFol + 'BinRadMaxWid.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axV[1,0].plot(df[:,1], df[:,10], ls='dashed', c='r', clip_on=False, zorder=3, label='$r_\\mathrm{max}/\\lambda$')
  axV[1,0].plot(df[:,1], df[:,1], ls='dotted', c='k', label='$r_0/\\lambda$')

  with open(inFol + 'BinRadMaxAng.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axA[0].plot(df[:,1], 1 - df[:,3]/np.pi, c='r', ls='dashed', clip_on=False, zorder=3)
  
  # ----- Experimental data for pinned -----
  fname = 'exptData/LesageVolVsContRadSq.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  for i in range(len(df[:, 0])):
    if df[i, 2] > 1:
      continue
    axV[0,0].plot(df[i, 0] ** 0.5, df[i, 1] * df[i, 0] ** 1.5,
              's', mec='r', mfc='None', clip_on=False, zorder=3)
    axI.plot(df[i, 0] ** 0.5, df[i, 1] * df[i, 0] ** 1.5,
         's', mec='r', mfc='None', zorder=3)

  fname = 'exptData/MoriVolByContCubeVsContSqByCapSq.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  axV[0,0].plot(0.5 / df[:, 0] ** 0.5, df[:, 1] / df[:, 0] ** 1.5,
            'd', mec='r', mfc='None', clip_on=False, zorder=3)
  axI.plot(0.5 / df[:, 0] ** 0.5, df[:, 1] / df[:, 0] ** 1.5,
       'd', mec='r', mfc='None', zorder=3)

  fname = 'exptData/sasetty23stability.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  axV[0,0].plot(df[:, 2] / df[:, 3] / 2,
            df[:, 1] / (df[:, 3] * 1e-3) ** 3,
            'v', mec='r', mfc='None', clip_on=False)
  axI.plot(df[:, 2] / df[:, 3] / 2,
       df[:, 1] / (df[:, 3] * 1e-3) ** 3,
       'v', mec='r', mfc='None')

  fname = 'exptData/gunde01measurement.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f, skiprows=2)
  capLen = (df[:, 2] * 1e-3 / df[:, 1] / 9.81) ** 0.5
  axV[0,0].plot(df[:, 0] * 1e-3 / capLen,
            df[:, 4] * 1e-6 * 1e-3 / capLen ** 3,
            '^', mec='r', mfc='None', clip_on=False)
  axI.plot(df[:, 0] * 1e-3 / capLen,
       df[:, 4] * 1e-6 * 1e-3 / capLen ** 3,
       '^', mec='r', mfc='None')
  axV[0,0].set_xlim([0, 4])
  
  with open(inFol + 'BinAngMaxVol.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axV[0,1].plot(1 - df[:,3]/np.pi, df[:, 7], c='b', clip_on=False)
  axV[1,1].plot(1 - df[:,3]/np.pi, df[:, 6], c='b', label='$R_h/\\lambda$', clip_on=False)
  axA[1].plot(df[:,1], 1 - df[:,3]/np.pi, c='b')
  axI = inset_axes(axV[0,1], width="40%", height="50%", loc='upper left')
  axI.yaxis.set_label_position("right")
  axI.yaxis.tick_right()
  axI.tick_params(which='both', direction='in', top=True, left=True, right=True, pad=6)
  axI.set_xscale('log')
  axI.set_yscale('log')
  axI.set_xlim([0.07, 1])
  axI.set_ylim([0.01, 100])
  axI.plot(1 - df[:,3]/np.pi, df[:, 7], c='b')
  axI.text(0.03, 0.97, '$\\mathrm{(b)~spreading}$',
         transform=axI.transAxes, va='top', ha='left')
  axA[1].text(0.02, 0.99, '$\\mathrm{(b)~spreading}$', transform=axA[1].transAxes, va='top', ha='left')
  
  with open(inFol + 'BinAngMaxWid.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axV[1,1].plot(1 - df[:,3]/np.pi, df[:,10], ls='dashed', c='b', clip_on=False, zorder=3, label='$r_\\mathrm{max}/\\lambda$')
  
  with open(inFol + 'BinAngMaxRad.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axA[1].plot(df[:,1], 1 - df[:,3]/np.pi, c='b', ls='dashed', clip_on=False, zorder=3)
  xx = np.linspace(0, 1)
  axV[0,1].plot(xx, 4 * np.pi * (0.0104 * xx * 180) ** 3 / 3, ls='dotted', c='k')
  axI.plot(xx, 4 * np.pi * (0.0104 * xx * 180) ** 3 / 3, ls='dotted', c='k')
  axA[1].plot(3.219 * xx ** 2, xx, ls='dotted', c='k', zorder=3)
  axV[1,1].set_xlabel('$\\phi_0/\\pi$')
  axV[1,1].text(0.02, 0.97, '$\\mathrm{(d)~spreading}$', transform=axV[1,1].transAxes, va='top', ha='left')

  # ----- Experimental data for spreading -----
  fname = 'exptData/demirkir24life.txt'
  print('open', fname)
  with open(fname) as f: df = np.loadtxt(f, skiprows=1)
  for i in range(len(df[:, 0])):
    rad = df[i, 1] * 1e-6
    density = df[i, 2] - 0.08988 * 1e-6
    surf = df[i, 3] * 1e-3
    capLen = (surf / density / 9.81) ** 0.5
    mid = (df[i, 0] + df[i, 4]) / 2 / 180
    if df[i, 0] - mid * 180 > 20:
      continue
    axV[0,1].errorbar(mid, 4 * np.pi / 3 * rad ** 3 / capLen ** 3,
                xerr=[[df[i, 0] / 180 - mid], [mid - df[i, 4] / 180]],
                fmt='^', c='b', mfc='None', clip_on=False, zorder=3)
    axI.errorbar(mid, 4 * np.pi / 3 * rad ** 3 / capLen ** 3,
           xerr=[[df[i, 0] / 180 - mid], [mid - df[i, 4] / 180]],
           fmt='^', c='b', mfc='None', clip_on=False, zorder=3)

  fname = 'exptData/allred21role.txt'
  print('open', fname)
  with open(fname) as f: df = np.loadtxt(f, skiprows=1)
  for i in range(len(df[:, 0])):
    if (max(df[i, 2:]) - min(df[i, 2:])) > 20:
      continue
    if max(df[i, 2:]) < 20:
      continue
    rad = df[i, 1] / 2
    capLen = df[i, 0] / df[i, 4] / 0.0208 / 2 ** 0.5
    vol = 4 * np.pi / 3 * rad ** 3 / capLen ** 3
    mn = df[i, 2] / 180
    mid = df[i, 4] / 180
    mx = df[i, 3] / 180
    if mid < mn or mid > mx:
      continue
    axV[0,1].plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
    axV[0,1].plot([mn, mx], [vol, vol], c='b', zorder=3)
    axI.plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
    axI.plot([mn, mx], [vol, vol], c='b', zorder=3)

  fname = 'exptData/huang25effects.txt'
  print('open', fname)
  surf = 72.25e-3
  density = 998
  capLen = (surf / density / 9.81) ** 0.5
  with open(fname) as f: df = np.loadtxt(f, skiprows=1)
  for i in range(len(df[:, 0])):
    if df[i, 0] < 50:
      continue
    axV[0,1].errorbar(df[i, 0] / 180, df[i, 3] / capLen ** 3,
                xerr=[[df[i, 1] / 180 - df[i, 0] / 180],
                  [df[i, 0] / 180 - df[i, 2] / 180]],
                fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
    axI.errorbar(df[i, 0] / 180, df[i, 3] / capLen ** 3,
           xerr=[[df[i, 1] / 180 - df[i, 0] / 180],
               [df[i, 0] / 180 - df[i, 2] / 180]],
           fmt='d', c='b', mfc='None', clip_on=False, zorder=3)


  axV[1,0].legend(loc="upper left", bbox_to_anchor=(0.01, 0.9), frameon=False, borderaxespad=0)
  axV[1,1].legend(loc="upper left", bbox_to_anchor=(0.01, 0.9), frameon=False, borderaxespad=0)

  axV[1,0].set_xlim([0, 4])
  axV[1,1].set_xlim([0, 1])
  axV[0,0].set_ylim([0, 30])
  axV[1,0].set_ylim([0, 4])
  figA.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.98)
  
  # ----- Save figures -----
  figV.tight_layout(pad=0)
  figA.tight_layout(pad=0)
  figV.set_figwidth(10)
  figA.set_figwidth(10)
  figV.set_figheight(6)
  figA.set_figheight(3)

  fname = outFol + 'MaxVol.pdf'
  print('saving ', fname)
  figV.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)

  fname = outFol + 'Ang.pdf'
  print('saving ', fname)
  figA.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)

def plot_graphical_abstract(nam='rad ang'):
  figProf, axProf = plt.subplots(1)
  figProf.set_figwidth(6)
  p = -1.2
  s = 1.2
  for cont in nam.split():
    for fname in reversed(sorted(os.listdir(inFol))):
      if 'prof' in fname or 'loop' not in fname or 'txt' not in fname or cont not in fname:
        continue
      with open(inFol + fname, encoding='utf-8') as f: df = np.loadtxt(f)
      if df.ndim < 2:
        continue
      df[:, 0] /= df[:, 4]
      df[:, 1] /= df[:, 4]
      df[:, 5] /= df[:, 4]
      df[:, 6] /= df[:, 4] ** 3
      df[:, 7] /= df[:, 4] ** 2
      df[:, 8] /= df[:, 4]
      df[:, 4] /= df[:, 4]
      indVol = np.argmax(df[:, 6])
      angl = 1 - df[indVol, 2] / np.pi
      if 'rad' in cont and round(df[0, 0] * 10) != 5:
        continue
      if 'ang' in cont and round(angl * 100) != 40:
        continue
      if 'rad' in cont:
        spac = p
      if 'ang' in cont:
        spac = s
      for hei in range(5):
        heiInd = np.argmin(abs((hei + 1) * df[indVol, 1] / 5 - df[:indVol + 1, 1]))
        fPath = inFol + f'prof{hei:05}' + fname
        if not os.path.exists(fPath):
          print("run", './run', '1', str(df[heiInd, 5]), fPath)
          subprocess.run(['./run', '1', str(df[heiInd, 5]), fPath])
        with open(fPath, encoding='utf-8') as f: prof = np.loadtxt(f)
        footInd = np.argmin(abs(df[heiInd, 6] - prof[:, 6]))
        xProf = np.concatenate((-prof[:footInd, 0][::-1], prof[:footInd, 0]))
        xProf = xProf + spac
        yProf = np.concatenate((prof[:footInd, 1][::-1] - prof[footInd, 1],
                    prof[:footInd, 1] - prof[footInd, 1]))
        axProf.plot(xProf, yProf, c=colRB(cont, df[heiInd, 5]),
              clip_on=False, zorder=4)
  axProf.set_axis_off()
  axProf.set_ylim([-0.5, 2.5])
  axProf.set_xlim([-3, 3])
  axProf.set_aspect('equal', adjustable='box')
  axProf.plot([-3, p - 0.5], [0, 0], c='k', clip_on=False)
  axProf.plot([p + 0.5, 3], [0, 0], c='k', clip_on=False)
  axProf.plot([p - 0.5, p + 0.5], [-0.2, -0.2], c='k', clip_on=False)
  axProf.plot([p - 0.5, p - 0.5], [-0.2, 0], c='k', clip_on=False)
  axProf.plot([p + 0.5, p + 0.5], [-0.2, 0], c='k', clip_on=False)
  axProf.plot([-2.5, -2.5], [0.5, 1.5], c='k', clip_on=False)
  axProf.text(-2.6, 1, "$\\lambda$", ha='right', va='center', c='k')
  axProf.text(p, -0.35, "$\\textrm{pinned}$", ha='center', va='center', c='k')
  axProf.text(s, -0.35, "$\\textrm{spreading}$", ha='center', va='center', c='k')
  outName = 'plots/abstract.pdf'
  print('saving ', outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
  plot_profiles_and_height_vs_vol(nam='loop_rad loop_ang')
  plot_volume_and_angle()
  plot_graphical_abstract()
