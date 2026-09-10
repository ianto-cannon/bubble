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

# ------------------------------------------------------------
# Function 1: profiles and height vs volume
# ------------------------------------------------------------
def plot_profiles(nam='rad ang'):
  figProf, axProf = plt.subplots(2, sharex=True)

  for cont in nam.split():
    axInd = 0
    if 'ang' in cont:
      axInd = 1
    elif 'rad' in cont:
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

      # ----- Skip drawing profiles for some cases -----
      if 'rad' in cont and round(df[0, 0] * 10) % 10 != 5:
        continue
      if 'ang' in cont and round(angl * 100) % 20 != 10:
        continue
      
      # Horizontal spacing
      if 'ang' in cont:
        spac = (.7, 2.3, 5.1, 9.5, 16)[round(4 - df[indVol, 2] * 5 / np.pi)]
      if 'rad' in cont:
        spac = (1, 4, 9, 16)[round(df[0, 0] - 0.5)]
        axProf[axInd].plot((spac - df[0, 0], spac + df[0, 0]), (0, 0),
                   c='w', clip_on=False, zorder=3)
        axProf[axInd].plot((spac - df[0, 0], spac + df[0, 0]), (-cav, -cav),
                   c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac - df[0, 0], spac - df[0, 0]), (-cav, 0),
                   c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac + df[0, 0], spac + df[0, 0]), (-cav, 0),
                   c='k', clip_on=False, zorder=3, lw=1)

      # Load and draw profiles for each height level
      for hei in range(5):
        if 'rad' in fname and spac < 2 and hei < 4: continue
        if hei<4: col = 'grey'
        elif 'rad' in fname: col = 'r'
        elif 'ang' in fname: col = 'b'
        heiInd = np.argmin(abs((hei + 1) * df[indVol, 1] / 5 - df[:indVol + 1, 1]))
        fPath = os.path.join(inFol, f'prof{hei:05}' + fname)
        if not os.path.exists(fPath):
          print("run", './run', '1', str(df[heiInd, 5]), fPath)
          subprocess.run(['./run', '1', str(df[heiInd, 5]), fPath])
        with open(fPath, encoding='utf-8') as f:
          prof = np.loadtxt(f)
        footInd = np.argmin(abs(df[heiInd, 6] - prof[:, 6]))
        if prof[footInd,2] <= np.pi*1e-3: listy='dashed'
        else: listy='solid'
        xProf = np.concatenate((-prof[:footInd, 0][::-1], prof[:footInd, 0]))
        xProf = xProf + spac
        yProf = np.concatenate((prof[:footInd, 1][::-1] - prof[footInd, 1],
                    prof[:footInd, 1] - prof[footInd, 1]))
        axProf[axInd].plot(xProf, yProf, c=col, ls=listy, clip_on=False, zorder=4)

        if spac > 2: continue
        if 'ang' in fname: continue

        # Annotations: phi0, r0, s, phi, g
        xAn = xProf[-1]
        yAn = yProf[-1]
        Xarr, Yarr = [], []
        cir = 0.15
        for phi in range(51):
          X = xAn + cir * np.cos(phi * np.pi / 50)
          Y = yAn + cir * np.sin(phi * np.pi / 50)
          for i in reversed(range(len(xProf))):
            if yProf[i] > Y: break
          if xProf[i] > X: break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr, Yarr, c='k', zorder=5, lw=2)

        axProf[axInd].text(xAn + 0.12, yAn + 0.05, "$\\phi_0$",
                   ha='left', va='bottom', c='k', zorder=4)
        if 'rad' in cont:
          axProf[axInd].plot((spac, xProf[-1]), (-cav, -cav),
                     c='k', clip_on=False, zorder=4, lw=2)
          axProf[axInd].text((spac + xProf[-1]) / 2, 0.1 - cav,
                     "$r_0$", ha='center', va='bottom', c='k')
        if 'ang' in cont:
          axProf[axInd].plot((spac, xProf[-1]), (0, 0),
                     c='k', clip_on=False, zorder=5, lw=2)
          axProf[axInd].text((spac + xProf[-1]) / 2, -0.1,
                     "$r_0$", ha='center', va='top', c='k')

        r = np.argmax(xProf)
        axProf[axInd].plot((spac, xProf[r]), (yProf[r], yProf[r]),
                   c='k', clip_on=False, zorder=5, lw=2)
        axProf[axInd].text((spac + xProf[r]) / 2, yProf[r]-0.1,
                   "$r_\\mathrm{max}$", ha='center', va='top', c='k', zorder=4)
        
        h = int(0.5 * len(xProf))
        t = int(0.67 * len(xProf))
        axProf[axInd].plot(xProf[h:t], yProf[h:t], c='k', zorder=5, lw=2)
        theta = np.arctan2(yProf[t + 1] - yProf[t], xProf[t + 1] - xProf[t]) - np.pi / 2
        tri = RegularPolygon((xProf[t], yProf[t]), 3,
                   radius=0.1, orientation=theta,
                   color='k', zorder=5)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(xProf[t] + 0.12, yProf[t] + 0.05, '$s$',
                   va='center', ha='left', c='k', zorder=4)

        t = int(0.6 * len(xProf))
        xAn = xProf[t]
        yAn = yProf[t]
        axProf[axInd].plot((xAn, xAn + 0.3), (yAn, yAn),
                   color='k', zorder=4, lw=2)
        Xarr, Yarr = [], []
        for phi in range(51):
          X = xAn + cir * np.cos(phi * np.pi / 50)
          Y = yAn + cir * np.sin(phi * np.pi / 50)
          for i in range(len(xProf)):
            if xProf[i] > X:
              break
          if yProf[i] > Y:
            break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr, Yarr, c='k', zorder=4)
        axProf[axInd].text(xAn + 0.12, yAn + 0.05, "$\\phi$",
                   ha='left', va='bottom', c='k', zorder=4)


    gravX, gravTailY, gravHeadY = 19.5, 2.8, 2
    axProf[axInd].plot([gravX, gravX], [gravTailY, gravHeadY], c='k')
    tri = RegularPolygon((gravX, gravHeadY), 3,
               radius=0.1, orientation=np.pi,
               color='k', zorder=4)
    axProf[axInd].add_patch(tri)
    axProf[axInd].text(gravX + 0.1, (gravHeadY + gravTailY) / 2,
               '$g$', va='center', ha='left', c='k')
    
    axProf[axInd].tick_params(axis='y', which='both', direction='in', right=True)
    axProf[axInd].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axProf[axInd].set_ylabel('$\\frac{ z }{\\lambda}$', rotation=0, size=22, labelpad=15)
    axProf[axInd].set_ylim([0, 3])
    axProf[axInd].set_xlim([0, 20])
    axProf[axInd].set_aspect('equal', adjustable='box')

    if 'ang' in cont:
      axProf[axInd].text(5e-3, 0.99, '$\\mathrm{(b)~spreading}$',
                 transform=axProf[axInd].transAxes, va='top', ha='left')
    if 'rad' in cont:
      axProf[axInd].text(5e-3, 0.99, '$\\mathrm{(a)~pinned}$',
                 transform=axProf[axInd].transAxes, va='top', ha='left')

  # ----- Save figures -----
  figProf.set_figwidth(12)
  figProf.subplots_adjust(hspace=-0.05)
  outName = outFol + 'pin.pdf'
  print('saving ', outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)

# ------------------------------------------------------------
# Function 2: volume vs radius and angle vs radius
# ------------------------------------------------------------
def plot_volume_and_angle():
  figV, axV = plt.subplots(1, 2, sharey=True)
  figR, axR = plt.subplots(1, 2, sharey=True)
  figA, axA = plt.subplots(1, 2, sharey=True)
  figHei, axHei = plt.subplots(1, 2, sharey=True)
  for ax in axV:
    ax.tick_params(which='both', direction='in', top=True, right=True)
  for ax in axR:
    ax.tick_params(which='both', direction='in', top=True, right=True)
  for ax in axA:
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlim([0, 4])
    ax.set_xlabel('$r_0/\\lambda$')
  for ax in axHei:
    ax.set_ylim([0, 3])
    ax.set_xlim([0, 20])
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlabel('$V/\\lambda^3$')
  axHei[0].set_ylabel('$\\frac{h}{\\lambda}$', rotation=0, size=22, labelpad=10)
  
  xx=np.linspace(0,1)
  axR[0].plot(xx, xx**(1/3), c='k', lw=.1)
  xx=np.linspace(1,4)
  axR[0].plot(xx, xx, c='k', lw=.1)
  xx=np.linspace(0,.5)
  axR[1].plot(xx, 3.832/2*xx, c='k', lw=.1)
  xx=np.linspace(.5,1)
  axR[1].plot(xx, 3.832*xx**2, c='k', lw=.1)
  
  with open(inFol + 'BinRadMaxVol.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  sideInds = df[:,2]<=np.pi*1e-3
  axV[0].plot(df[~sideInds,0], df[~sideInds, 6], c='r', clip_on=False)
  axV[0].plot(df[sideInds,0], df[sideInds, 6], c='r', clip_on=False, ls='dashed')
  axR[0].plot(df[~sideInds,0], df[~sideInds, 5], c='r')
  axR[0].plot(df[sideInds,0], df[sideInds, 5], c='r', ls='dashed')
  axHei[0].plot(df[~sideInds,6], df[~sideInds, 1], c='r', zorder=4, clip_on=False)
  axHei[0].plot(df[sideInds,6], df[sideInds, 1], c='r', zorder=4, clip_on=False, ls='dashed')
  #Make radius bins larger to counteract noise
  minLen=100
  minAng = np.array([ np.min(1 - df[i:i+minLen, 2] / np.pi) for i in range(0, len(df[:,2])-minLen, minLen )])
  rad = np.array([ df[i + minLen//2, 0] for i in range( 0, len(df[:,2])-minLen, minLen )])
  sideInds = minAng>=1-1e-3
  axA[0].plot(rad[~sideInds], minAng[~sideInds]**2, c='r', clip_on=False, zorder=3)
  axA[0].plot(rad[sideInds], minAng[sideInds]**2, c='r', clip_on=False, zorder=3, ls='dashed')
  axI = inset_axes(axV[0], width="38%", height="48%", loc='upper left')
  axI.yaxis.set_label_position("right")
  axI.yaxis.tick_right()
  axI.tick_params(which='both', direction='in', top=True, left=True, right=True)
  axI.set_xscale('log')
  axI.set_yscale('log')
  axI.set_xlim([6e-2, 1.2])
  axI.set_ylim([0.3, 8])
  axI.set_yticks([0.5, 1, 2, 5])
  axI.set_yticklabels(['$0.5$', '$1$', '$2$', '$5$'])
  axI.plot(df[:,0], df[:, 6], c='r')

  xx = np.linspace(0, 4,10000)
  axV[0].plot(xx, 2 * np.pi * xx, lw=.1, c='k')
  axI.plot(xx, 2 * np.pi * xx, lw=.1, c='k')

  axV[0].set_xlabel('$r_0/\\lambda$')
  axR[0].set_xlabel('$r_0/\\lambda$')
  axA[0].set_ylim([0, 1.05])
  axA[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
  axR[0].axvspan(3.219, 3.832, color='whitesmoke')
  axV[0].axvspan(3.219, 3.832, color='whitesmoke')
  axV[0].text(0.97, 0.95, '$\\mathrm{(a)~pinned}$', transform=axV[0].transAxes, va='top', ha='right')
  axR[0].text(0.97, 0.03, '$\\mathrm{(a)~pinned}$', transform=axR[0].transAxes, va='bottom', ha='right')
  axV[0].set_ylabel('$\\frac{V}{\\lambda^3}$', size=22, rotation=0, labelpad=15)
  axR[0].set_ylabel('$\\frac{R_h}{\\lambda}$', size=22, rotation=0, labelpad=15)
  axA[0].axvspan(3.219, 3.832, color='whitesmoke')
  axA[0].set_ylabel('$\\frac{\\phi_0^2}{\\pi^2}$', size=22, rotation=0, labelpad=10)
  axA[0].text(0.98, 0.01, '$\\mathrm{(a)~pinned}$', transform=axA[0].transAxes, va='bottom', ha='right')

  with open(inFol + 'BinRadMaxWid.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axR[0].plot(df[:,0], df[:,9], ls='dotted', c='r', clip_on=False, zorder=3, label='$r_\\mathrm{max}/\\lambda$')
  #axR[0].plot(df[:,0], df[:,0], ls='dotted', c='grey', label='$r_0/\\lambda$')

  #with open(inFol + 'BinRadMaxAng.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  #axA[0].plot(df[:,0], (1 - df[:,2]/np.pi)**2, c='r', ls='dashed')
  axA[0].plot(df[:,0], df[:,0]/3.5, c='k', lw=.1)
  axA[1].plot(df[:,0], df[:,0]/3.5, c='k', lw=.1)
  def line(x, a, b, c, d):
    m = (d - b) / (c - a)
    q = b - m * a
    print(f"y = {m:.4g}x + {q:.4g}")
    return m * x + q
  line, = axA[0].plot(df[:,0], line(df[:,0],.4,.25,3.219,1), c='k', lw=.1, zorder=3)
  line.set_dashes([50, 30])  # 5 pt dash, 3 pt gap
  #axA[0].plot(df[:,0], (df[:,0]-3.219)/3.6+1, c='k', lw=.5)
  
  kwargs = {'marker': 'o', 'ms': 5, 'clip_on': False, 'zorder': 4, 'alpha':.7, 'mew':0}
  for fname in reversed(sorted(os.listdir(inFol))):
    if 'txt' not in fname: continue
    if 'loop' not in fname: continue
    if 'prof' in fname: continue
    with open(inFol + fname, encoding='utf-8') as f: df = np.loadtxt(f)
    if df.ndim < 2: continue
    indVol = np.argmax(df[:, 6])
    angl = 1 - df[indVol, 2] / np.pi
    if 'rad' in fname: 
      axInd = 0
      extremInd = np.argmax(df[:, 2])
      axV[0].plot( df[:indVol, 0], df[:indVol, 6], c='grey', lw=0.1)
      axR[0].plot( df[:indVol, 0], df[:indVol, 5], c='grey', lw=0.1)
      axI.plot( df[:indVol, 0], df[:indVol, 6], c='grey', lw=0.1)
    elif 'ang' in fname: 
      axInd = 1
      extremInd = np.argmax(df[:, 0])
      df[0, 0]=0
      if round(angl * 100) == 100:
        extremInd=1
        df[extremInd, 0]=3.831698723 #from BinAngMaxRad
      axV[1].plot( [angl**3, angl**3], [0, df[indVol, 6]], c='grey', lw=0.1)
      axR[1].plot( (1 - df[:indVol, 2] / np.pi), df[:indVol, 5], c='grey', lw=0.1)
    else: continue
    axHei[axInd].plot(df[:indVol + 1, 6], -df[:indVol + 1, 1], c='grey', lw=0.5, alpha=0.5)
    if 'ang' in fname and round(angl * 100) % 20 == 10 and angl > .4:
      axHei[axInd].text(df[indVol, 6], -df[indVol, 1] +.05,
                rf"$\frac{{\phi_0}}{{\pi}}\!=\!{angl:.1f}$", va='bottom', ha='center', zorder=5)
    if 'rad' in fname and round(df[0, 0] * 10) % 10 == 5 and df[0, 0] > 0.05:
      axHei[axInd].text(df[indVol, 6], -df[indVol, 1] +.05,
                rf"$\frac{{r_0}}{{\lambda}}\!=\!{df[0, 0]:.1f}$", va='bottom', ha='center', zorder=5)
    axA[axInd].plot( [df[0, 0], df[extremInd, 0]], 
      [ (1 - df[0, 2] / np.pi)**2, (1 - df[extremInd, 2] / np.pi)**2], 
      c='grey', lw=0.1)
    if 'rad' in fname:
      if round(df[0, 0] * 10) % 10 != 5: continue
      axV[0].plot( df[:indVol, 0], df[:indVol, 6], c='grey')
      axR[0].plot( df[:indVol, 0], df[:indVol, 5], c='grey')
      axI.plot( df[:indVol, 0], df[:indVol, 6], c='grey')
    if 'ang' in fname:
      if round(angl * 100) % 20 != 10: continue
      axV[1].plot( [angl**3, angl**3], [0, df[indVol, 6]], c='grey')
      axR[1].plot( (1 - df[:indVol, 2] / np.pi), df[:indVol, 5], c='grey')
    axA[axInd].plot( [df[0, 0], df[extremInd, 0]], [ (1 - df[0, 2] / np.pi)**2, (1 - df[extremInd, 2] / np.pi)**2], c='grey')
    if 'ang' in fname and round(angl * 100) == 90:
      axA[axInd].plot( [.5,.5], [ (1 - df[0, 2] / np.pi)**2, 1], c='magenta')
      axA[axInd].plot( [.5, df[extremInd, 0]], [ (1 - df[0, 2] / np.pi)**2, (1 - df[extremInd, 2] / np.pi)**2], c='magenta')
    axHei[axInd].plot(df[:indVol + 1, 6], -df[:indVol + 1, 1], c='grey', zorder=3)
    for hei in range(5):
      if hei<4: col = 'grey'
      elif 'rad' in fname: col = 'r'
      elif 'ang' in fname: col = 'b'
      heiInd = np.argmin(abs((hei + 1) * df[indVol, 1] / 5 - df[:indVol + 1, 1]))
      axHei[axInd].annotate('', xy=(df[heiInd, 6], -df[heiInd, 1]), xytext=(df[heiInd-1, 6], -df[heiInd-1, 1]),
             arrowprops=dict(arrowstyle='->', color=col), zorder=4)
      if 'rad' in fname: 
        axV[0].annotate('', xy=(df[heiInd, 0], df[heiInd, 6]), xytext=(df[heiInd, 0], df[heiInd-1, 6]),
               arrowprops=dict(arrowstyle='->', color=col), zorder=4)
        axI.annotate('', xy=(df[heiInd, 0], df[heiInd, 6]), xytext=(df[heiInd, 0], df[heiInd-1, 6]),
               arrowprops=dict(arrowstyle='->', color=col), zorder=4)
        axR[0].annotate('', xy=(df[heiInd, 0], df[heiInd, 5]), xytext=(df[heiInd-1, 0], df[heiInd-1, 5]),
               arrowprops=dict(arrowstyle='->', color=col), zorder=4)
        axA[0].annotate('', xy=(df[heiInd, 0], (1 - df[heiInd, 2] / np.pi)**2), 
                        xytext=(df[heiInd, 0], (1 - df[heiInd-1, 2] / np.pi)**2),
               arrowprops=dict(arrowstyle='->', color=col), zorder=4)
      if 'ang' in fname: 
        axV[1].annotate('', xy=((1 - df[heiInd, 2] / np.pi)**3, df[heiInd, 6]), 
          xytext=( (1 - df[heiInd, 2] / np.pi)**3, df[heiInd-1, 6]),
               arrowprops=dict(arrowstyle='->', color=col), zorder=4)
        if df[heiInd, 5] <=4: 
          axR[1].annotate('', xy=(1 - df[heiInd, 2] / np.pi, df[heiInd, 5]), 
            xytext=( 1 - df[heiInd, 2] / np.pi, df[heiInd-1, 5]),
                 arrowprops=dict(arrowstyle='->', color=col), zorder=4)
        axA[1].annotate('', xy=(df[heiInd, 0], (1 - df[heiInd, 2] / np.pi)**2), 
                        xytext=(df[heiInd-1, 0], (1 - df[heiInd-1, 2] / np.pi)**2),
               arrowprops=dict(arrowstyle='->', color=col), zorder=4)
  
  # ----- Experimental data for pinned -----
  fname = 'exptData/LesageVolVsContRadSq.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  for i in range(len(df[:, 0])):
    if df[i, 2] > 1:
      continue
    axV[0].plot(df[i, 0] ** 0.5, df[i, 1] * df[i, 0] ** 1.5,
              's', mec='r', mfc='None', clip_on=False, zorder=3)
    axI.plot(df[i, 0] ** 0.5, df[i, 1] * df[i, 0] ** 1.5,
         's', mec='r', mfc='None', zorder=3)

  fname = 'exptData/MoriVolByContCubeVsContSqByCapSq.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  axV[0].plot(0.5 / df[:, 0] ** 0.5, df[:, 1] / df[:, 0] ** 1.5,
            'd', mec='r', mfc='None', clip_on=False, zorder=3)
  axI.plot(0.5 / df[:, 0] ** 0.5, df[:, 1] / df[:, 0] ** 1.5,
       'd', mec='r', mfc='None', zorder=3)

  fname = 'exptData/sasetty23stability.txt'
  print('open', fname)
  with open(fname) as f:
    df = np.loadtxt(f)
  axV[0].plot(df[:, 2] / df[:, 3] / 2,
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
  axV[0].plot(df[:, 0] * 1e-3 / capLen,
            df[:, 4] * 1e-6 * 1e-3 / capLen ** 3,
            '^', mec='r', mfc='None', clip_on=False)
  axI.plot(df[:, 0] * 1e-3 / capLen,
       df[:, 4] * 1e-6 * 1e-3 / capLen ** 3,
       '^', mec='r', mfc='None')
  axV[0].set_xlim([0, 4])
  
  with open(inFol + 'BinAngMaxVol.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axHei[1].plot(df[:,6], df[:, 1], c='b', zorder=4, clip_on=False)
  axV[1].plot( (1 - df[:,2]/np.pi)**3, df[:, 6], c='b', clip_on=False)
  axR[1].plot( (1 - df[:,2]/np.pi), df[:, 5], c='b', clip_on=False)
  axA[1].plot(df[:,0], (1 - df[:,2]/np.pi)**2, c='b')
  axV[1].text(0.03, 0.95, '$\\mathrm{(b)~spreading}$', transform=axV[1].transAxes, va='top', ha='left')
  axA[1].text(0.98, 0.01, '$\\mathrm{(b)~spreading}$', transform=axA[1].transAxes, va='bottom', ha='right')
  
  with open(inFol + 'BinAngMaxWid.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  axR[1].plot( (1 - df[:,2]/np.pi), df[:,9], ls='dotted', c='b', clip_on=False, zorder=3, label='$r_\\mathrm{max}/\\lambda$')
  
  #with open(inFol + 'BinAngMaxRad.txt', encoding='utf-8') as f: df = np.loadtxt(f)
  #axA[1].plot(df[:,0], (1 - df[:,2]/np.pi)**2, c='b', ls='dashed')
  xx = np.linspace(0, 1)
  axV[1].plot(xx**3, 4 * np.pi * (0.0104 * xx * 180) ** 3 / 3, lw=.1, c='k')
  xx = np.linspace(0, 4)
  line, = axA[1].plot(xx, xx/3.219, lw=.1, c='k', zorder=3)
  line.set_dashes([50, 30])  # 5 pt dash, 3 pt gap
  axA[1].axvspan(3.219, 3.832, color='whitesmoke')
  axV[1].set_xlabel('$\\phi_0^3/\\pi^3$')
  axR[1].set_xlabel('$\\phi_0/\\pi$')
  axR[1].text(0.03, 0.97, '$\\mathrm{(b)~spreading}$', transform=axR[1].transAxes, va='top', ha='left')

  # ----- Experimental data for spreading -----
  fname = 'exptData/demirkir24life.txt'
  print('open', fname)
  with open(fname) as f: df = np.loadtxt(f, skiprows=1)
  for i in range(len(df[:, 0])):
    rad = df[i, 1] * 1e-6
    density = df[i, 2] - 0.08988 * 1e-6
    surf = df[i, 3] * 1e-3
    capLen = (surf / density / 9.81) ** 0.5
    mid =  .5*(df[i, 0]/180)**3 + .5*(df[i, 4]/180)**3
    if df[i, 0] - df[i, 4] > 20:
      continue
    axV[1].errorbar(mid, 4 * np.pi / 3 * rad ** 3 / capLen ** 3,
                xerr=[[ (df[i, 0] / 180)**3 - mid], [mid - (df[i, 4] / 180)**3 ]],
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
    axV[1].plot(mid**3, vol, 'v', c='b', mfc='None', zorder=3)
    axV[1].plot([mn**3, mx**3], [vol, vol], c='b', zorder=3)

  fname = 'exptData/huang25effects.txt'
  print('open', fname)
  surf = 72.25e-3
  density = 998
  capLen = (surf / density / 9.81) ** 0.5
  with open(fname) as f: df = np.loadtxt(f, skiprows=1)
  for j in range(3): df[:,j] = (df[:,j]/180)**3
  for i in range(len(df[:, 0])):
    if df[i, 0] < (50/180)**3:
      continue
    axV[1].errorbar( df[i, 0], df[i, 3] / capLen ** 3,
                xerr = [ [df[i, 1] - df[i, 0]] , [df[i, 0] - df[i, 2]] ],
                fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
  axR[0].set_xlim([0, 4])
  axV[1].set_xlim([0, 1])
  axR[1].set_xlim([0, 1])
  axV[0].set_ylim([0, 25])
  axR[0].set_ylim([0, 4])
  #axR[0].set_ylim([.05,1])
  #axR[0].set_xlim([1e-4,10])
  #axR[0].set_yscale('log')
  #axR[0].set_xscale('log')
  #axR[1].set_xscale('log')
  axHei[1].text(5e-3, 0.99, '$\\mathrm{(b)~spreading}$',
              transform=axHei[1].transAxes, va='top', ha='left')
  axHei[0].text(5e-3, 0.99, '$\\mathrm{(a)~pinned}$',
              transform=axHei[0].transAxes, va='top', ha='left')
  
  figA.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.98)
  
  # ----- Save figures -----
  figV.tight_layout(pad=0)
  figR.tight_layout(pad=0)
  figA.tight_layout(pad=0)
  figV.set_figwidth(10)
  figR.set_figwidth(10)
  figA.set_figwidth(10)
  figV.set_figheight(3)
  figR.set_figheight(3)
  figA.set_figheight(3)
  
  figHei.set_figwidth(10)
  figHei.set_figheight(3)
  figHei.tight_layout(pad=0.7)

  fname = outFol + 'MaxVol.pdf'
  print('saving ', fname)
  figV.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0.03)

  fname = outFol + 'Rad.pdf'
  print('saving ', fname)
  figR.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0.03)

  fname = outFol + 'Ang.pdf'
  print('saving ', fname)
  figA.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)
  
  outName = outFol + 'heightVsVol.pdf'   # uses last cont
  print('saving ', outName)
  figHei.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)

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
        if hei<4: col = 'grey'
        elif 'rad' in fname: col = 'r'
        elif 'ang' in fname: col = 'b'
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
        axProf.plot(xProf, yProf, c=col,
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
  plot_profiles(nam='loop_rad loop_ang')
  plot_volume_and_angle()
  plot_graphical_abstract()
