import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
plt.rcParams.update({"text.usetex": True,'font.size' : 14,})

def colRB(cont,r):
  if r<1: 
    colVal = r**.5
    if 'rad' in cont: return ( colVal, 0, 0 )
    else:             return ( 0, 0, colVal )
  else: 
    colVal = 1 - 1/r
    if 'rad' in cont: return ( 1, colVal, colVal )
    else:             return ( colVal, colVal, 1 )

def plot_drop_height_vs_rad(nam='rad ang bub'): 
  import os
  from matplotlib.patches import RegularPolygon
  #from bubble import AdamsBashforthProfile
  inFol = 'simData/'
  outFol = 'plots/'
  cav=.3
  figProf, axProf = plt.subplots(2, sharex=True)
  figHei, axHei = plt.subplots(1, 2, sharey=True)
  figV, axVV = plt.subplots(2, 2, sharex='col', sharey='row')
  axVol = axVV[0,:]
  axWid = axVV[1,:]
  figV.subplots_adjust(hspace=0.07)
  figAng, axAng = plt.subplots(1, 2, sharey=True)
  for cont in nam.split():
    x=[]
    dfDet=[]
    z=[]
    zM=[]
    radM=[]
    for fname in reversed(sorted(os.listdir(inFol))):
      if 'prof' in fname: continue
      if 'txt' not in fname: continue
      if cont not in fname: continue
      with open(inFol+fname, encoding = 'utf-8') as f: df = np.loadtxt(f)
      if df.ndim<2: continue
      df[:,0] /= df[:,4]
      df[:,1] /= df[:,4]
      df[:,5] /= df[:,4]
      df[:,6] /= df[:,4]**3
      df[:,7] /= df[:,4]**2
      df[:,8] /= df[:,4]
      df[:,4] /= df[:,4]
      indVol = np.argmax(df[:,6])
      angl = 1 - df[indVol,2]/np.pi
      if 'ang' in cont: 
        if angl>185/180: x.append(np.nan)
        else: x.append(angl)
        z.append(df[indVol,0])
        zM.append(np.max(df[:indVol+1,0]))
        radM.append(np.max(df[:indVol+1,9]))
        axInd=1
      if 'rad' in cont: 
        x.append(df[indVol,0])
        z.append( 1 - df[indVol,2]/np.pi )
        zM.append(np.min( 1 - df[:indVol+1,2]/np.pi ))
        radM.append(np.max(df[:indVol+1,9]))
        axInd=0
      if 'bub' in cont: 
        x.append(df[indVol,5])
        z.append(df[indVol,0])
        zM.append(np.max(df[:indVol+1,0]))
        axInd=0
      dfDet.append(df[indVol,:])
      if '0.txt' not in fname: continue
      axHei[axInd].plot(df[:indVol+1,6], -df[:indVol+1,1], c='lightgrey', lw=.5)
      if 'ang' in cont and round(angl*100)%20!=0:continue
      if 'ang' in cont and angl>50/180: axHei[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${angl:.1f}$", va='bottom', ha='center')
      if 'rad' in cont and round(df[0,0]*10)%10!=5: continue 
      if 'rad' in cont and df[0,0]<3.2 and df[0,0]>.05: axHei[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${df[0,0]:.1f}$", va='bottom', ha='center')
      if 'bub' in cont and df[0,5]>.5 and df[0,5]<=1: axHei[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${df[0,5]:.1f}$", va='center', ha='left')
      axHei[axInd].plot(df[:indVol+1,6], -df[:indVol+1,1], c='k', zorder=3)
      if 'rad' in cont and round(df[0,0]*10)%10==0: continue 
      if 'ang' in cont and round(angl*100)%20!=0:continue
      axRt = axProf[axInd].inset_axes((18/21.5, 2.6/3, (21.1-18)/21.5, .2/3))
      axRt.set_xscale('log')
      axRt.set_xlabel('$R_h/\\lambda$')
      axRt.set_xlim([.1,10])
      axRt.set_yticks([])
      axRt.tick_params(which='both', direction='in', top=True, right=True)
      for ri in range(21):
        Rt=10**( (ri-10)/10 )
        axRt.plot( (Rt,Rt), (0,1), lw=6, c=colRB(cont,Rt), zorder=-1)
      if 'ang' in cont: spac=(.5,2.3,5.1,9.5,17)[ round( 4-df[indVol,2]*5/np.pi) ]
      if 'rad' in cont: 
        spac=(1,4.5,10,17.5)[ round(df[0,0]-.5) ] 
        axProf[axInd].plot((spac-df[0,0], spac+df[0,0]), (0,0), c='w', clip_on=False, zorder=3)
        axProf[axInd].plot((spac-df[0,0], spac+df[0,0]), (-cav,-cav), c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac-df[0,0], spac-df[0,0]), (-cav,0), c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac+df[0,0], spac+df[0,0]), (-cav,0), c='k', clip_on=False, zorder=3, lw=1)
      if spac>6 and spac<11: drawCoord=True
      else: drawCoord=False
      for hei in reversed(range(5)):
        heiInd = np.argmin( abs( (hei+1)*df[indVol,1]/5 - df[:indVol+1,1] ) )
        fPath = os.path.join(inFol, f'prof{hei:05}' + fname)
        if os.path.exists(fPath): 
          with open(fPath, encoding='utf-8') as f: prof = np.loadtxt(f)
        else: 
          print(f"File not found: {fPath}")
          continue
        footInd = np.argmin( abs( df[heiInd,6] - prof[:,6] ))
        axHei[axInd].plot(prof[footInd,6], -prof[footInd,1], 'o', ms=5, c=colRB( cont, df[heiInd,5] ), clip_on=False, zorder=4)#, mfc='None'
        xProf=np.concatenate(( -prof[:footInd,0][::-1] , prof[:footInd,0] ))
        xProf=xProf+spac
        yProf=np.concatenate(( prof[:footInd,1][::-1] - prof[footInd,1] , prof[:footInd,1] - prof[footInd,1] ))
        axProf[axInd].plot(xProf,yProf, c=colRB(cont,df[heiInd,5]), clip_on=False, zorder=4)
        if not drawCoord: continue
        if hei<4: continue
        xAn = xProf[-1]
        yAn = yProf[-1]
        Xarr = []
        Yarr = []
        for phi in range(51):
          X = xAn + .17*np.cos(phi*np.pi/50)
          Y = yAn + .17*np.sin(phi*np.pi/50)
          for i in range(len(xProf)):
            if xProf[i]>X: break
          if yProf[i]>Y: break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr,Yarr,c='k', zorder=5, lw=2)
        if 'rad' in cont: 
          axProf[axInd].text(xAn+.1, yAn+.1, "$\\phi_0$", ha='left', va='bottom', c='k') 
          axProf[axInd].plot((spac,xProf[-1]), (-cav,-cav), c='k', clip_on=False, zorder=4, lw=2)
          axProf[axInd].text( (spac+xProf[-1])/2, .1-cav, "$r_c$", ha='center', va='bottom', c='k') 
        if 'ang' in cont: 
          axProf[axInd].text(xAn+.1, yAn+.1, "$\\phi_c$", ha='left', va='bottom', c='k') 
          axProf[axInd].plot((spac,xProf[-1]), (0,0), c='k', clip_on=False, zorder=5, lw=2)
          axProf[axInd].text( (spac+xProf[-1])/2, -.1, "$r_0$", ha='center', va='top', c='k') 
        h=int(0.5*(len(xProf)))
        t=int(0.8*(len(xProf)))
        axProf[axInd].plot(xProf[h:t],yProf[h:t], c='k', zorder=5, lw=2)
        theta = np.arctan2(yProf[t+1]-yProf[t], xProf[t+1]-xProf[t])-np.pi/2
        tri = RegularPolygon( (xProf[t], yProf[t]), 3, radius=0.1, orientation=theta, color='k', zorder=5)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(xProf[t]+0.1, yProf[t], '$s$', va='center', ha='left', c='k', zorder=4)
        t=int(0.63*(len(xProf)))
        xAn = xProf[t]
        yAn = yProf[t]
        axProf[axInd].plot((xAn,xAn+.3), (yAn,yAn), color='k', zorder=4, lw=2)
        Xarr = []
        Yarr = []
        for phi in range(51):
          X = xAn + .17*np.cos(phi*np.pi/50)
          Y = yAn + .17*np.sin(phi*np.pi/50)
          for i in range(len(xProf)):
            if xProf[i]>X: break
          if yProf[i]>Y: break
          Xarr.append(X)
          Yarr.append(Y)
        axProf[axInd].plot(Xarr,Yarr,c='k', zorder=4)
        axProf[axInd].text(xAn+.1, yAn+.1, "$\\phi$", ha='left', va='bottom', c='k', zorder=4) 
        gravX=21
        gravTailY=1.6
        gravHeadY=.8
        axProf[axInd].plot([gravX,gravX],[gravTailY,gravHeadY], c='k')
        tri = RegularPolygon( (gravX, gravHeadY), 3, radius=0.1, orientation=np.pi, color='k', zorder=4)
        axProf[axInd].add_patch(tri)
        axProf[axInd].text(gravX+0.1, (gravHeadY+gravTailY)/2, '$g$', va='center', ha='left', c='k')
    x = np.asarray(x)
    dfDet = np.asarray(dfDet)
    z = np.asarray(z)
    axWid[axInd].plot(x,dfDet[:,5],c='b')
    axWid[axInd].plot(x,-dfDet[:,1],c='k')
    axWid[axInd].plot(x,radM,c='g', clip_on=False, zorder=3)
    axWid[axInd].set_ylim([0,3.219])
    maxVind=np.argmax(dfDet[:,6])
    print('maxVol',dfDet[maxVind,:])
    maxVind=np.argmax(-dfDet[:,1])
    print('maxHeight',dfDet[maxVind,:])
    axAng[axInd].tick_params(direction='in')
    axHei[axInd].tick_params(which='both', direction='in', top=True, right=True)
    axHei[axInd].set_xlabel('$V/\\lambda^3$')
    axAng[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=axAng[axInd].transAxes,va='top',ha='left')
    axWid[axInd].tick_params(which='both', direction='in', top=True, right=True)
    axHei[0].set_ylabel('$\\frac{h}{\\lambda}$',rotation=0,size=22,labelpad=10)
    axProf[axInd].tick_params(axis='y', which='both', direction='in', right=True)
    axProf[axInd].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axProf[axInd].set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=15)
    axProf[axInd].set_ylim([0,3])
    axProf[axInd].set_xlim([0,21.5])
    axProf[axInd].set_aspect('equal', adjustable='box')
    axHei[axInd].set_ylim([0,3])
    axHei[axInd].set_xlim([0,20])
    axVol[axInd].set_ylim([0,30])
    axVol[axInd].tick_params(which='both', direction='in', top=True, right=True)
    axAng[axInd].tick_params(which='both', direction='in', top=True, right=True)
    if 'bub' in cont:
      axVol[axInd].plot(x,dfDet[:,6],c='k',clip_on=False)
      axVol[axInd].set_xlabel('$R_t$')
      axWid[axInd].plot(x,z, c='k',clip_on=False)
    if 'ang' in cont:
      axVol[axInd].plot(x,dfDet[:,6],c='b',clip_on=False)
      axI = inset_axes(axVol[axInd], width="40%", height="50%", loc='upper left')
      axI.yaxis.set_label_position("right")
      axI.yaxis.tick_right()
      axI.tick_params(which='both', direction='in', top=True, left=True, right=True, pad=6)
      axI.set_xscale('log')
      axI.set_yscale('log')
      axI.set_xlim([.07,1])
      axI.set_ylim([.01,100])
      axI.plot(x,dfDet[:,6],c='b')
      axAng[axInd].plot( z[::30], x[::30], '.', c='b', clip_on=False, zorder=3)
      axAng[axInd].plot( zM, x, '-', c='b', clip_on=False, zorder=3)
      axAng[axInd].plot( radM, x, '-.', c='g', clip_on=False, zorder=3)
      figAng.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.98)
      xx=np.linspace(0,1)
      axVol[axInd].plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dashed', c='k')
      axI.plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dashed', c='k')
      axAng[axInd].plot(3.219*xx**2, xx, ls='dashed', c='k', zorder=3)
      print(4*np.pi*(.0104*180)**3/3, 'dotted')
      axAng[axInd].set_ylabel('$\\phi_c/\\pi$')
      axWid[axInd].set_xlabel('$\\phi_c/\\pi$')
      axWid[axInd].text(-.08,.7,'$\\frac{h}{\\lambda}$',c='k',transform=axWid[axInd].transAxes,size=22,ha='center')
      axWid[axInd].text(-.08,.5,'$\\frac{R_t}{\\lambda}$',c='b',transform=axWid[axInd].transAxes,size=22,ha='center')
      axWid[axInd].text(-.08,.3,'$\\frac{r_0}{\\lambda}$',c='r',transform=axWid[axInd].transAxes,size=22,ha='center')
      axProf[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=axProf[axInd].transAxes,va='top',ha='left')
      axHei[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=axHei[axInd].transAxes,va='top',ha='left')
      axWid[axInd].set_xlim([0,1])
      axWid[axInd].plot(x,z, c='r',clip_on=False)
      axVol[axInd].set_xlim([0,1])
      axVol[axInd].text(1-5e-3,.99,'$\\mathrm{(a)}$',transform=axVol[axInd].transAxes,va='top',ha='right')
      axVol[axInd].set_ylabel('$\\frac{V_s}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      axAng[axInd].set_xlabel('$\\frac{r_0}{\\lambda}$',size=22,rotation=0,labelpad=10)
      axAng[axInd].set_xlim([0,4])
      fname = 'exptData/demirkir24life.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        rad = df[i,1]*1e-6
        density = df[i,2] -	0.08988*1e-6
        surf = df[i,3]*1e-3
        capLen = (surf/density/9.81)**.5
        mid = (df[i,0]+df[i,4])/2/180
        if df[i,0]-mid*180 > 20: continue
        print(i, [df[i,0]-mid])
        axVol[axInd].errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]/180-mid], [mid-df[i,4]/180] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
        axI.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]/180-mid], [mid-df[i,4]/180] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
      fname = 'exptData/allred21role.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        if (max(df[i,2:]) - min(df[i,2:])) > 20: continue
        if max(df[i,2:]) < 20: continue
        rad = df[i,1]/2
        capLen = df[i,0]/df[i,4]/.0208/2**.5
        vol = 4*np.pi/3 * rad**3 / capLen**3
        mn=df[i,2]/180
        mid=df[i,4]/180
        mx=df[i,3]/180
        if mid<mn: continue
        if mid>mx: continue
        axVol[axInd].plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
        axVol[axInd].plot([mn,mx], [vol,vol], c='b', zorder=3)
        axI.plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
        axI.plot([mn,mx], [vol,vol], c='b', zorder=3)
      fname = 'exptData/huang25effects.txt'
      print('open',fname)
      surf=72.25e-3
      density=998
      capLen = (surf/density/9.81)**.5
      with open(fname) as f: df = np.loadtxt(f, skiprows=1)
      for i in range(len(df[:,0])):
        if df[i,0]<50: continue
        axVol[axInd].errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
        axI.errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
      rads = np.pi*np.arange(0.8, -0.1, -0.2)
      print('rads',rads/np.pi)
    if 'rad' in cont:
      axVol[axInd].plot((3.832,*x),(0,*dfDet[:,6]),c='r',clip_on=False)
      from mpl_toolkits.axes_grid1.inset_locator import inset_axes
      axI = inset_axes(axVol[axInd], width="40%", height="50%", loc='upper left')
      axI.yaxis.set_label_position("right")
      axI.yaxis.tick_right()
      axI.tick_params(which='both', direction='in', top=True, left=True, right=True)
      axI.set_xscale('log')
      axI.set_yscale('log')
      axI.set_xlim([6e-2,1.2])
      axI.set_ylim([.3,8])
      axI.set_yticks([.5,1,2,5])
      axI.set_yticklabels(['$0.5$','$1$','$2$','$5$'])
      axI.plot(x,dfDet[:,6],c='r')
      axAng[axInd].plot( x[::15], z[::15], '.', c='r', clip_on=False, zorder=3)
      axAng[axInd].plot( x, zM, c='r', clip_on=False, zorder=3)
      figAng.subplots_adjust(left=0.1, right=0.88, bottom=0.2, top=0.98)
      xx=np.linspace(0,4)
      axVol[axInd].plot( xx, 2*np.pi*xx, linestyle='dashed', c='k')
      axI.plot( xx, 2*np.pi*xx, linestyle='dashed', c='k')
      axAng[axInd].plot( xx, (xx/3.5)**.5, linestyle='dashed', c='k', zorder=3)
      axAng[axInd].set_xlabel('$r_c/\\lambda$')
      axWid[axInd].set_xlabel('$r_c/\\lambda$')
      axP = axWid[axInd].twinx()
      axP.tick_params(direction='in')
      axWid[axInd].tick_params(right=False)
      axWid[axInd].text(-.08,.4,'$\\frac{R_t}{\\lambda}$',c='b',transform=axWid[axInd].transAxes,size=22,ha='center')
      axWid[axInd].text(-.08,.6,'$\\frac{h}{\\lambda}$',c='k',transform=axWid[axInd].transAxes,size=22,ha='center')
      axWid[axInd].text(1.12,.5,'$\\frac{\\phi_0}{\\pi}$',c='r',transform=axWid[axInd].transAxes,size=22,ha='center')
      axP.set_ylim([.5,1])
      axAng[axInd].set_ylim([0,1.05])
      axAng[axInd].set_yticks([0,.25,.5,.75,1])
      axVol[axInd].axvspan(3.219, 4, color='lightgrey')
      axVol[axInd].text(1-5e-3,.99,'$\\mathrm{(a)}$',transform=axVol[axInd].transAxes,va='top',ha='right')
      axVol[axInd].set_ylabel('$\\frac{V_p}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      axAng[axInd].axvspan(3.219, 4, color='lightgrey')
      axAng[axInd].set_ylabel('$\\frac{\\phi_0}{\\pi}$',size=22,rotation=0,labelpad=10)
      axProf[axInd].text(5e-3,.99,'$\\mathrm{(a)}$',transform=axProf[axInd].transAxes,va='top',ha='left')
      axHei[axInd].text(5e-3,.99,'$\\mathrm{(a)}$',transform=axHei[axInd].transAxes,va='top',ha='left')
      axWid[axInd].set_xlim([0,4])
      axP.plot( x, z, c='r',clip_on=False, zorder=3)#,'.',ms=5
      fname = 'exptData/LesageVolVsContRadSq.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      for i in range(len(df[:,0])):
        if df[i,2]>1:continue
        axVol[axInd].plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', clip_on=False, zorder=3)
        axI.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', zorder=3)
      fname = 'exptData/MoriVolByContCubeVsContSqByCapSq.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      axVol[axInd].plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', clip_on=False, zorder=3)
      axI.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', zorder=3)
      fname = 'exptData/sasetty23stability.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      axVol[axInd].plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None', clip_on=False)
      axI.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None')
      fname = 'exptData/gunde01measurement.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=2)
      capLen=(df[:,2]*1e-3/df[:,1]/9.81)**.5
      axVol[axInd].plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None', clip_on=False)
      axI.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None')
      axVol[axInd].set_xlim([0,4])
  figV.set_figwidth(10)
  figAng.set_figwidth(10)
  figV.set_figheight(6)
  figAng.set_figheight(3)
  fname = outFol+'MaxVolVs_'+cont+'.pdf'
  print('savin ',fname)
  figV.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)
  fname = outFol+'Ang_'+cont+'.pdf'
  print('savin ',fname)
  figAng.savefig(fname, transparent=True, format='pdf')
  fname = outFol+'heightVsVol_'+cont+'.pdf'
  print('savin ',fname)
  figHei.set_figwidth(10)
  figHei.set_figheight(3)
  figHei.tight_layout(pad=.7)
  figHei.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
  figProf.set_figwidth(12)
  figProf.subplots_adjust(hspace=-.05)
  outName = outFol+f'pin.pdf'
  print('savin ',outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)
  return

def plot_graphical_abstract(nam='rad ang'): 
  import os
  #from bubble import AdamsBashforthProfile
  simFol = 'simData/'
  plotFol = 'plots/'
  figProf, axProf = plt.subplots(1)
  figProf.set_figwidth(6)
  p=-1.2
  s=1.2
  for cont in nam.split():
    for fname in reversed(sorted(os.listdir(simFol))):
      if 'prof' in fname: continue
      if 'txt' not in fname: continue
      if cont not in fname: continue
      with open(simFol+fname, encoding = 'utf-8') as f: df = np.loadtxt(f)
      if df.ndim<2: continue
      df[:,0] /= df[:,4]
      df[:,1] /= df[:,4]
      df[:,5] /= df[:,4]
      df[:,6] /= df[:,4]**3
      df[:,7] /= df[:,4]**2
      df[:,8] /= df[:,4]
      df[:,4] /= df[:,4]
      indVol = np.argmax(df[:,6])
      angl = 1 - df[indVol,2]/np.pi
      #print('angl',angl,round(angl*100))
      if '0.txt' not in fname: continue
      if 'rad' in cont and round(df[0,0]*10)!=5: continue 
      if 'ang' in cont and round(angl*100)!=40:continue
      if 'rad' in cont: spac=p
      if 'ang' in cont: spac=s
      for hei in range(5):#5
        heiInd = np.argmin( abs( (hei+1)*df[indVol,1]/5 - df[:indVol+1,1] ) )
        #AdamsBashforthProfile(1, df[heiInd,5], fname=simFol+f'prof{hei:05}'+fname)
        with open(simFol+f'prof{hei:05}'+fname, encoding = 'utf-8') as f: prof = np.loadtxt(f)
        print(f'loaded '+simFol+f'prof{hei:05}'+fname)
        footInd = np.argmin( abs( df[heiInd,6] - prof[:,6] ))
        xProf=np.concatenate(( -prof[:footInd,0][::-1] , prof[:footInd,0] ))
        xProf=xProf+spac
        yProf=np.concatenate(( prof[:footInd,1][::-1] - prof[footInd,1] , prof[:footInd,1] - prof[footInd,1] ))
        axProf.plot(xProf,yProf, c=colRB(cont,df[heiInd,5]), clip_on=False, zorder=4)
  axProf.set_axis_off()
  axProf.set_ylim([-.5,2.5])
  axProf.set_xlim([-3,3])
  #axProf.set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=15)
  axProf.get_xaxis().set_visible(False)
  axProf.tick_params(which='both', direction='in', top=True, right=True)
  axProf.set_aspect('equal', adjustable='box')
  axProf.plot([-3,p-.5],[0,0], c='k', clip_on=False)
  axProf.plot([p+.5,3],[0,0], c='k', clip_on=False)
  axProf.plot([p-.5,p+.5],[-.2,-.2], c='k', clip_on=False)
  axProf.plot([p-.5,p-.5],[-.2,0], c='k', clip_on=False)
  axProf.plot([p+.5,p+.5],[-.2,0], c='k', clip_on=False)
  axProf.plot([-2.5,-2.5],[.5,1.5], c='k', clip_on=False)
  axProf.text(-2.6, 1, "$\\lambda$", ha='right', va='center', c='k') 
  axProf.text(p, -.35, "$\\textrm{pinned}$", ha='center', va='center', c='k') 
  axProf.text(s, -.35, "$\\textrm{spreading}$", ha='center', va='center', c='k') 
  outName = 'plots/abstract.pdf'
  print('savin ',outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)
  return

plot_drop_height_vs_rad(nam='loop_rad loop_ang')
