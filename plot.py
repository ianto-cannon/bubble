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
  from bubble import AdamsBashforthProfile
  inFol = '../mergeDdgc/data/'
  outFol = 'plots/'
  cav=.3
  figProf, axProf = plt.subplots(2, sharex=True)
  fig, ax = plt.subplots(1, 2, sharey=True)
  for cont in nam.split():
    figV, axVV = plt.subplots(2, sharex=True)
    figV.subplots_adjust(hspace=0.07)
    fig2, ax2 = plt.subplots(1)
    x=[]
    dfDet=[]
    z=[]
    zM=[]
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
        axInd=1
      if 'rad' in cont: 
        x.append(df[indVol,0])
        z.append( 1 - df[indVol,2]/np.pi )
        zM.append(np.min( 1 - df[:indVol+1,2]/np.pi ))
        axInd=0
      if 'bub' in cont: 
        x.append(df[indVol,5])
        z.append(df[indVol,0])
        zM.append(np.max(df[:indVol+1,0]))
        axInd=0
      dfDet.append(df[indVol,:])
      if '0.txt' not in fname: continue
      ax[axInd].plot(df[:indVol+1,6], -df[:indVol+1,1], c='lightgrey', lw=.5)
      #if 'ang' in cont and round(angl*100)%10!=0:continue
      if 'ang' in cont and round(angl*100)%20!=0:continue
      if 'ang' in cont and angl>50/180: ax[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${angl:.1f}$", va='bottom', ha='center')
      #if 'rad' in cont and round(df[0,0]*10)%5!=0: continue 
      if 'rad' in cont and round(df[0,0]*10)%10!=5: continue 
      if 'rad' in cont and df[0,0]<3.2 and df[0,0]>.05: ax[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${df[0,0]:.1f}$", va='bottom', ha='center')
      if 'bub' in cont and df[0,5]>.5 and df[0,5]<=1: ax[axInd].text(df[indVol,6], -df[indVol,1]+.05, rf"${df[0,5]:.1f}$", va='center', ha='left')
      ax[axInd].plot(df[:indVol+1,6], -df[:indVol+1,1], c='k', zorder=3)
      if 'rad' in cont and round(df[0,0]*10)%10==0: continue 
      if 'ang' in cont and round(angl*100)%20!=0:continue
      #axRt = axProf[axInd].inset_axes((15/18.5, 2.6/3, (18.1-15)/18.5, .2/3))
      axRt = axProf[axInd].inset_axes((18/21.5, 2.6/3, (21.1-18)/21.5, .2/3))
      axRt.set_xscale('log')
      axRt.set_xlabel('$R_h/\\lambda$')
      axRt.set_xlim([.1,10])
      axRt.set_yticks([])
      axRt.tick_params(which='both', direction='in', top=True, right=True)
      for ri in range(21):
        Rt=10**( (ri-10)/10 )
        axRt.plot( (Rt,Rt), (0,1), lw=6, c=colRB(cont,Rt), zorder=-1)
      #axProf[axInd].plot( (5.5,5.5), (2.5,1.5), c='grey')
      #tri = RegularPolygon( (5.5,1.5), 3, radius=0.1, orientation=np.pi, color='grey', zorder=3)
      #axProf[axInd].add_patch(tri)
      #axProf[axInd].text(6, 2, '$g$', va='center', ha='left', c='grey')
      #if 'ang' in cont: spac=(.5,1.8,4.1,8,14.5,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160)[ round( 4-df[indVol,2]*5/np.pi) ]
      if 'ang' in cont: spac=(.5,2.3,5.1,9.5,17)[ round( 4-df[indVol,2]*5/np.pi) ]
      if 'rad' in cont: 
        #spac=(1,3.5,8,14.5,20,30,40,50)[ round(df[0,0]-.5) ] 
        spac=(1,4.5,10,17.5)[ round(df[0,0]-.5) ] 
        axProf[axInd].plot((spac-df[0,0], spac+df[0,0]), (0,0), c='w', clip_on=False, zorder=3)
        axProf[axInd].plot((spac-df[0,0], spac+df[0,0]), (-cav,-cav), c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac-df[0,0], spac-df[0,0]), (-cav,0), c='k', clip_on=False, zorder=3, lw=1)
        axProf[axInd].plot((spac+df[0,0], spac+df[0,0]), (-cav,0), c='k', clip_on=False, zorder=3, lw=1)
      #if spac>6 and spac<10: drawCoord=True
      if spac>6 and spac<11: drawCoord=True
      else: drawCoord=False
      for hei in reversed(range(5)):
        #if drawCoord and hei<4: continue
        heiInd = np.argmin( abs( (hei+1)*df[indVol,1]/5 - df[:indVol+1,1] ) )
        #AdamsBashforthProfile(1, df[heiInd,5], fname=folName+f'prof{hei:05}'+fname)
        with open(inFol+f'prof{hei:05}'+fname, encoding = 'utf-8') as f: prof = np.loadtxt(f)
        footInd = np.argmin( abs( df[heiInd,6] - prof[:,6] ))
        ax[axInd].plot(prof[footInd,6], -prof[footInd,1], 'o', ms=5, c=colRB( cont, df[heiInd,5] ), clip_on=False, zorder=4)#, mfc='None'
        xProf=np.concatenate(( -prof[:footInd,0][::-1] , prof[:footInd,0] ))
        xProf=xProf+spac
        yProf=np.concatenate(( prof[:footInd,1][::-1] - prof[footInd,1] , prof[:footInd,1] - prof[footInd,1] ))
        axProf[axInd].plot(xProf,yProf, c=colRB(cont,df[heiInd,5]), clip_on=False, zorder=4)
        #if hei!=4:continue
        #if 'rad' in cont: axProf[axInd].plot(xProf,yProf*0, c='w', clip_on=False)
        #if 'ang' in cont: continue
        #if round(df[0,0]-.5): continue
        if not drawCoord: continue
        if hei<4: continue
        xAn = xProf[-1]
        yAn = yProf[-1]
        #axProf[axInd].plot((xAn,xAn+.3), (yAn,yAn), color='grey')
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
        #axProf[axInd].plot( (xProf[h],xProf[h]), (0,yProf[h]), c='k', zorder=4)
        #axProf[axInd].text( xProf[h]-.1, yProf[h]/2, "$h$", ha='right', va='center', c='k') 
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
    ax2.plot(x,dfDet[:,5],c='b')
    ax2.plot(x,-dfDet[:,1],c='k')
    ax2.set_ylim([0,3.219])
    axV = axVV[0]
    axM = axVV[1]
    maxVind=np.argmax(dfDet[:,6])
    print('maxVol',dfDet[maxVind,:])
    maxVind=np.argmax(-dfDet[:,1])
    print('maxHeight',dfDet[maxVind,:])
    axM.tick_params(direction='in')
    ax[axInd].tick_params(which='both', direction='in', top=True, right=True)
    ax[axInd].set_xlabel('$V/\\lambda^3$')
    axM.text(5e-3,.99,'$\\mathrm{(b)}$',transform=axM.transAxes,va='top',ha='left')
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    ax[0].set_ylabel('$\\frac{h}{\\lambda}$',rotation=0,size=22,labelpad=10)
    axProf[axInd].tick_params(axis='y', which='both', direction='in', right=True)
    axProf[axInd].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axProf[axInd].set_ylabel('$\\frac{ z }{\\lambda}$',rotation=0,size=22,labelpad=15)
    axProf[axInd].set_ylim([0,3])
    #axProf[axInd].set_xlim([0,18.5])
    axProf[axInd].set_xlim([0,21.5])
    axProf[axInd].set_aspect('equal', adjustable='box')
    ax[axInd].set_ylim([0,3])
    ax[axInd].set_xlim([0,20])
    axV.set_ylim([0,30])
    axV.tick_params(which='both', direction='in', top=True, right=True)
    axM.tick_params(which='both', direction='in', top=True, right=True)
    figV.set_figwidth(6)
    fig2.set_figwidth(5)
    figV.set_figheight(6)
    fig2.set_figheight(3)
    if 'bub' in cont:
      axV.plot(x,dfDet[:,6],c='k',clip_on=False)
      axV.set_xlabel('$R_t$')
      ax2.plot(x,z, c='k',clip_on=False)
    if 'ang' in cont:
      axV.plot(x,dfDet[:,6],c='b',clip_on=False)
      axI = inset_axes(axV, width="40%", height="50%", loc='upper left')
      axI.yaxis.set_label_position("right")
      axI.yaxis.tick_right()
      axI.tick_params(which='both', direction='in', top=True, left=True, right=True, pad=6)
      axI.set_xscale('log')
      axI.set_yscale('log')
      axI.set_xlim([.07,1])
      axI.set_ylim([.01,100])
      axI.plot(x,dfDet[:,6],c='b')
      axM.plot( x[::30], z[::30], '.', c='b', clip_on=False, zorder=3)
      axM.plot( x, zM, '-', c='b', clip_on=False, zorder=3)
      fig2.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.98)
      #figProf.subplots_adjust(left=0.05, right=0.97, bottom=0.2, top=0.98)
      xx=np.linspace(0,1)
      axV.plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dashed', c='k')
      axI.plot(xx, 4*np.pi*(.0104*xx*180)**3/3, ls='dashed', c='k')
      axM.plot(xx, 3.219*xx**2, ls='dashed', c='k', zorder=3)
      #axM.plot(xx, .887*(np.pi*xx)**3 /2/np.pi/np.sin(np.pi*xx), ls='solid', c='k')
      #axM.plot(xx, np.sqrt( 6 * np.sin(np.pi*xx) * np.cos(np.pi*xx)**3 / (2 + np.sin(np.pi*xx) ) / (1 - np.sin(np.pi*xx) )**2 ), ls='dashed', c='k')
      print(4*np.pi*(.0104*180)**3/3, 'dotted')
      axM.set_xlabel('$\\phi_c/\\pi$')
      ax2.set_xlabel('$\\phi_c/\\pi$')
      ax2.text(-.08,.7,'$\\frac{h}{\\lambda}$',c='k',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.5,'$\\frac{R_t}{\\lambda}$',c='b',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.3,'$\\frac{r_0}{\\lambda}$',c='r',transform=ax2.transAxes,size=22,ha='center')
      axProf[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=axProf[axInd].transAxes,va='top',ha='left')
      ax[axInd].text(5e-3,.99,'$\\mathrm{(b)}$',transform=ax[axInd].transAxes,va='top',ha='left')
      ax2.set_xlim([0,1])
      ax2.plot(x,z, c='r',clip_on=False)
      axV.set_xlim([0,1])
      axV.text(1-5e-3,.99,'$\\mathrm{(a)}$',transform=axV.transAxes,va='top',ha='right')
      axV.set_ylabel('$\\frac{V_s}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      axM.set_ylabel('$\\frac{r_0}{\\lambda}$',size=22,rotation=0,labelpad=10)
      axM.set_ylim([0,4])
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
        axV.errorbar( mid, 4*np.pi/3 * rad**3 / capLen**3, xerr=[ [df[i,0]/180-mid], [mid-df[i,4]/180] ], fmt='^', c='b', mfc='None',clip_on=False, zorder=3)
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
        axV.plot(mid, vol, 'v', c='b', mfc='None', zorder=3)
        axV.plot([mn,mx], [vol,vol], c='b', zorder=3)
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
        axV.errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
        axI.errorbar(df[i,0]/180, df[i,3]/capLen**3, xerr=[ [ df[i,1]/180-df[i,0]/180 ] , [ df[i,0]/180-df[i,2]/180 ] ], fmt='d', c='b', mfc='None', clip_on=False, zorder=3)
      #ax[axInd].set_yticklabels([])
      #ax[axInd].set_ylabel('')
      #fig.subplots_adjust(left=0.03, right=0.86, bottom=0.2, top=0.98)
      rads = np.pi*np.arange(0.8, -0.1, -0.2)
      print('rads',rads/np.pi)
    if 'rad' in cont:
      axV.plot((3.832,*x),(0,*dfDet[:,6]),c='r',clip_on=False)
      from mpl_toolkits.axes_grid1.inset_locator import inset_axes
      axI = inset_axes(axV, width="40%", height="50%", loc='upper left')
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
      #axM.plot( x, z, c='r', clip_on=False, zorder=3)
      #axM.plot( (x[:-2]+x[1:-1]+x[2:])/3, (z[:-2]+z[1:-1]+z[2:])/3, c='r', clip_on=False, zorder=3)
      axM.plot( x[::15], z[::15], '.', c='r', clip_on=False, zorder=3)
      axM.plot( x, zM, c='r', clip_on=False, zorder=3)
      fig2.subplots_adjust(left=0.1, right=0.88, bottom=0.2, top=0.98)
      xx=np.linspace(0,4)
      axV.plot( xx, 2*np.pi*xx, linestyle='dashed', c='k')
      axI.plot( xx, 2*np.pi*xx, linestyle='dashed', c='k')
      axM.plot( xx, (xx/3.5)**.5, linestyle='dashed', c='k', zorder=3)
      axM.set_xlabel('$r_c/\\lambda$')
      #axI.set_xlabel('$r_c/\\lambda$',labelpad=-5)
      ax2.set_xlabel('$r_c/\\lambda$')
      axP = ax2.twinx()
      axP.tick_params(direction='in')
      ax2.tick_params(right=False)
      ax2.text(-.08,.4,'$\\frac{R_t}{\\lambda}$',c='b',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(-.08,.6,'$\\frac{h}{\\lambda}$',c='k',transform=ax2.transAxes,size=22,ha='center')
      ax2.text(1.12,.5,'$\\frac{\\phi_0}{\\pi}$',c='r',transform=ax2.transAxes,size=22,ha='center')
      axP.set_ylim([.5,1])
      axM.set_ylim([0,1.05])
      axM.set_yticks([0,.25,.5,.75,1])
      axV.axvspan(3.219, 4, color='lightgrey')
      axV.text(1-5e-3,.99,'$\\mathrm{(a)}$',transform=axV.transAxes,va='top',ha='right')
      axV.set_ylabel('$\\frac{V_p}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      #axI.set_ylabel('$\\frac{V_p}{\\lambda^3}$',size=22,rotation=0,labelpad=15)
      axM.axvspan(3.219, 4, color='lightgrey')
      axM.set_ylabel('$\\frac{\\phi_0}{\\pi}$',size=22,rotation=0,labelpad=10)
      axProf[axInd].text(5e-3,.99,'$\\mathrm{(a)}$',transform=axProf[axInd].transAxes,va='top',ha='left')
      ax[axInd].text(5e-3,.99,'$\\mathrm{(a)}$',transform=ax[axInd].transAxes,va='top',ha='left')
      ax2.set_xlim([0,4])
      axP.plot( x, z, c='r',clip_on=False, zorder=3)#,'.',ms=5
      fname = 'exptData/LesageVolVsContRadSq.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      for i in range(len(df[:,0])):
        if df[i,2]>1:continue
        axV.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', clip_on=False, zorder=3)
        axI.plot(df[i,0]**.5, df[i,1]*df[i,0]**1.5, 's', mec='r', mfc='None', zorder=3)
      fname = 'exptData/MoriVolByContCubeVsContSqByCapSq.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      axV.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', clip_on=False, zorder=3)
      axI.plot(.5/df[:,0]**.5, df[:,1]/df[:,0]**1.5, 'd', mec='r', mfc='None', zorder=3)
      fname = 'exptData/sasetty23stability.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f)
      axV.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None', clip_on=False)
      axI.plot(df[:,2]/df[:,3]/2, df[:,1]/(df[:,3]*1e-3)**3, 'v', mec='r', mfc='None')
      fname = 'exptData/gunde01measurement.txt'
      print('open',fname)
      with open(fname) as f: df = np.loadtxt(f, skiprows=2)
      capLen=(df[:,2]*1e-3/df[:,1]/9.81)**.5
      axV.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None', clip_on=False)
      axI.plot(df[:,0]*1e-3/capLen, df[:,4]*1e-6*1e-3/capLen**3, '^', mec='r', mfc='None')
      axV.set_xlim([0,4])
      #fig.subplots_adjust( left=0.14, right=0.97, bottom=0.2, top=0.98)
    fname = outFol+'MaxVolVs_'+cont+'.pdf'
    print('savin ',fname)
    figV.savefig(fname, transparent=True, format='pdf', bbox_inches='tight', pad_inches=0)
    fname = outFol+'ax2_'+cont+'.pdf'
    print('savin ',fname)
    fig2.savefig(fname, transparent=True, format='pdf')
  fname = outFol+'heightVsVol_'+cont+'.pdf'
  print('savin ',fname)
  fig.set_figwidth(10)
  fig.set_figheight(3)
  fig.tight_layout(pad=.7)
  fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
  figProf.set_figwidth(12)
  #axProf[1].set_xlabel('$x/\\lambda$',labelpad=-5)
  figProf.subplots_adjust(hspace=-.05)
  outName = outFol+f'pin.pdf'
  print('savin ',outName)
  figProf.savefig(outName, transparent=True, bbox_inches='tight', pad_inches=0)
  return

def plot_graphical_abstract(nam='rad ang'): 
  import os
  from bubble import AdamsBashforthProfile
  simFol = '../mergeDdgc/data/'
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
