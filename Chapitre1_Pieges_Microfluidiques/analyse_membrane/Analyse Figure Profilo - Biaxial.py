"""
=============================================================================
ANALYSE BIAXIALE — GRAND AXE + PETIT AXE — MEMBRANE PDMS ELLIPTIQUE
=============================================================================
Modèle : PLAQUE ELLIPTIQUE 2D (Kirchhoff, Timoshenko 1959)
  Déformée : w(x,y) = w0 · (1 − x²/a² − y²/b²)²
  Flèche centrale par unité de pression :
      w0/P = 1/(8D) · 1/(3/a⁴ + 3/b⁴ + 2/(a²b²)),  D = E·t³/[12(1−ν²)]

Le modèle 2D prédit UNE flèche centrale w0 (sommet unique), commune aux
deux axes. On fitte donc un SEUL module effectif E_eff sur les données des
DEUX axes réunies, avec e0 FIXÉ au gap physique et géométrie (a,b,t,ν) FIXÉE.

Projection le long d'un axe (profil 1D mesuré) :
  - Grand axe (y=0) :  w(s) = w0 · (1 − (s−s0)²/a²)²   → biparabolique, demi-long. a
  - Petit axe (x=0) :  w(s) = w0 · (1 − (s−s0)²/b²)²   → biparabolique, demi-long. b

Figures de synthèse (SVG) :
  1. w_max vs ΔP — les deux axes + modèle elliptique commun (E_eff, e0)
  2. Profils de déformation (fit biparabolique par pression)
  3. Surface 3D reconstruite (géométrie géométrique a,b fixée)

Export Excel : profils par pression + paramètres de fit + surface 3D.

UTILISATION (Pyzo) :
  F5 → dossier grand axe → ligne grand axe
     → dossier petit axe → ligne petit axe
     → figures de synthèse affichées
=============================================================================
"""

import os, re, csv, warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, minimize_scalar as _ms
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, RectBivariateSpline
from skimage import io, color
import tkinter as tk
from tkinter import filedialog
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION  <<<  MODIFIEZ CES PARAMÈTRES AVANT DE LANCER  >>>
# =============================================================================
CFG = {
    'scale_um_px'         : 2.75,    # µm/pixel
    'lambda_nm'           : 661.0,   # nm
    'a_grand_um'          : 3750.0,  # µm demi-grand axe (géométrique, FIXÉ)
    'b_petit_um'          : 2800.0,  # µm demi-petit axe (géométrique, FIXÉ)
    't_fixed_um'          : 1400.0,  # µm épaisseur membrane (FIXÉE)
    'E_pa'                : 1.3e6,   # Pa module de Young nominal
    'E_uncertainty_pa'    : 0.3e6,   # Pa incertitude E
    'nu'                  : 0.5,     # Poisson PDMS
    'profile_width'       : 5,       # px demi-largeur bande
    'peak_dist'           : 2,       # px distance min entre pics
    'gauss_sigma'         : 12.0,    # px sigma lissage interfrange
    'e0_um'               : 12.0,    # µm gap initial verre-PDMS (FIXÉ)
    'n_3d_points'         : 80,      # points par axe pour la surface 3D
    'output_dir'          : None,    # None = dossier grand axe
}

# Constantes globales
LAMBDA_NM = 661.0; SCALE_UM_PX = 2.75; PEAK_DIST_PX = 2
GAUSS_SIGMA = 12.0; SG_WINDOW = 15; SG_ORDER = 3
A_UM = 3750.0; B_UM = 2800.0; T_UM = 1400.0
E_PA = 1.3e6; E_UNC = 0.3e6; NU = 0.5
E0_UM = 12.0


# =============================================================================
# UTILITAIRES — CHARGEMENT ET PROFIL
# =============================================================================
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def load_images(folder):
    exts = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp')
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)],
                   key=natural_sort_key)
    if not files: raise FileNotFoundError(f"Pas d'images dans {folder}")
    imgs = []
    for f in files:
        img = io.imread(os.path.join(folder, f))
        if img.ndim == 3: img = color.rgb2gray(img)
        imgs.append(img.astype(np.float32))
    print(f"  {len(imgs)} image(s) : {', '.join(files)}")
    return imgs, files

def extract_profile(image, p1, p2, width=5):
    r0,c0=p1; r1,c1=p2
    L = int(np.hypot(r1-r0, c1-c0))
    if L==0: raise ValueError("Points identiques")
    dr,dc=(r1-r0)/L,(c1-c0)/L; dr_p,dc_p=-dc,dr
    H,W=image.shape
    acc=np.zeros(L+1); cnt=np.zeros(L+1,int)
    for off in range(-width,width+1):
        for i in range(L+1):
            r=int(round(r0+i*dr+off*dr_p)); c=int(round(c0+i*dc+off*dc_p))
            if 0<=r<H and 0<=c<W: acc[i]+=image[r,c]; cnt[i]+=1
    mask=cnt>0
    prof=np.where(mask, acc/np.where(mask,cnt,1), 0.0)
    mn,mx=prof.min(),prof.max()
    if mx>mn: prof=(prof-mn)/(mx-mn)
    s_px=np.arange(L+1,dtype=float)
    return s_px, s_px*SCALE_UM_PX, prof

def parse_pressure(filenames):
    dP=[]
    for f in filenames:
        m=re.search(r'(-?\d+(?:[.,]\d+)?)', os.path.splitext(f)[0])
        dP.append(abs(float(m.group(1).replace(',','.'))) if m else np.nan)
    return np.array(dP)


# =============================================================================
# DÉTECTION DES FRANGES
# =============================================================================
def sg_smooth(arr):
    w=min(SG_WINDOW,len(arr)-1)
    if w%2==0: w-=1
    if w<SG_ORDER+2: return arr.copy()
    return savgol_filter(arr,w,SG_ORDER)

def detect_fringes(s_px, prof):
    sm=sg_smooth(prof)
    amp=sm.max()-sm.min()
    if amp<1e-6: return np.array([],int),np.array([]),np.array([],int),sm
    d=np.diff(sm.astype(float)); sgn=np.sign(d)
    for i in range(1,len(sgn)):
        if sgn[i]==0: sgn[i]=sgn[i-1]
    if len(sgn)>0 and sgn[0]==0: sgn[0]=1
    sc=np.diff(sgn); chg=np.where(sc!=0)[0]
    maxs,mins=[],[]
    for i in chg:
        cands=[c for c in range(i,min(i+3,len(sm)))]
        if sc[i]<0: maxs.append(max(cands,key=lambda c:sm[c]))
        else:       mins.append(min(cands,key=lambda c:sm[c]))
    def dedup(idx,mode):
        if not idx: return np.array([],int)
        idx=sorted(set(idx)); out=[idx[0]]
        for x in idx[1:]:
            if x-out[-1]<PEAK_DIST_PX:
                if mode=='max' and sm[x]>sm[out[-1]]: out[-1]=x
                elif mode=='min' and sm[x]<sm[out[-1]]: out[-1]=x
            else: out.append(x)
        return np.array(out,int)
    pk=dedup(maxs,'max'); vl=dedup(mins,'min')
    events=[(p,'max') for p in pk]+[(v,'min') for v in vl]
    events.sort(key=lambda x:x[0])
    if not events: return np.array([],int),np.array([]),np.array([],int),sm
    clean=[events[0]]
    for pos,typ in events[1:]:
        pp,pt=clean[-1]
        if typ==pt:
            if typ=='max' and sm[pos]>sm[pp]: clean[-1]=(pos,typ)
            elif typ=='min' and sm[pos]<sm[pp]: clean[-1]=(pos,typ)
        else: clean.append((pos,typ))
    pk_f=np.array([p for p,t in clean if t=='max'],int)
    vl_f=np.array([p for p,t in clean if t=='min'],int)
    return pk_f, s_px[pk_f].astype(float), vl_f, sm


# =============================================================================
# MÉTHODE D — INTÉGRATION DE PENTE
# =============================================================================
def method_D(s_px, s_um, peaks_pos_px):
    lam=LAMBDA_NM/1000.0
    if len(peaks_pos_px)<2: return None,None,None
    pos=np.sort(peaks_pos_px.astype(float))
    ifr=np.diff(pos); mid=0.5*(pos[:-1]+pos[1:])
    f=interp1d(mid,ifr,kind='linear',fill_value=(ifr[0],ifr[-1]),bounds_error=False)
    Lum=np.clip(gaussian_filter1d(np.clip(f(s_px),1,None),sigma=GAUSS_SIGMA),1,None)*SCALE_UM_PX
    theta=lam/(2*Lum)
    w=np.cumsum(theta*np.gradient(s_um)); w-=w.min()
    return s_um, w, Lum


# =============================================================================
# MODÈLE ELLIPTIQUE — PROFIL ET FLÈCHE CENTRALE
# =============================================================================
def elliptic_profile(s_um, w_max, s0, half_len):
    """Profil biparabolique projeté : w(s)=w_max·(1−((s−s0)/half_len)²)²"""
    u=(s_um-s0)/half_len
    return np.where(np.abs(u)<=1, w_max*(1-u**2)**2, 0.0)

def elliptic_K(a_um, b_um, t_um, E_pa, nu):
    """
    Pente K = w0/ΔP  [µm/Pa] du modèle plaque elliptique :
      w0 = P/(8D)·1/(3/a⁴+3/b⁴+2/(a²b²)),  D = E·t³/[12(1−ν²)]
    """
    a=a_um*1e-6; b=b_um*1e-6; t=t_um*1e-6
    D=E_pa*t**3/(12*(1-nu**2))
    denom=3/a**4+3/b**4+2/(a**2*b**2)
    return (1.0/(8*D)/denom)*1e6


# =============================================================================
# FIT DU PROFIL — w_max et s0 libres, half_len fixé
# =============================================================================
def fit_profile(s_um, w_D, half_len):
    """Fit biparabolique. Retourne (w_max, s0, r2, s_model, w_model)."""
    if w_D is None or len(w_D)<5: return None,None,None,None,None
    w0=float(np.max(w_D)); s_c=float(s_um[np.argmax(w_D)])
    if w0<=0: return None,None,None,None,None
    def res(s0): return np.sum((w_D-elliptic_profile(s_um,w0,s0,half_len))**2)
    opt=_ms(res,bounds=(s_c-half_len,s_c+half_len),method='bounded')
    s0_i=opt.x
    try:
        popt,_=curve_fit(
            lambda s,wm,s0: elliptic_profile(s,wm,s0,half_len),
            s_um,w_D,p0=[w0,s0_i],
            bounds=([0,s_c-2*half_len],[w0*8,s_c+2*half_len]),maxfev=12000)
        wm,s0f=float(popt[0]),float(popt[1])
    except: wm,s0f=w0,s0_i
    wp=elliptic_profile(s_um,wm,s0f,half_len)
    ss_res=np.sum((w_D-wp)**2); ss_tot=np.sum((w_D-w_D.mean())**2)
    r2=1-ss_res/ss_tot if ss_tot>0 else 0
    s_mod=np.linspace(s0f-half_len,s0f+half_len,800)
    w_mod=elliptic_profile(s_mod,wm,s0f,half_len)
    return wm,s0f,r2,s_mod,w_mod


# =============================================================================
# FIT GLOBAL ELLIPTIQUE — E_eff COMMUN aux deux axes, e0 FIXÉ
# =============================================================================
def fit_Eeff_common(res_G, res_P):
    """
    Fitte UN SEUL module effectif E_eff sur les w_max des DEUX axes réunis.
    Modèle : w_max = e0(FIXÉ) + K_ellip(a,b,t,E_eff,ν)·ΔP
    Le modèle 2D prédit la même flèche centrale w0 sur les deux axes,
    donc un E_eff commun est physiquement justifié.

    Retourne (E_eff, r2, e0).
    """
    dP=[]; wm=[]
    for r in res_G + res_P:
        if r['wm_fit'] is not None and np.isfinite(r['dP_mbar']) and r['wm_fit']>0:
            dP.append(r['dP_mbar']*100)   # Pa
            wm.append(r['wm_fit'])
    dP=np.array(dP); wm=np.array(wm)
    if len(dP)<2: return None,None,E0_UM

    e0_fixed=E0_UM
    def model(dP_pa, E_eff):
        return e0_fixed + elliptic_K(A_UM,B_UM,T_UM,E_eff,NU)*dP_pa

    try:
        popt,_=curve_fit(model, dP, wm, p0=[E_PA],
                         bounds=([1e5],[1e8]), maxfev=20000)
        Eeff=float(popt[0])
    except Exception as e:
        print(f"  [Fit E_eff] {e}"); return None,None,e0_fixed

    w_pred=model(dP,Eeff)
    ss_res=np.sum((wm-w_pred)**2); ss_tot=np.sum((wm-wm.mean())**2)
    r2=1-ss_res/ss_tot if ss_tot>0 else 0

    print(f"  E_eff commun = {Eeff/1e6:.4f} MPa  (nominal {E_PA/1e6:.2f} MPa)  "
          f"e0={e0_fixed:.0f}µm FIXÉ  R²={r2:.4f}")
    return Eeff, r2, e0_fixed


# =============================================================================
# SÉLECTION INTERACTIVE
# =============================================================================
class LineSelector:
    def __init__(self, image, title=""):
        self.p1=self.p2=None; self._s1=self._s2=self._lp=None
        self.fig,self.ax=plt.subplots(figsize=(13,8))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.imshow(image,cmap='gray',vmin=0,vmax=1)
        self.ax.set_title("Clic GAUCHE=départ  |  Clic DROIT=arrivée  |  'Valider'",fontsize=10)
        self.fig.canvas.mpl_connect('button_press_event',self._click)
        ab=self.fig.add_axes([0.38,0.01,0.24,0.045])
        self.btn=Button(ab,'Valider',color='#2ecc71',hovercolor='#27ae60')
        self.btn.on_clicked(lambda _:plt.close(self.fig))
        plt.tight_layout(rect=[0,0.07,1,1]); plt.show()
    def _click(self,ev):
        if ev.inaxes!=self.ax or ev.xdata is None: return
        col,row=int(ev.xdata),int(ev.ydata)
        if ev.button==1:
            self.p1=(row,col)
            if self._s1: self._s1.remove()
            self._s1=self.ax.scatter(col,row,c='lime',s=100,zorder=6,marker='D')
        elif ev.button==3:
            self.p2=(row,col)
            if self._s2: self._s2.remove()
            self._s2=self.ax.scatter(col,row,c='red',s=90,zorder=6)
        if self.p1 and self.p2:
            if self._lp: self._lp[0].remove()
            self._lp=self.ax.plot([self.p1[1],self.p2[1]],[self.p1[0],self.p2[0]],
                                   'y--',lw=1.8,zorder=5)
        self.fig.canvas.draw()
    def get_points(self):
        if self.p1 is None or self.p2 is None: raise ValueError("Deux points requis")
        return self.p1,self.p2


# =============================================================================
# ANALYSE D'UN AXE — sauvegarde silencieuse des figures individuelles (SVG)
# =============================================================================
def analyse_axis(folder, axis_name, half_len_geom, out_dir):
    print(f"\n{'='*65}")
    print(f"  ANALYSE — {axis_name.upper()}  (demi-longueur géom. = {half_len_geom:.0f} µm)")
    print(f"{'='*65}")
    imgs,fnames=load_images(folder)
    dP_mbar=parse_pressure(fnames)
    print(f"  Pressions : {[f'{p:.0f}' for p in dP_mbar]} mbar")

    print(f"\n  Tracer la ligne sur la dernière image ({fnames[-1]}) ...")
    sel=LineSelector(imgs[-1],title=f"Ligne {axis_name} — {fnames[-1]}")
    p1,p2=sel.get_points()
    L_um=np.hypot(p2[0]-p1[0],p2[1]-p1[1])*SCALE_UM_PX
    print(f"  Longueur ligne = {L_um:.0f} µm")

    results=[]
    for idx,(img,fname) in enumerate(zip(imgs,fnames)):
        label=os.path.splitext(fname)[0]
        s_px,s_um,prof=extract_profile(img,p1,p2,width=CFG['profile_width'])
        pk,real_pos,vl,sm=detect_fringes(s_px,prof)
        s_D,w_D,Lum=method_D(s_px,s_um,real_pos)
        wm,s0f,r2,s_mod,w_mod=fit_profile(s_um,w_D,half_len_geom)

        results.append({'label':label,'dP_mbar':dP_mbar[idx],'dP_pa':dP_mbar[idx]*100,
                        's_um':s_um,'prof':prof,'sm':sm,'pk':pk,'vl':vl,
                        'real_pos':real_pos,'s_D':s_D,'w_D':w_D,'Lum':Lum,
                        'wm_fit':wm,'s0_fit':s0f,'r2':r2,'s_mod':s_mod,'w_mod':w_mod,
                        'n_peaks':len(pk)})

        # Figure individuelle SVG (mesure portion + modèle prolongé)
        fig,axes=plt.subplots(3,1,figsize=(13,12))
        fig.suptitle(f'{axis_name} — {label}  (ΔP={dP_mbar[idx]:.0f} mbar)',
                     fontsize=13,fontweight='bold')
        ax=axes[0]
        ax.plot(s_um,prof,color='#bdc3c7',lw=0.8,alpha=0.7,label='Brut')
        ax.plot(s_um,sm,color='#2c3e50',lw=1.4,label='Lissé S-G')
        if len(pk)>0: ax.plot(s_um[pk],sm[pk],'v',color='#27ae60',ms=5,zorder=6,
                               label=f'Maxima ({len(pk)})')
        if len(vl)>0: ax.plot(s_um[vl],sm[vl],'^',color='#3498db',ms=4,zorder=5,
                               label=f'Minima ({len(vl)})')
        ax.set_ylabel('Intensité norm.'); ax.legend(fontsize=7,ncol=2)
        ax.set_xlim(s_um[0],s_um[-1]); ax.grid(True,alpha=0.25)
        # Déformation : mesure (portion) + modèle elliptique prolongé (axe entier)
        ax2=axes[1]
        if s_D is not None:
            if w_mod is not None:
                ax2.plot(s_mod,w_mod,'--',color='#8e44ad',lw=1.8,alpha=0.9,
                         label=f'Modèle elliptique prolongé (2L={2*half_len_geom:.0f}µm)  '
                               f'w_max={wm:.2f}µm  R²={r2:.3f}')
                ax2.axvline(s0f,color='#8e44ad',lw=0.8,ls=':',alpha=0.5)
                ax2.axvline(s0f-half_len_geom,color='#bbbbbb',lw=0.8,alpha=0.6)
                ax2.axvline(s0f+half_len_geom,color='#bbbbbb',lw=0.8,alpha=0.6)
            ax2.plot(s_D,w_D,color='#27ae60',lw=2.8,zorder=6,
                     label='Mesure (portion analysée)')
            ax2.axvspan(s_D[0],s_D[-1],color='#27ae60',alpha=0.06,zorder=0)
        ax2.set_xlabel('s (µm)'); ax2.set_ylabel('w (µm)'); ax2.legend(fontsize=7)
        if w_mod is not None: ax2.set_xlim(s_mod[0],s_mod[-1])
        elif s_D is not None: ax2.set_xlim(s_D[0],s_D[-1])
        ax2.set_ylim(bottom=0); ax2.grid(True,alpha=0.25)
        # Interfrange
        ax3=axes[2]
        if Lum is not None:
            ax3.plot(s_um,Lum,color='#8e44ad',lw=1.5,label='Λ(s) lissé')
            if len(real_pos)>1:
                rp=np.sort(real_pos)
                ax3.plot(0.5*(rp[:-1]+rp[1:])*SCALE_UM_PX,
                         np.diff(rp)*SCALE_UM_PX,'o',color='#27ae60',ms=3,alpha=0.6,
                         label='Interfranges mesurés')
        ax3.set_xlabel('s (µm)'); ax3.set_ylabel('Λ (µm)')
        ax3.set_ylim(0,150); ax3.grid(True,alpha=0.25); ax3.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir,f'fringe_{axis_name.replace(" ","_")}_{label}.svg'),
                    format='svg',bbox_inches='tight')
        plt.close(fig)
        print(f"    [{idx+1}] {label:20s}  n_max={len(pk):3d}  "
              f"wm={wm:.2f}µm  R²={r2:.3f}" if wm else
              f"    [{idx+1}] {label:20s}  n_max={len(pk):3d}  wm=N/A")

    return results


# =============================================================================
# FIGURE 1 — w_max vs ΔP : deux axes + modèle elliptique commun (SVG)
# =============================================================================
def fig_wmax_vs_dP(res_G, res_P, Eeff, r2_glob, e0, out_dir):
    dP_range_mbar=np.linspace(0,200,300)
    dP_range_pa  =dP_range_mbar*100

    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=False)
    fig.suptitle(
        f'Déflexion w_max vs Dépression — Modèle plaque elliptique 2D\n'
        f'a={A_UM:.0f}µm  b={B_UM:.0f}µm  t={T_UM:.0f}µm  ν={NU}  '
        f'e₀={e0:.0f}µm (fixé)  E_eff={Eeff/1e6:.3f}MPa (commun)  R²={r2_glob:.4f}',
        fontsize=11,fontweight='bold')

    # Courbe elliptique COMMUNE (même w0 pour les deux axes)
    K_c  = elliptic_K(A_UM,B_UM,T_UM,Eeff,NU)
    K_hi = elliptic_K(A_UM,B_UM,T_UM,max(Eeff-E_UNC,1e5),NU)
    K_lo = elliptic_K(A_UM,B_UM,T_UM,Eeff+E_UNC,NU)
    z_c  = e0 + K_c *dP_range_pa
    z_hi = e0 + K_hi*dP_range_pa
    z_lo = e0 + K_lo*dP_range_pa

    for ax,res,axis_name,cmap_,mk,col_dark in zip(
            axes,[res_G,res_P],['Grand axe','Petit axe'],
            [cm.Blues,cm.Reds],['o','s'],['#1a5276','#7b241c']):

        # Modèle elliptique commun (identique sur les deux panneaux)
        ax.plot(dP_range_mbar,z_c,'-',color='#117733',lw=2.5,
                label=f'Modèle elliptique commun\nE_eff={Eeff/1e6:.3f}MPa  R²={r2_glob:.4f}')
        ax.fill_between(dP_range_mbar,z_lo,z_hi,color='#117733',alpha=0.13,
                        label=f'Bande E_eff±{E_UNC/1e6:.1f}MPa')
        ax.axhline(e0,color='#117733',lw=0.8,ls='--',alpha=0.5)
        ax.text(2,e0+1.5,f'e₀={e0:.0f}µm (fixé)',fontsize=7,color='#117733',va='bottom')

        # Points expérimentaux de cet axe
        n=len(res)
        colors=[cmap_(0.4+0.5*i/max(n-1,1)) for i in range(n)]
        for i,(r,col) in enumerate(zip(res,colors)):
            if r['wm_fit'] is not None and np.isfinite(r['dP_mbar']):
                w_abs=r['wm_fit']+e0
                ax.scatter(r['dP_mbar'],w_abs,color=col,s=85,zorder=7,
                           edgecolors=col_dark,lw=1.0,marker=mk)

        ax.set_title(axis_name,fontsize=11,fontweight='bold')
        ax.set_xlabel('Dépression ΔP (mbar)',fontsize=10)
        ax.set_ylabel(f'Distance verre→PDMS (µm)  [e₀={e0:.0f}µm à P=0]',fontsize=10)
        ax.legend(fontsize=8,loc='upper left')
        ax.grid(True,alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'synthese_wmax_vs_dP.svg'),
                format='svg',bbox_inches='tight')
    plt.show(); plt.close()

    # Affichage terminal : E par image
    print(f"\n  E par image (elliptique inversé, géométrie fixée) :")
    K_unit=elliptic_K(A_UM,B_UM,T_UM,1.0,NU)   # K(E)=K(1)/E
    for name,res in [('Grand axe',res_G),('Petit axe',res_P)]:
        print(f"  ── {name} ──")
        for r in res:
            if r['wm_fit'] and r['dP_pa']>0:
                E_i=K_unit*r['dP_pa']/r['wm_fit']
                print(f"    {r['label']:16s} ΔP={r['dP_mbar']:.0f}mbar  "
                      f"w_max={r['wm_fit']:.2f}µm  E={E_i/1e6:.3f}MPa")


# =============================================================================
# FIGURE 2 — Profils superposés (fit biparabolique) (SVG)
# =============================================================================
def fig_profiles(res_G, res_P, out_dir):
    cmap=cm.plasma
    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    fig.suptitle('Profils de déformation — Fit biparabolique elliptique (pleine membrane)',
                 fontsize=13,fontweight='bold')

    for ax,res,half_geom,axis_name in zip(
            axes,[res_G,res_P],[A_UM,B_UM],['Grand axe','Petit axe']):
        n=len(res)
        for i,r in enumerate(res):
            col=cmap(0.1+0.8*i/max(n-1,1))
            lbl=f"{r['label']} ({r['dP_mbar']:.0f}mbar)"
            if r['wm_fit'] is not None and r['s0_fit'] is not None:
                s_full=np.linspace(r['s0_fit']-half_geom,r['s0_fit']+half_geom,600)
                w_full=elliptic_profile(s_full,r['wm_fit'],r['s0_fit'],half_geom)
                ax.plot(s_full,w_full,'-',color=col,lw=2.0,
                        label=f"{lbl}  w_max={r['wm_fit']:.1f}µm",zorder=5)
        ax.set_title(f'{axis_name}  (demi-long. {half_geom:.0f}µm)',fontsize=10)
        ax.set_xlabel('Position s (µm)',fontsize=10)
        ax.set_ylabel('Déformation w (µm)',fontsize=10)
        ax.grid(True,alpha=0.25); ax.set_ylim(bottom=0)
        ax.legend(fontsize=7,loc='upper left',ncol=1)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'synthese_profiles.svg'),
                format='svg',bbox_inches='tight')
    plt.show(); plt.close()


# =============================================================================
# FIGURE 3 — SURFACE 3D (géométrie a,b géométriques fixées) (SVG)
# =============================================================================
def fig_3d_surface(res_G, res_P, out_dir):
    """
    Nappe 3D à la pression max, modèle elliptique 2D vrai :
      w(x,y) = w0·(1 − x²/a² − y²/b²)²   (a,b géométriques fixés)
    z absolu = e0 + w(x,y).
    """
    n=CFG['n_3d_points']
    half_G=A_UM; half_P=B_UM   # géométrie fixée

    dPs_G={r['dP_mbar']:r for r in res_G
           if r['wm_fit'] is not None and np.isfinite(r['dP_mbar'])}
    dPs_P={r['dP_mbar']:r for r in res_P
           if r['wm_fit'] is not None and np.isfinite(r['dP_mbar'])}
    common_dP=sorted(set(dPs_G.keys())&set(dPs_P.keys()))
    if not common_dP: common_dP=sorted(dPs_G.keys())
    if not common_dP:
        print("  [3D] Aucune donnée."); return [],None,None,[]

    dp_max=max(common_dP)
    wm_g=dPs_G[dp_max]['wm_fit']
    wm_p=dPs_P.get(dp_max,{}).get('wm_fit',wm_g) if dp_max in dPs_P else wm_g
    wm_max=(wm_g+wm_p)/2

    x=np.linspace(-half_G,half_G,n); y=np.linspace(-half_P,half_P,n)
    X,Y=np.meshgrid(x,y)
    u=(X/half_G)**2+(Y/half_P)**2
    inside=u<=1.0
    W_rel=np.where(inside, wm_max*(1-u)**2, np.nan)   # vrai modèle elliptique 2D
    W_abs=W_rel+E0_UM
    W_rest=np.where(inside,E0_UM,np.nan)

    theta=np.linspace(0,2*np.pi,300)
    xe=half_G*np.cos(theta)/1000; ye=half_P*np.sin(theta)/1000

    fig=plt.figure(figsize=(13,9))
    ax3d=fig.add_subplot(111,projection='3d')
    fig.suptitle(
        f'Surface 3D elliptique — Pression max ΔP={dp_max:.0f}mbar\n'
        f'w(x,y)=w0(1−x²/a²−y²/b²)²  a={half_G:.0f}µm b={half_P:.0f}µm  '
        f'e₀={E0_UM:.0f}µm  w_max={wm_max:.1f}µm',
        fontsize=11,fontweight='bold')

    surf=ax3d.plot_surface(X/1000,Y/1000,W_abs,cmap='coolwarm',alpha=0.85,
                           linewidth=0,antialiased=True,
                           vmin=E0_UM,vmax=E0_UM+wm_max)
    fig.colorbar(surf,ax=ax3d,shrink=0.5,aspect=12,
                 label='Distance verre→PDMS (µm)',pad=0.1)
    ax3d.plot_surface(X/1000,Y/1000,W_rest,color='#aab4c8',alpha=0.25,
                      linewidth=0,antialiased=False)
    ax3d.plot(xe,ye,np.zeros_like(xe),'k-',lw=1.5,alpha=0.6,
              label=f'Ellipse géom. a={half_G:.0f}µm b={half_P:.0f}µm')
    ax3d.plot(xe,ye,np.full_like(xe,E0_UM),'--',color='#2980b9',lw=1.5,alpha=0.8,
              label=f'PDMS@P=0 (z=e₀={E0_UM:.0f}µm)')
    ax3d.scatter([0],[0],[E0_UM+wm_max],color='#e74c3c',s=80,zorder=10,
                 label=f'Centre z={E0_UM+wm_max:.1f}µm')
    ax3d.plot([0,0],[0,0],[E0_UM,E0_UM+wm_max],'r-',lw=2.0,alpha=0.8)
    mid=n//2
    ax3d.plot(X[mid,:]/1000,Y[mid,:]/1000,W_abs[mid,:],'k-',lw=1.5,alpha=0.9,zorder=5)
    ax3d.plot(X[:,mid]/1000,Y[:,mid]/1000,W_abs[:,mid],'k-',lw=1.5,alpha=0.9,zorder=5)
    ax3d.set_xlabel('x — Grand axe (mm)',fontsize=9,labelpad=8)
    ax3d.set_ylabel('y — Petit axe (mm)',fontsize=9,labelpad=8)
    ax3d.set_zlabel('z : dist. verre→PDMS (µm)',fontsize=9,labelpad=8)
    ax3d.set_xlim(-half_G/1000,half_G/1000); ax3d.set_ylim(-half_P/1000,half_P/1000)
    ax3d.set_zlim(0,E0_UM+wm_max*1.1)
    ax3d.legend(fontsize=8,loc='upper left')
    ax3d.view_init(elev=25,azim=-60)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'surface_3D.svg'),format='svg',bbox_inches='tight')
    plt.show(); plt.close()

    surfaces=[(dp_max,W_rel,'#e74c3c')]
    return surfaces,X,Y,common_dP


# =============================================================================
# EXPORT EXCEL
# =============================================================================
def export_excel(res_G, res_P, Eeff, r2_glob, e0, out_dir):
    wb=Workbook()
    def hdr(ws,r,c,txt,fill):
        cell=ws.cell(r,c,txt)
        cell.font=Font(name='Arial',bold=True,color='FFFFFF',size=10)
        cell.fill=PatternFill('solid',fgColor=fill)
        cell.alignment=Alignment(horizontal='center')
        cell.border=Border(left=Side(style='thin'),right=Side(style='thin'),
                           top=Side(style='thin'),bottom=Side(style='thin'))
    def dat(ws,r,c,v):
        cell=ws.cell(r,c,v)
        if isinstance(v,float): cell.number_format='0.0000'
        cell.alignment=Alignment(horizontal='center')

    # Feuille 1 : Paramètres + fit elliptique
    ws1=wb.active; ws1.title='Paramètres et fit'
    K_c=elliptic_K(A_UM,B_UM,T_UM,Eeff,NU) if Eeff else 0
    params=[('λ (nm)',LAMBDA_NM),('Échelle (µm/px)',SCALE_UM_PX),
            ('a géom. (µm)',A_UM),('b géom. (µm)',B_UM),
            ('t (µm)',T_UM),('E nominal (Pa)',E_PA),('σE (Pa)',E_UNC),('ν',NU),
            ('e0 fixé (µm)',e0),
            ('E_eff commun (Pa)',Eeff or 'N/A'),
            ('E_eff commun (MPa)',Eeff/1e6 if Eeff else 'N/A'),
            ('K elliptique (µm/mbar)',K_c*100 if Eeff else 'N/A'),
            ('R² fit global',r2_glob or 'N/A')]
    hdr(ws1,1,1,'Paramètre','1D6A2E'); hdr(ws1,1,2,'Valeur','1D6A2E')
    for i,(k,v) in enumerate(params,2):
        ws1.cell(i,1,k); dat(ws1,i,2,v)
    ws1.column_dimensions['A'].width=35; ws1.column_dimensions['B'].width=18

    # Feuille 2 : Résultats expérimentaux
    ws2=wb.create_sheet('Résultats exp.')
    K_unit=elliptic_K(A_UM,B_UM,T_UM,1.0,NU)
    row=1
    for axis_name,res,fill,half_geom in [
            ('GRAND AXE',res_G,'1F4E79',A_UM),
            ('PETIT AXE',res_P,'7B0000',B_UM)]:
        ws2.merge_cells(start_row=row,start_column=1,end_row=row,end_column=8)
        hdr(ws2,row,1,f'{axis_name} — demi-long. {half_geom:.0f}µm',fill); row+=1
        for j,h in enumerate(['Image','ΔP(mbar)','ΔP(Pa)','N_max',
                               'w_max_fit(µm)','s0(µm)','R²','E_image(MPa)'],1):
            hdr(ws2,row,j,h,fill)
        row+=1
        for r in res:
            dp=r['dP_pa']
            E_i=K_unit*dp/r['wm_fit']/1e6 if (r['wm_fit'] and dp>0) else 0
            for j,v in enumerate([r['label'],r['dP_mbar'],dp,r['n_peaks'],
                                   r['wm_fit'] or 0,r['s0_fit'] or 0,r['r2'] or 0,E_i],1):
                dat(ws2,row,j,v)
            row+=1
        row+=2
    for col in range(1,9): ws2.column_dimensions[get_column_letter(col)].width=16

    # Feuille 3 : Profils 1D
    ws3=wb.create_sheet('Profils 1D')
    ws3['A1']='Profils de déformation 1D — biparabolique elliptique'
    ws3['A1'].font=Font(bold=True,size=11)
    ws3['A2']='w(s)=w_max·(1−((s−s0)/L)²)²   L=demi-longueur de l axe'
    n_pts=100; col=1
    for axis_name,res,half in [('Grand axe',res_G,A_UM),('Petit axe',res_P,B_UM)]:
        for r in res:
            ws3.merge_cells(start_row=3,start_column=col,end_row=3,end_column=col+1)
            ws3.cell(3,col,f"{axis_name} — {r['label']} ({r['dP_mbar']:.0f}mbar)")
            ws3.cell(3,col).font=Font(bold=True)
            ws3.cell(4,col,'s (µm)'); ws3.cell(4,col+1,'w(s) (µm)')
            if r['wm_fit'] and r['s0_fit']:
                s_arr=np.linspace(r['s0_fit']-half,r['s0_fit']+half,n_pts)
                w_arr=elliptic_profile(s_arr,r['wm_fit'],r['s0_fit'],half)
                for k,(s,w) in enumerate(zip(s_arr,w_arr),5):
                    ws3.cell(k,col,round(float(s),2)); ws3.cell(k,col+1,round(float(w),4))
            col+=3

    # Feuille 4 : Surface 3D (toutes pressions, modèle elliptique 2D vrai)
    ws4=wb.create_sheet('Surface 3D')
    ws4['A1']='Surface 3D z(x,y,ΔP) — REF: z=0=face verre'
    ws4['A1'].font=Font(bold=True,size=11)
    ws4['A2']=f'z=e0+w_max*(1-(x²/a²+y²/b²))²  e0={E0_UM:.0f}um a={A_UM:.0f}um b={B_UM:.0f}um'
    dPs_G_dict={r['dP_mbar']:r['wm_fit'] for r in res_G if r['wm_fit']}
    dPs_P_dict={r['dP_mbar']:r['wm_fit'] for r in res_P if r['wm_fit']}
    all_dP=sorted(set(list(dPs_G_dict.keys())+list(dPs_P_dict.keys())))
    x_sub=np.linspace(-A_UM,A_UM,20); y_sub=np.linspace(-B_UM,B_UM,20)
    print(f"  Excel Surface 3D : {len(all_dP)} pression(s)")
    row=4
    for dp in all_dP:
        wm_g=dPs_G_dict.get(dp,None); wm_p=dPs_P_dict.get(dp,None)
        if wm_g is None and wm_p is None: continue
        if wm_g is None: wm_g=wm_p
        if wm_p is None: wm_p=wm_g
        wm_mean=(wm_g+wm_p)/2
        ws4.cell(row,1,f'ΔP={dp:.0f} mbar  wGA={wm_g:.2f}µm wPA={wm_p:.2f}µm moy={wm_mean:.2f}µm')
        ws4.cell(row,1).font=Font(bold=True); row+=1
        ws4.cell(row,1,'x\\y(µm)')
        for j,y in enumerate(y_sub,2): ws4.cell(row,j,round(float(y),0))
        row+=1
        for xi in x_sub:
            ws4.cell(row,1,round(float(xi),0))
            for j,yi in enumerate(y_sub,2):
                u=(xi/A_UM)**2+(yi/B_UM)**2
                w_rel=wm_mean*(1-u)**2 if u<=1.0 else 0.0
                ws4.cell(row,j,round(float(w_rel+E0_UM),3))
            row+=1
        row+=2
        print(f"    ΔP={dp:.0f}mbar  wm={wm_mean:.2f}µm  z_centre={wm_mean+E0_UM:.2f}µm")

    out_xl=os.path.join(out_dir,'analyse_biaxiale.xlsx')
    wb.save(out_xl)
    print(f"\n  → Excel sauvegardé : {out_xl}")
    return out_xl


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def main():
    global LAMBDA_NM,SCALE_UM_PX,PEAK_DIST_PX,GAUSS_SIGMA
    global A_UM,B_UM,T_UM,E_PA,E_UNC,NU,E0_UM

    LAMBDA_NM   =CFG['lambda_nm'];   SCALE_UM_PX=CFG['scale_um_px']
    PEAK_DIST_PX=CFG['peak_dist'];   GAUSS_SIGMA=CFG['gauss_sigma']
    A_UM        =CFG['a_grand_um'];  B_UM       =CFG['b_petit_um']
    T_UM        =CFG['t_fixed_um'];  E_PA       =CFG['E_pa']
    E_UNC       =CFG['E_uncertainty_pa']; NU    =CFG['nu']
    E0_UM       =CFG.get('e0_um', 12.0)

    def ask_folder(title):
        root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
        f=filedialog.askdirectory(title=title,initialdir=os.path.expanduser('~'))
        root.destroy(); return f

    print("\n=== ANALYSE BIAXIALE — MODÈLE ELLIPTIQUE 2D ===")
    folder_G=ask_folder("Sélectionnez le dossier — GRAND AXE")
    if not folder_G: print("Annulé."); return
    folder_P=ask_folder("Sélectionnez le dossier — PETIT AXE")
    if not folder_P: print("Annulé."); return

    out_dir=CFG['output_dir'] or folder_G
    os.makedirs(out_dir,exist_ok=True)

    print(f"\n  Grand axe : {folder_G}")
    print(f"  Petit axe : {folder_P}")
    print(f"  Sortie    : {out_dir}")
    print(f"  a={A_UM:.0f}µm  b={B_UM:.0f}µm  t={T_UM:.0f}µm  E={E_PA/1e6:.2f}MPa  e0={E0_UM:.0f}µm")

    res_G=analyse_axis(folder_G,'Grand axe',A_UM,out_dir)
    res_P=analyse_axis(folder_P,'Petit axe', B_UM,out_dir)

    # Fit global : E_eff COMMUN aux deux axes, e0 fixé
    print(f"\n{'─'*65}")
    print("  FIT GLOBAL ELLIPTIQUE — E_eff COMMUN (e0 fixé, géométrie fixée)")
    print(f"{'─'*65}")
    Eeff,r2_glob,e0=fit_Eeff_common(res_G,res_P)
    if Eeff is None:
        print("  Fit impossible. Abandon des figures de synthèse."); return

    print("\n  Génération des figures (SVG) ...")
    fig_wmax_vs_dP(res_G,res_P,Eeff,r2_glob,e0,out_dir)
    fig_profiles(res_G,res_P,out_dir)
    fig_3d_surface(res_G,res_P,out_dir)

    print("\n  Export Excel ...")
    export_excel(res_G,res_P,Eeff,r2_glob,e0,out_dir)

    print(f"\n{'='*65}")
    print(f"  TERMINÉ — Résultats (SVG) dans : {out_dir}")
    print(f"  E_eff commun = {Eeff/1e6:.3f} MPa  (e0={e0:.0f}µm fixé, R²={r2_glob:.4f})")
    print(f"{'='*65}\n")


main()