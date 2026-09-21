"""
=============================================================================
ANALYSE BIAXIALE — GRAND AXE + PETIT AXE — MEMBRANE PDMS ELLIPTIQUE
=============================================================================
Analyse les franges d'interférence sur le grand axe ET le petit axe.

Modèle : Schomburg Eq.5 (plaque rectangulaire encastrée)
  w_max = W⁴(1−ν²)/(66·t³·E) · ΔP   avec  W = largeur effective

Pour chaque axe, deux courbes théoriques sont tracées :
  - Schomburg avec la vraie largeur géométrique (W_geom = 2a ou 2b)
  - Schomburg avec une largeur effective W_eff fittée sur les données exp.
    → W_eff donne la "largeur de membrane mécaniquement effective"

Figures de synthèse :
  1. w_max vs ΔP — les deux axes sur la même figure
  2. Profils de déformation superposés — les deux axes
  3. Surface 3D reconstruite à partir des profils des deux axes

Export Excel : profils par pression + paramètres de fit.

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
    'scale_um_px'         : 2.75,    # µm/pixel — calibrer avec une mire
    'lambda_nm'           : 661.0,   # nm longueur d'onde laser
    'a_grand_um'          : 3750.0,  # µm demi grand axe géométrique
    'b_petit_um'          : 2800.0,  # µm demi petit axe géométrique
    't_fixed_um'          : 1400.0,  # µm épaisseur membrane
    'E_pa'                : 1.3e6,   # Pa module de Young PDMS
    'E_uncertainty_pa'    : 0.3e6,   # Pa incertitude sur E
    'nu'                  : 0.5,     # Poisson PDMS
    'profile_width'       : 5,       # px demi-largeur bande
    'peak_dist'           : 2,       # px distance min entre pics
    'gauss_sigma'         : 12.0,    # px sigma lissage interfrange
    # Gap initial verre-PDMS à P=0 (µm)
    # Utilisé comme offset z pour la surface 3D et l'export
    'e0_um'               : 12.0,
    'n_3d_points'         : 80,      # points par axe pour la surface 3D
    'output_dir'          : None,    # None = dossier grand axe
}

# Constantes globales
LAMBDA_NM = 661.0; SCALE_UM_PX = 2.75; PEAK_DIST_PX = 2
GAUSS_SIGMA = 12.0; SG_WINDOW = 15; SG_ORDER = 3
A_UM = 3750.0; B_UM = 2800.0; T_UM = 1400.0
E_PA = 1.3e6; E_UNC = 0.3e6; NU = 0.5
E0_UM = 12.0  # gap initial verre-PDMS (µm)


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
    """Extrait la pression en mbar depuis le nom de fichier."""
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
# MODÈLE SCHOMBURG — PROFIL ET w_max
# =============================================================================
def schomburg_profile(s_um, w_max, s0, half_len):
    """w(s) = w_max·(1−((s−s0)/half_len)²)²"""
    u=(s_um-s0)/half_len
    return np.where(np.abs(u)<=1, w_max*(1-u**2)**2, 0.0)

def schomburg_wmax(dP_pa, W_um):
    """
    w_max (µm) = W⁴(1−ν²)/(66·t³·E) · ΔP
    W_um : largeur totale de la membrane dans le sens de l'axe (µm)
    """
    W=W_um*1e-6; t=T_UM*1e-6
    return (W**4*(1-NU**2))/(66*t**3*E_PA)*dP_pa*1e6

def schomburg_wmax_bounds(dP_pa, W_um):
    """Bandes haute et basse pour E±σE."""
    W=W_um*1e-6; t=T_UM*1e-6
    K_hi=(W**4*(1-NU**2))/(66*t**3*(E_PA-E_UNC))*1e6
    K_lo=(W**4*(1-NU**2))/(66*t**3*(E_PA+E_UNC))*1e6
    return K_hi*dP_pa, K_lo*dP_pa


# =============================================================================
# FIT DU PROFIL — w_max et s0 libres, half_len fixé
# =============================================================================
def fit_profile(s_um, w_D, half_len):
    """Fit biparabolique. Retourne (w_max, s0, r2, s_model, w_model)."""
    if w_D is None or len(w_D)<5: return None,None,None,None,None
    w0=float(np.max(w_D)); s_c=float(s_um[np.argmax(w_D)])
    if w0<=0: return None,None,None,None,None
    def res(s0): return np.sum((w_D-schomburg_profile(s_um,w0,s0,half_len))**2)
    opt=_ms(res,bounds=(s_c-half_len,s_c+half_len),method='bounded')
    s0_i=opt.x
    try:
        popt,_=curve_fit(
            lambda s,wm,s0: schomburg_profile(s,wm,s0,half_len),
            s_um,w_D,p0=[w0,s0_i],
            bounds=([0,s_c-2*half_len],[w0*8,s_c+2*half_len]),maxfev=12000)
        wm,s0f=float(popt[0]),float(popt[1])
    except: wm,s0f=w0,s0_i
    wp=schomburg_profile(s_um,wm,s0f,half_len)
    ss_res=np.sum((w_D-wp)**2); ss_tot=np.sum((w_D-w_D.mean())**2)
    r2=1-ss_res/ss_tot if ss_tot>0 else 0
    s_mod=np.linspace(s0f-half_len,s0f+half_len,800)
    w_mod=schomburg_profile(s_mod,wm,s0f,half_len)
    return wm,s0f,r2,s_mod,w_mod


# =============================================================================
# FIT GLOBAL — W_eff paramètre libre (Schomburg inversé)
# =============================================================================
def fit_Weff(dP_arr_mbar, wmax_arr):
    """
    Fitte la largeur effective W_eff dans le modèle Schomburg :
      w_max = K(W_eff) · ΔP   avec  K ∝ W_eff⁴

    Paramètre libre : W_eff (µm) = largeur effective de membrane.
    Retourne (W_eff_um, r2, dP_fit, w_fit).
    """
    valid=np.isfinite(dP_arr_mbar)&np.isfinite(wmax_arr)&(dP_arr_mbar>0)&(wmax_arr>0)
    if valid.sum()<2: return None,None,None,None
    dP_v=dP_arr_mbar[valid]*100   # Pa
    wm_v=wmax_arr[valid]

    def model(dP_pa, W_um):
        return schomburg_wmax(dP_pa, W_um)

    try:
        popt,_=curve_fit(model, dP_v, wm_v,
                         p0=[5000.0],
                         bounds=([100.0],[30000.0]),
                         maxfev=10000)
        Weff=float(popt[0])
    except Exception as e:
        print(f"  [Fit W_eff] {e}"); return None,None,None,None

    w_pred=schomburg_wmax(dP_v,Weff)
    ss_res=np.sum((wm_v-w_pred)**2); ss_tot=np.sum((wm_v-wm_v.mean())**2)
    r2=1-ss_res/ss_tot if ss_tot>0 else 0

    dP_fit=np.linspace(0,dP_v.max()*1.15,300)
    w_fit=schomburg_wmax(dP_fit,Weff)

    print(f"  W_eff = {Weff:.0f} µm  (2×W_eff/2 = {Weff/2:.0f} µm par demi-axe)  R²={r2:.4f}")
    return Weff, r2, dP_fit/100, w_fit  # dP en mbar pour affichage


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
# ANALYSE D'UN AXE — sauvegarde silencieuse des figures individuelles
# =============================================================================
def analyse_axis(folder, axis_name, half_len_geom, out_dir):
    """
    Analyse complète d'un axe.
    half_len_geom : demi-longueur géométrique (a ou b).
    Retourne liste de dicts avec toutes les données.
    """
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

        # Figure individuelle sauvegardée sans affichage
        fig,axes=plt.subplots(3,1,figsize=(13,12))
        fig.suptitle(f'{axis_name} — {label}  (ΔP={dP_mbar[idx]:.0f} mbar)',
                     fontsize=13,fontweight='bold')
        # Profil
        ax=axes[0]
        ax.plot(s_um,prof,color='#bdc3c7',lw=0.8,alpha=0.7,label='Brut')
        ax.plot(s_um,sm,color='#2c3e50',lw=1.4,label='Lissé S-G')
        if len(pk)>0: ax.plot(s_um[pk],sm[pk],'v',color='#27ae60',ms=5,zorder=6,
                               label=f'Maxima ({len(pk)})')
        if len(vl)>0: ax.plot(s_um[vl],sm[vl],'^',color='#3498db',ms=4,zorder=5,
                               label=f'Minima ({len(vl)})')
        ax.set_ylabel('Intensité norm.'); ax.legend(fontsize=7,ncol=2)
        ax.set_xlim(s_um[0],s_um[-1]); ax.grid(True,alpha=0.25)
        # Déformation
        ax2=axes[1]
        if s_D is not None:
            ax2.plot(s_D,w_D,color='#27ae60',lw=2,label='Méthode D')
            if w_mod is not None:
                ax2.plot(s_mod,w_mod,':',color='#95a5a6',lw=1.2,
                         label=f'Modèle géom. ({half_len_geom:.0f} µm)')
                wp=schomburg_profile(s_D,wm,s0f,half_len_geom)
                ax2.plot(s_D,wp,'--',color='#e67e22',lw=2,
                         label=f'Fit  w_max={wm:.2f}µm  R²={r2:.3f}')
        ax2.set_ylabel('w (µm)'); ax2.legend(fontsize=7)
        if s_D is not None: ax2.set_xlim(s_D[0],s_D[-1])
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
        fig.savefig(os.path.join(out_dir,f'fringe_{axis_name.replace(" ","_")}_{label}.png'),
                    dpi=150,bbox_inches='tight')
        plt.close(fig)
        print(f"    [{idx+1}] {label:20s}  n_max={len(pk):3d}  "
              f"wm={wm:.2f}µm  R²={r2:.3f}" if wm else
              f"    [{idx+1}] {label:20s}  n_max={len(pk):3d}  wm=N/A")

    return results


# =============================================================================
# FIGURE 1 — w_max vs ΔP : les deux axes + modèles Schomburg
# =============================================================================
def fig_wmax_vs_dP(res_G, res_P, Weff_G, Weff_P, out_dir):
    """
    Une figure avec deux panneaux (grand axe / petit axe) + légende commune.
    Pour chaque axe :
      - Courbe Schomburg avec largeur géométrique (trait fin)
      - Courbe Schomburg avec W_eff fittée (trait épais) + bande E±σ
      - Points expérimentaux colorés par pression
    """
    dP_range_mbar=np.linspace(0,200,300)
    dP_range_pa  =dP_range_mbar*100

    fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=False)
    fig.suptitle(
        f'Distance verre→PDMS vs Dépression — Modèle Schomburg\n'
        f't={T_UM:.0f}µm  E={E_PA/1e6:.2f}MPa  ν={NU}  '
        f'Gap initial e₀={E0_UM:.0f}µm  (w=0 → z=e₀ à P=0)',
        fontsize=12,fontweight='bold')

    for ax,res,half_geom,Weff,axis_name,col_geom,col_fit in zip(
            axes,
            [res_G,res_P],
            [A_UM,B_UM],
            [Weff_G,Weff_P],
            ['Grand axe','Petit axe'],
            ['#2980b9','#c0392b'],
            ['#1a5276','#7b241c']):

        W_geom=2*half_geom

        # ── Courbe géométrique (trait fin, pointillé) ─────────────────────
        # Offset e0 : à P=0, la membrane est déjà à e0=12µm du verre
        w_geom=np.array([schomburg_wmax(dp,W_geom)+E0_UM for dp in dP_range_pa])
        ax.plot(dP_range_mbar,w_geom,'--',color=col_geom,lw=1.5,alpha=0.7,
                label=f'Schomburg géom. W={W_geom:.0f}µm (e₀={E0_UM:.0f}µm)')

        # ── Courbe W_eff fittée + bande E±σ ──────────────────────────────
        if Weff is not None:
            w_eff=np.array([schomburg_wmax(dp,Weff) for dp in dP_range_pa])
            # Bande : K ∝ 1/E
            t=T_UM*1e-6; W=Weff*1e-6
            K_center=(W**4*(1-NU**2))/(66*t**3*E_PA)*1e6
            K_hi    =(W**4*(1-NU**2))/(66*t**3*(E_PA-E_UNC))*1e6
            K_lo    =(W**4*(1-NU**2))/(66*t**3*(E_PA+E_UNC))*1e6
            ax.plot(dP_range_mbar,K_center*dP_range_pa+E0_UM,'-',color=col_fit,lw=2.5,
                    label=f'Schomburg W_eff={Weff:.0f}µm (e₀={E0_UM:.0f}µm)')
            ax.fill_between(dP_range_mbar,
                            K_lo*dP_range_pa+E0_UM,
                            K_hi*dP_range_pa+E0_UM,
                            color=col_fit,alpha=0.15,
                            label=f'Bande E±{E_UNC/1e6:.1f}MPa')

        # ── Points expérimentaux ──────────────────────────────────────────
        n=len(res)
        cmap=cm.Blues if 'Grand' in axis_name else cm.Reds
        colors=[cmap(0.4+0.5*i/max(n-1,1)) for i in range(n)]
        for i,(r,col) in enumerate(zip(res,colors)):
            if r['wm_fit'] is not None and np.isfinite(r['dP_mbar']):
                # Offset e0 : les points représentent e0 + déflexion relative
                w_abs = r['wm_fit'] + E0_UM
                ax.scatter(r['dP_mbar'],w_abs,
                           color=col,s=90,zorder=7,
                           edgecolors=col_fit,lw=1.0,
                           marker='o' if 'Grand' in axis_name else 's')
                ax.annotate(r['label'],(r['dP_mbar'],w_abs),
                            textcoords='offset points',xytext=(5,3),
                            fontsize=7,color=col_fit)

        if Weff is not None:
            ax.set_title(f'{axis_name}\nW_geom={W_geom:.0f}µm  →  '
                         f'W_eff={Weff:.0f}µm  '
                         f'(W_eff/2={Weff/2:.0f}µm)\n'
                         f'Gap initial e₀={E0_UM:.0f}µm  '
                         f'(~{2*E0_UM/0.661:.0f} franges visibles à P=0)',
                         fontsize=9)
        else:
            ax.set_title(f'{axis_name}  W_geom={W_geom:.0f}µm',fontsize=10)

        ax.set_xlabel('Dépression ΔP (mbar)',fontsize=10)
        ax.set_ylabel(f'Distance verre→PDMS (µm)  [e₀={E0_UM:.0f}µm à P=0]',fontsize=10)
        ax.legend(fontsize=8,loc='upper left')
        ax.grid(True,alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'synthese_wmax_vs_dP.png'),
                dpi=150,bbox_inches='tight')
    plt.show(); plt.close()

    # Affichage terminal
    print(f"\n  {'Axe':<12} {'W_geom (µm)':>12} {'W_eff (µm)':>12} "
          f"{'W_eff/2 (µm)':>13} {'Ratio W_eff/W_geom':>18}")
    print(f"  {'─'*72}")
    for name,wg,weff in [('Grand axe',2*A_UM,Weff_G),('Petit axe',2*B_UM,Weff_P)]:
        if weff:
            print(f"  {name:<12} {wg:>12.0f} {weff:>12.0f} "
                  f"{weff/2:>13.0f} {weff/wg:>18.3f}")


# =============================================================================
# FIGURE 2 — Profils superposés avec modèles
# =============================================================================
def fig_profiles(res_G, res_P, Weff_G, Weff_P, out_dir):
    """
    Deux panneaux (GA / PA). Pour chaque image :
      - Données exp. en trait épais plein + remplissage hachuré léger
      - Fit biparabolique (prolongé hors zone observée en pointillé)
      - Schomburg avec W_eff en trait mixte (sur la pleine largeur effective)
    """
    cmap=cm.plasma

    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    fig.suptitle('Profils de déformation — Fit biparabolique (pleine membrane)',
                 fontsize=13,fontweight='bold')

    for ax,res,half_geom,Weff,axis_name in zip(
            axes,[res_G,res_P],[A_UM,B_UM],[Weff_G,Weff_P],
            ['Grand axe','Petit axe']):

        half_eff=Weff/2 if Weff else half_geom
        n=len(res)

        for i,r in enumerate(res):
            col=cmap(0.1+0.8*i/max(n-1,1))
            lbl=f"{r['label']} ({r['dP_mbar']:.0f}mbar)"

            # ── Fit biparabolique uniquement (pleine membrane) ────────────
            if r['wm_fit'] is not None and r['s0_fit'] is not None:
                s_full=np.linspace(r['s0_fit']-half_geom,r['s0_fit']+half_geom,600)
                w_full=schomburg_profile(s_full,r['wm_fit'],r['s0_fit'],half_geom)
                ax.plot(s_full,w_full,'-',color=col,lw=2.0,
                        label=f"{lbl}  w_max={r['wm_fit']:.1f}µm",zorder=5)

        ax.set_title(f'{axis_name}  '
                     f'(W_eff={Weff:.0f}µm)' if Weff else axis_name,fontsize=10)
        ax.set_xlabel('Position s (µm)',fontsize=10)
        ax.set_ylabel('Déformation w (µm)',fontsize=10)
        ax.grid(True,alpha=0.25); ax.set_ylim(bottom=0)
        ax.legend(fontsize=7,loc='upper left',ncol=1)

    # Pas de légende globale — labels dans chaque panneau
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'synthese_profiles.png'),
                dpi=150,bbox_inches='tight')
    plt.show(); plt.close()


# =============================================================================
# FIGURE 3 — SURFACE 3D RECONSTRUITE (PRESSION MAX UNIQUEMENT)
# =============================================================================
def fig_3d_surface(res_G, res_P, Weff_G, Weff_P, out_dir):
    """
    Affiche la nappe 3D à la PRESSION MAXIMALE uniquement.

    w(x,y) = e0 + w_max * (1-(x/a_eff)²)² * (1-(y/b_eff)²)²
    Référentiel : z=0 = face inférieure du verre.

    La base XY est l'ellipse de demi-axes a_eff et b_eff (issus du fit W_eff).
    Elle est tracée au sol (z=0) et à z=e0 (position repos membrane).
    """
    n   = CFG['n_3d_points']
    half_G = Weff_G/2 if Weff_G else A_UM   # a_eff en µm
    half_P = Weff_P/2 if Weff_P else B_UM   # b_eff en µm

    # ── Données à la pression maximale ───────────────────────────────────
    dPs_G = {r['dP_mbar']:r for r in res_G
             if r['wm_fit'] is not None and np.isfinite(r['dP_mbar'])}
    dPs_P = {r['dP_mbar']:r for r in res_P
             if r['wm_fit'] is not None and np.isfinite(r['dP_mbar'])}

    common_dP = sorted(set(dPs_G.keys()) & set(dPs_P.keys()))
    if not common_dP:
        common_dP = sorted(dPs_G.keys())
    if not common_dP:
        print("  [3D] Aucune donnée disponible."); return [], None, None, []

    dp_max = max(common_dP)
    wm_g   = dPs_G[dp_max]['wm_fit']
    wm_p   = dPs_P.get(dp_max, {}).get('wm_fit', wm_g) if dp_max in dPs_P else wm_g
    wm_max = (wm_g + wm_p) / 2   # µm, déflexion relative

    # ── Grille elliptique (masquée hors ellipse) ──────────────────────────
    x = np.linspace(-half_G, half_G, n)
    y = np.linspace(-half_P, half_P, n)
    X, Y = np.meshgrid(x, y)
    ux = (X/half_G)**2; uy = (Y/half_P)**2
    inside = (ux + uy) <= 1.0   # masque elliptique strict

    W_rel = np.where(inside, wm_max*(1 - ux - uy)**2, np.nan)  # déflexion relative
    W_abs = W_rel + E0_UM   # z absolu (z=0 face verre)

    # Nappe à P=0 (plan horizontal à z=e0)
    W_rest = np.where(inside, E0_UM, np.nan)

    # ── Ellipse de bord (contour au sol et à z=e0) ───────────────────────
    theta_ell = np.linspace(0, 2*np.pi, 300)
    xe = half_G * np.cos(theta_ell) / 1000   # mm
    ye = half_P * np.sin(theta_ell) / 1000   # mm

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    ax3d = fig.add_subplot(111, projection='3d')

    fig.suptitle(
        f'Surface 3D membrane PDMS — Pression max : ΔP = {dp_max:.0f} mbar\n'
        f'a_eff = {half_G:.0f} µm  |  b_eff = {half_P:.0f} µm  |  '
        f'e₀ = {E0_UM:.0f} µm  |  w_max = {wm_max:.1f} µm\n'
        f'Référentiel : z = 0 = face inférieure du verre',
        fontsize=11, fontweight='bold'
    )

    # ── Nappe déformée (z = e0 + w) ───────────────────────────────────────
    surf = ax3d.plot_surface(
        X/1000, Y/1000, W_abs,
        cmap='coolwarm', alpha=0.85,
        linewidth=0, antialiased=True,
        vmin=E0_UM, vmax=E0_UM + wm_max
    )
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=12,
                 label='Distance verre→PDMS (µm)', pad=0.1)

    # ── Nappe de repos P=0 (plan semi-transparent) ────────────────────────
    ax3d.plot_surface(
        X/1000, Y/1000, W_rest,
        color='#aab4c8', alpha=0.25,
        linewidth=0, antialiased=False
    )

    # ── Ellipse de bord au sol (z=0) ─────────────────────────────────────
    ax3d.plot(xe, ye,
              np.zeros_like(xe),
              'k-', lw=1.5, alpha=0.6,
              label=f'Ellipse W_eff : a={half_G:.0f}µm, b={half_P:.0f}µm')

    # ── Ellipse de bord à z=e0 (position repos) ───────────────────────────
    ax3d.plot(xe, ye,
              np.full_like(xe, E0_UM),
              '--', color='#2980b9', lw=1.5, alpha=0.8,
              label=f'PDMS@P=0  (z=e₀={E0_UM:.0f}µm)')

    # ── Ellipse de bord à z=e0+wmax (sommet déformé) ─────────────────────
    # Valeur au bord = e0 (le bord est encastré, w_bord=0)
    # Marquer le centre : point à z_max
    ax3d.scatter([0],[0],[E0_UM + wm_max],
                 color='#e74c3c', s=80, zorder=10,
                 label=f'Centre : z={E0_UM+wm_max:.1f}µm (w={wm_max:.1f}µm)')

    # ── Flèche verticale au centre ────────────────────────────────────────
    ax3d.plot([0,0],[0,0],[E0_UM, E0_UM+wm_max],
              'r-', lw=2.0, alpha=0.8)
    ax3d.text(0, half_P/1000*0.3, E0_UM + wm_max/2,
              f'  w={wm_max:.1f}µm', color='#e74c3c', fontsize=9)

    # ── Coupes 1D sur les deux axes ───────────────────────────────────────
    mid = n//2
    # Grand axe (y=0)
    ax3d.plot(X[mid,:]/1000, Y[mid,:]/1000, W_abs[mid,:],
              'k-', lw=1.5, alpha=0.9, zorder=5)
    # Petit axe (x=0)
    ax3d.plot(X[:,mid]/1000, Y[:,mid]/1000, W_abs[:,mid],
              'k-', lw=1.5, alpha=0.9, zorder=5)

    # ── Mise en forme ─────────────────────────────────────────────────────
    ax3d.set_xlabel('x — Grand axe (mm)', fontsize=9, labelpad=8)
    ax3d.set_ylabel('y — Petit axe (mm)',  fontsize=9, labelpad=8)
    ax3d.set_zlabel('z : dist. verre→PDMS (µm)', fontsize=9, labelpad=8)
    ax3d.set_xlim(-half_G/1000, half_G/1000)
    ax3d.set_ylim(-half_P/1000, half_P/1000)
    ax3d.set_zlim(0, E0_UM + wm_max * 1.1)
    ax3d.legend(fontsize=8, loc='upper left')

    # Angle de vue
    ax3d.view_init(elev=25, azim=-60)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'surface_3D.png'),
                dpi=150, bbox_inches='tight')
    plt.show(); plt.close()

    # Données pour export Excel
    surfaces = [(dp_max, W_rel, '#e74c3c')]
    return surfaces, X, Y, common_dP


# =============================================================================
# EXPORT EXCEL
# =============================================================================
def export_excel(res_G, res_P, Weff_G, Weff_P, surfaces_3d, X, Y, common_dP, out_dir):
    wb=Workbook()

    # Styles
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

    # ── Feuille 1 : Paramètres + résultats de fit ─────────────────────────
    ws1=wb.active; ws1.title='Paramètres et fit'
    params=[('λ (nm)',LAMBDA_NM),('Échelle (µm/px)',SCALE_UM_PX),
            ('a géom. (µm)',A_UM),('b géom. (µm)',B_UM),
            ('t (µm)',T_UM),('E (Pa)',E_PA),('σE (Pa)',E_UNC),('ν',NU),
            ('W_eff Grand axe (µm)',Weff_G or 'N/A'),
            ('W_eff/2 Grand axe (µm)',Weff_G/2 if Weff_G else 'N/A'),
            ('W_eff Petit axe (µm)',Weff_P or 'N/A'),
            ('W_eff/2 Petit axe (µm)',Weff_P/2 if Weff_P else 'N/A'),
            ('Ratio W_eff_G/W_geom_G',Weff_G/(2*A_UM) if Weff_G else 'N/A'),
            ('Ratio W_eff_P/W_geom_P',Weff_P/(2*B_UM) if Weff_P else 'N/A')]
    hdr(ws1,1,1,'Paramètre','1F4E79'); hdr(ws1,1,2,'Valeur','1F4E79')
    for i,(k,v) in enumerate(params,2):
        ws1.cell(i,1,k); dat(ws1,i,2,v)
    ws1.column_dimensions['A'].width=35; ws1.column_dimensions['B'].width=18

    # ── Feuille 2 : Résultats expérimentaux ──────────────────────────────
    ws2=wb.create_sheet('Résultats exp.')
    row=1
    for axis_name,res,fill,half_geom,Weff in [
            ('GRAND AXE',res_G,'1F4E79',A_UM,Weff_G),
            ('PETIT AXE',res_P,'7B0000',B_UM,Weff_P)]:
        ws2.merge_cells(start_row=row,start_column=1,end_row=row,end_column=9)
        hdr(ws2,row,1,f'{axis_name} — W_geom={2*half_geom:.0f}µm  W_eff={Weff:.0f}µm' if Weff
            else f'{axis_name} — W_geom={2*half_geom:.0f}µm',fill)
        row+=1
        for j,h in enumerate(['Image','ΔP(mbar)','ΔP(Pa)','N_max',
                               'w_max_fit(µm)','s0(µm)','R²',
                               'w_Sch_geom(µm)','w_Sch_eff(µm)'],1):
            hdr(ws2,row,j,h,fill)
        row+=1
        for r in res:
            dp=r['dP_pa']
            w_geom=schomburg_wmax(dp,2*half_geom) if dp>0 else 0
            w_eff=schomburg_wmax(dp,Weff) if (Weff and dp>0) else 0
            for j,v in enumerate([r['label'],r['dP_mbar'],dp,r['n_peaks'],
                                   r['wm_fit'] or 0,r['s0_fit'] or 0,r['r2'] or 0,
                                   w_geom,w_eff],1):
                dat(ws2,row,j,v)
            row+=1
        row+=2
    for col in range(1,10): ws2.column_dimensions[get_column_letter(col)].width=16

    # ── Feuille 3 : Profils 1D par pression ──────────────────────────────
    ws3=wb.create_sheet('Profils 1D')
    ws3['A1']='Profils de déformation 1D — Schomburg avec W_eff'
    ws3['A1'].font=Font(bold=True,size=11)
    ws3['A2']=(f'w(s) = w_max·(1−((s−s₀)/W_eff×½)²)²   '
               f'w_max = [W_eff⁴(1−ν²)/(66t³E)]·ΔP')

    n_pts=100; col=1
    for axis_name,res,half_eff in [
            ('Grand axe',res_G,Weff_G/2 if Weff_G else A_UM),
            ('Petit axe',res_P,Weff_P/2 if Weff_P else B_UM)]:
        for r in res:
            ws3.merge_cells(start_row=3,start_column=col,end_row=3,end_column=col+1)
            ws3.cell(3,col,f"{axis_name} — {r['label']} ({r['dP_mbar']:.0f}mbar)")
            ws3.cell(3,col).font=Font(bold=True)
            ws3.cell(4,col,'s (µm)'); ws3.cell(4,col+1,'w(s) (µm)')
            if r['wm_fit'] and r['s0_fit']:
                s_arr=np.linspace(r['s0_fit']-half_eff,r['s0_fit']+half_eff,n_pts)
                w_arr=schomburg_profile(s_arr,r['wm_fit'],r['s0_fit'],half_eff)
                for k,(s,w) in enumerate(zip(s_arr,w_arr),5):
                    ws3.cell(k,col,round(float(s),2))
                    ws3.cell(k,col+1,round(float(w),4))
            col+=3

    # ── Feuille 4 : Surface 3D (grille w(x,y)) ───────────────────────────
    ws4=wb.create_sheet('Surface 3D')
    ws4['A1']='Surface 3D z(x,y,ΔP) — REF: z=0=face verre | z=e0 → PDMS@P=0'
    ws4['A1'].font=Font(bold=True,size=11)
    half_G=Weff_G/2 if Weff_G else A_UM
    half_P=Weff_P/2 if Weff_P else B_UM
    ws4['A2']=f'z=e0+w_max*(1-(x²/a²+y²/b²))²  e0={E0_UM:.0f}um  a={half_G:.0f}um  b={half_P:.0f}um'

    # Toutes les pressions : union des deux axes
    dPs_G_dict={r['dP_mbar']:r['wm_fit'] for r in res_G if r['wm_fit']}
    dPs_P_dict={r['dP_mbar']:r['wm_fit'] for r in res_P if r['wm_fit']}
    all_dP=sorted(set(list(dPs_G_dict.keys())+list(dPs_P_dict.keys())))

    # Grille XY fixe 20×20
    x_sub=np.linspace(-half_G,half_G,20)
    y_sub=np.linspace(-half_P,half_P,20)

    print(f"  Excel Surface 3D : {len(all_dP)} pression(s)")
    row=4
    for dp in all_dP:
        wm_g=dPs_G_dict.get(dp,None)
        wm_p=dPs_P_dict.get(dp,None)
        if wm_g is None and wm_p is None: continue
        if wm_g is None: wm_g=wm_p
        if wm_p is None: wm_p=wm_g
        wm_mean=(wm_g+wm_p)/2

        ws4.cell(row,1,(f'ΔP={dp:.0f} mbar  '
                        f'wGA={wm_g:.2f}µm  wPA={wm_p:.2f}µm  moy={wm_mean:.2f}µm'))
        ws4.cell(row,1).font=Font(bold=True); row+=1

        ws4.cell(row,1,'x\\y(µm)')
        for j,y in enumerate(y_sub,2): ws4.cell(row,j,round(float(y),0))
        row+=1

        for xi in x_sub:
            ws4.cell(row,1,round(float(xi),0))
            for j,yi in enumerate(y_sub,2):
                ux=(xi/half_G)**2; uy=(yi/half_P)**2
                in_ell=(ux+uy)<=1.0
                w_rel=wm_mean*(1-ux-uy)**2 if in_ell else 0.0
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

    print("\n=== ANALYSE BIAXIALE MEMBRANE PDMS ===")
    folder_G=ask_folder("Sélectionnez le dossier — GRAND AXE")
    if not folder_G: print("Annulé."); return
    folder_P=ask_folder("Sélectionnez le dossier — PETIT AXE")
    if not folder_P: print("Annulé."); return

    out_dir=CFG['output_dir'] or folder_G
    os.makedirs(out_dir,exist_ok=True)

    print(f"\n  Grand axe : {folder_G}")
    print(f"  Petit axe : {folder_P}")
    print(f"  Sortie    : {out_dir}")
    print(f"  a={A_UM:.0f}µm  b={B_UM:.0f}µm  t={T_UM:.0f}µm  E={E_PA/1e6:.2f}MPa")

    # ── Analyse des deux axes ─────────────────────────────────────────────
    res_G=analyse_axis(folder_G,'Grand axe',A_UM,out_dir)
    res_P=analyse_axis(folder_P,'Petit axe', B_UM,out_dir)

    # ── Fit global W_eff pour chaque axe ─────────────────────────────────
    print(f"\n{'─'*65}")
    print("  FIT W_eff — GRAND AXE")
    dP_G=np.array([r['dP_mbar'] for r in res_G])
    wm_G=np.array([r['wm_fit'] if r['wm_fit'] else np.nan for r in res_G])
    Weff_G,r2_G,dP_fit_G,w_fit_G=fit_Weff(dP_G,wm_G)

    print(f"\n{'─'*65}")
    print("  FIT W_eff — PETIT AXE")
    dP_P=np.array([r['dP_mbar'] for r in res_P])
    wm_P=np.array([r['wm_fit'] if r['wm_fit'] else np.nan for r in res_P])
    Weff_P,r2_P,dP_fit_P,w_fit_P=fit_Weff(dP_P,wm_P)

    # ── Figures de synthèse ───────────────────────────────────────────────
    print("\n  Génération des figures ...")
    fig_wmax_vs_dP(res_G,res_P,Weff_G,Weff_P,out_dir)
    fig_profiles(res_G,res_P,Weff_G,Weff_P,out_dir)
    surfaces_3d,X,Y,common_dP=fig_3d_surface(res_G,res_P,Weff_G,Weff_P,out_dir)

    # ── Export Excel ──────────────────────────────────────────────────────
    print("\n  Export Excel ...")
    export_excel(res_G,res_P,Weff_G,Weff_P,surfaces_3d,X,Y,common_dP,out_dir)

    print(f"\n{'='*65}")
    print(f"  TERMINÉ — Résultats dans : {out_dir}")
    if Weff_G: print(f"  W_eff grand axe = {Weff_G:.0f} µm  (géom. = {2*A_UM:.0f} µm)")
    if Weff_P: print(f"  W_eff petit axe = {Weff_P:.0f} µm  (géom. = {2*B_UM:.0f} µm)")
    print(f"{'='*65}\n")


main()

## VIEWER 3D

"""
=============================================================================
VIEWER 3D — Nappe elliptique depuis l'Excel d'analyse biaxiale
=============================================================================
Lit le fichier analyse_biaxiale.xlsx généré par fringe_biaxial.py.
Permet de :
  - Sélectionner la pression à visualiser (menu interactif)
  - Ajuster le rapport d'aspect z/xy (facteur d'exagération verticale)
  - Exporter la figure en PNG

UTILISATION (Pyzo) :
  1. Renseigner les paramètres ci-dessous
  2. F5 → menu de sélection → figure 3D

=============================================================================
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, simpledialog
from openpyxl import load_workbook

# =============================================================================
# PARAMÈTRES  <<<  MODIFIER ICI  >>>
# =============================================================================

CFG = {
    # Fichier Excel (None → fenêtre de sélection)
    'excel_file'     : None,

    # Facteur d'exagération verticale (z_display = z_real / z_scale)
    # 1.0  → échelle réelle (z en µm, x/y en mm → z apparaît très aplati)
    # 10   → z exagéré ×10 (intermédiaire)
    # 50   → z exagéré ×50 (forme bien visible)
    # 'auto' → calcule automatiquement pour que w_max ≈ 20% de a_eff
    'z_scale'        : 'auto',

    # Colormap : 'coolwarm', 'plasma', 'viridis', 'RdYlBu_r', 'turbo'
    'cmap'           : 'coolwarm',

    # Angle de vue (degrés)
    'elev'           : 28,
    'azim'           : -55,

    # Résolution de la grille de rendu (points par axe)
    'n_render'       : 120,

    # Afficher les coupes 1D sur les axes
    'show_coupes'    : True,

    # Afficher la nappe de repos (P=0)
    'show_repos'     : True,

    # Sauvegarder la figure PNG dans le même dossier que l'Excel
    'save_png'       : True,
}


# =============================================================================
# LECTURE DE L'EXCEL
# =============================================================================

def parse_excel(xl_path):
    """
    Lit la feuille 'Surface 3D' de l'Excel.
    Retourne :
      meta   : dict avec a_eff, b_eff, e0 extraits de l'en-tête
      sheets : dict {pression_mbar: grille_z (array 2D)}
               + 'x_vals' et 'y_vals' (arrays 1D en µm)
    """
    wb = load_workbook(xl_path, data_only=True)

    # ── Métadonnées depuis feuille Paramètres ────────────────────────────
    meta = {'a_eff': None, 'b_eff': None, 'e0': 12.0}
    if 'Paramètres et fit' in wb.sheetnames:
        ws_p = wb['Paramètres et fit']
        for row in ws_p.iter_rows(values_only=True):
            if row[0] and row[1] is not None:
                key = str(row[0]).lower()
                val = row[1]
                if 'w_eff/2 grand' in key and isinstance(val, (int, float)):
                    meta['a_eff'] = float(val)
                elif 'w_eff/2 petit' in key and isinstance(val, (int, float)):
                    meta['b_eff'] = float(val)
                elif 'gap' in key and 'e' in key and isinstance(val, (int, float)):
                    meta['e0'] = float(val)

    # ── Données depuis feuille Surface 3D ────────────────────────────────
    if 'Surface 3D' not in wb.sheetnames:
        raise ValueError("Feuille 'Surface 3D' introuvable dans l'Excel.")
    ws = wb['Surface 3D']

    # Lire en-tête A2 pour extraire a_eff, b_eff, e0 si non trouvés
    a2 = ws['A2'].value or ''
    nums = re.findall(r'[\d.]+', str(a2))
    if len(nums) >= 3 and meta['a_eff'] is None:
        meta['e0']    = float(nums[0])
        meta['a_eff'] = float(nums[1])
        meta['b_eff'] = float(nums[2])

    # Parser les blocs de données : chaque pression est un bloc
    # Format : ligne "ΔP=XX mbar", puis ligne d'en-tête y, puis lignes x
    all_rows = list(ws.iter_rows(values_only=True))
    sheets = {}
    i = 0
    while i < len(all_rows):
        row = all_rows[i]
        # Détecter une ligne "ΔP=XX mbar"
        if row[0] and isinstance(row[0], str) and 'ΔP=' in row[0]:
            m = re.search(r'[\d.]+', row[0])
            dp = float(m.group()) if m else None
            if dp is not None:
                # Ligne suivante = en-tête y (1ère cellule = 'x\y', reste = valeurs y)
                i += 1
                hdr_row = all_rows[i]
                y_vals = [float(v) for v in hdr_row[1:]
                          if v is not None and str(v).replace('.','',1).lstrip('-').isdigit()]
                # Lignes de données
                x_vals = []
                grid   = []
                i += 1
                while i < len(all_rows):
                    drow = all_rows[i]
                    if drow[0] is None:
                        break
                    if isinstance(drow[0], str) and 'ΔP=' in drow[0]:
                        break
                    try:
                        x_val = float(drow[0])
                    except (TypeError, ValueError):
                        break
                    x_vals.append(x_val)
                    z_row = []
                    for v in drow[1:len(y_vals)+1]:
                        try:
                            z_row.append(float(v) if v is not None else np.nan)
                        except (TypeError, ValueError):
                            z_row.append(np.nan)
                    grid.append(z_row)
                    i += 1

                if x_vals and grid:
                    sheets[dp] = {
                        'x': np.array(x_vals, dtype=float),
                        'y': np.array(y_vals,  dtype=float),
                        'z': np.array(grid,    dtype=float),
                    }
                continue
        i += 1

    if not sheets:
        raise ValueError("Aucune donnée de pression trouvée dans 'Surface 3D'.")

    print(f"  Excel lu — {len(sheets)} pression(s) disponible(s) :")
    for dp in sorted(sheets):
        zdata = sheets[dp]['z']
        z_max = np.nanmax(zdata)
        print(f"    ΔP = {dp:>6.0f} mbar  |  z_max = {z_max:.2f} µm  "
              f"  (grille {zdata.shape[0]}×{zdata.shape[1]})")

    return meta, sheets


# =============================================================================
# SÉLECTION INTERACTIVE DE LA PRESSION
# =============================================================================

def choose_pressure(available_dP):
    """
    Affiche un menu tkinter pour choisir la pression.
    Retourne la pression sélectionnée (float).
    """
    root = tk.Tk()
    root.title("Sélection de la pression")
    root.attributes('-topmost', True)

    dPs_sorted = sorted(available_dP)
    selected   = tk.DoubleVar(value=dPs_sorted[-1])  # défaut = max

    tk.Label(root,
             text="Choisissez la dépression à visualiser :",
             font=('Arial', 11, 'bold'), pady=8).pack()

    frame = tk.Frame(root); frame.pack(padx=20, pady=5)
    for dp in dPs_sorted:
        tk.Radiobutton(frame, text=f"ΔP = {dp:.0f} mbar",
                       variable=selected, value=dp,
                       font=('Arial', 10)).pack(anchor='w')

    tk.Label(root, text="").pack()

    # Facteur d'échelle z
    tk.Label(root, text="Facteur d'exagération verticale z :",
             font=('Arial', 10)).pack()
    z_var = tk.StringVar(value=str(CFG['z_scale']))
    tk.Entry(root, textvariable=z_var, width=10,
             font=('Arial', 10)).pack(pady=4)
    tk.Label(root,
             text="'auto' = automatique  |  nombre = facteur manuel\n"
                  "1 = réel (très aplati)   50 = très exagéré",
             font=('Arial', 8), fg='#555555').pack()

    result = {}

    def validate():
        result['dp']      = selected.get()
        result['z_scale'] = z_var.get().strip()
        root.destroy()

    tk.Button(root, text="  Afficher la nappe 3D  ",
              command=validate,
              font=('Arial', 11, 'bold'),
              bg='#27ae60', fg='white', pady=6).pack(pady=12)

    root.mainloop()
    return result.get('dp', dPs_sorted[-1]), result.get('z_scale', 'auto')


# =============================================================================
# FIGURE 3D
# =============================================================================

def plot_3d(meta, sheets, dp_mbar, z_scale_str, out_dir):
    """
    Trace la nappe 3D pour la pression sélectionnée.
    z_scale : facteur d'exagération verticale (z_affiche = z_réel × z_scale)
              'auto' → calcul automatique.
    """
    if dp_mbar not in sheets:
        print(f"  Pression {dp_mbar} mbar introuvable.")
        return

    data   = sheets[dp_mbar]
    x_um   = data['x']   # µm
    y_um   = data['y']   # µm
    z_abs  = data['z']   # µm  (z absolu : e0 + w(x,y))

    e0     = meta.get('e0',    12.0)
    a_eff  = meta.get('a_eff', np.abs(x_um).max())
    b_eff  = meta.get('b_eff', np.abs(y_um).max())
    w_max  = float(np.nanmax(z_abs) - e0)

    # ── Facteur d'exagération ─────────────────────────────────────────────
    if str(z_scale_str).lower() == 'auto':
        # Cible : w_max apparent ≈ 20% de a_eff en unité d'affichage
        # x en mm → a_eff_mm = a_eff/1000
        # z en µm → z_mm = z/1000
        # On veut (w_max/1000)*factor = 0.20*(a_eff/1000)
        # factor = 0.20*a_eff / w_max
        if w_max > 0:
            z_factor = max(1.0, 0.20 * a_eff / w_max)
        else:
            z_factor = 1.0
        print(f"  z_scale auto → facteur = {z_factor:.1f}× "
              f"(w_max={w_max:.1f}µm, a_eff={a_eff:.0f}µm)")
    else:
        try:
            z_factor = float(z_scale_str)
        except ValueError:
            print(f"  z_scale '{z_scale_str}' invalide → auto")
            z_factor = max(1.0, 0.20 * a_eff / max(w_max, 1))

    # ── Grille haute résolution par interpolation ─────────────────────────
    from scipy.interpolate import RectBivariateSpline
    n = CFG['n_render']
    x_fine = np.linspace(x_um.min(), x_um.max(), n)
    y_fine = np.linspace(y_um.min(), y_um.max(), n)

    # Remplacer NaN par 0 pour l'interpolation (hors ellipse = e0)
    z_interp = np.where(np.isnan(z_abs), e0, z_abs)
    spline   = RectBivariateSpline(x_um, y_um, z_interp.T, kx=3, ky=3)
    Z_fine   = spline(x_fine, y_fine).T   # shape (n_y, n_x)

    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Masque elliptique sur la grille fine
    ux_f = (X_fine / a_eff)**2; uy_f = (Y_fine / b_eff)**2
    inside_f = (ux_f + uy_f) <= 1.0
    Z_fine_masked = np.where(inside_f, Z_fine, np.nan)
    Z_repos = np.where(inside_f, e0, np.nan)

    # Valeurs affichées (exagérées en z autour de e0)
    # z_display = e0 + (z_real - e0) * z_factor  [en µm]
    # puis on convertit x,y en mm et z en µm*factor pour l'affichage
    Z_disp   = e0 + (Z_fine_masked - e0) * z_factor
    Z_repos_d= np.full_like(Z_repos, e0)   # repos reste à e0 (z_factor ne s'applique pas ici)

    # ── Ellipse de bord ───────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 400)
    xe_mm = a_eff/1000 * np.cos(theta)
    ye_mm = b_eff/1000 * np.sin(theta)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    fig.suptitle(
        f'Nappe 3D — ΔP = {dp_mbar:.0f} mbar  |  '
        f'a_eff = {a_eff:.0f} µm  |  b_eff = {b_eff:.0f} µm\n'
        f'e₀ = {e0:.0f} µm  |  w_max = {w_max:.1f} µm  |  '
        f'Exagération z × {z_factor:.1f}',
        fontsize=11, fontweight='bold'
    )

    # Surface déformée
    surf = ax.plot_surface(
        X_fine/1000, Y_fine/1000, Z_disp,
        cmap=CFG['cmap'], alpha=0.88,
        linewidth=0, antialiased=True,
        vmin=e0, vmax=e0 + (w_max) * z_factor
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=14, pad=0.08)
    cbar.set_label(f'z réel (µm)  [affiché × {z_factor:.1f}]', fontsize=8)
    # Recalibrer les ticks de la colorbar en valeurs réelles
    z_ticks_disp = cbar.get_ticks()
    z_ticks_real = e0 + (z_ticks_disp - e0) / z_factor
    cbar.set_ticklabels([f'{v:.1f}' for v in z_ticks_real])

    # Plan de repos P=0
    if CFG['show_repos']:
        ax.plot_surface(
            X_fine/1000, Y_fine/1000, Z_repos_d,
            color='#a8bfd4', alpha=0.20,
            linewidth=0, antialiased=False
        )

    # Ellipse de bord au niveau du plan de repos
    ax.plot(xe_mm, ye_mm, np.full_like(xe_mm, e0),
            'k-', lw=1.8, alpha=0.7,
            label=f'Ellipse W_eff  a={a_eff:.0f}µm, b={b_eff:.0f}µm')

    # Coupes 1D
    if CFG['show_coupes']:
        mid_x = np.argmin(np.abs(x_fine))
        mid_y = np.argmin(np.abs(y_fine))
        # Grand axe (y=0)
        ax.plot(x_fine/1000,
                np.zeros(n),
                Z_disp[mid_y, :],
                'k-', lw=2.0, zorder=6, label='Coupe grand axe (y=0)')
        # Petit axe (x=0)
        ax.plot(np.zeros(n),
                y_fine/1000,
                Z_disp[:, mid_x],
                'k--', lw=2.0, zorder=6, label='Coupe petit axe (x=0)')

    # Point central
    ax.scatter([0], [0], [e0 + w_max * z_factor],
               color='#e74c3c', s=100, zorder=10)
    ax.text(0, b_eff/1000 * 0.25,
            e0 + w_max * z_factor * 1.02,
            f'  w={w_max:.1f}µm', color='#e74c3c', fontsize=9)

    # Flèche verticale au centre
    ax.plot([0, 0], [0, 0],
            [e0, e0 + w_max * z_factor],
            'r-', lw=1.5, alpha=0.7)

    # Axes et mise en forme
    ax.set_xlabel('x — Grand axe (mm)', fontsize=9, labelpad=8)
    ax.set_ylabel('y — Petit axe (mm)',  fontsize=9, labelpad=8)
    ax.set_zlabel(f'z (µm × {z_factor:.1f})', fontsize=9, labelpad=8)

    ax.set_xlim(-a_eff/1000, a_eff/1000)
    ax.set_ylim(-b_eff/1000, b_eff/1000)
    ax.set_zlim(0, (e0 + w_max * z_factor) * 1.15)

    # Proportions xy réelles (matplotlib 3.3+)
    try:
        ax.set_box_aspect([2*a_eff, 2*b_eff,
                           (e0 + w_max*z_factor) * (2*a_eff / (2*a_eff))])
    except AttributeError:
        pass

    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=CFG['elev'], azim=CFG['azim'])

    # Annotation rapport d'aspect réel
    ratio_zx = w_max / (2 * a_eff) * 100
    ratio_zy = w_max / (2 * b_eff) * 100
    ax.text2D(0.02, 0.02,
              f'Rapport réel  w/2a = {ratio_zx:.2f}%   w/2b = {ratio_zy:.2f}%\n'
              f'(sans exagération)',
              transform=ax.transAxes, fontsize=8, color='#555555')

    plt.tight_layout()

    if CFG['save_png']:
        fname = f'nappe_3D_dP{dp_mbar:.0f}mbar_zx{z_factor:.0f}.png'
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"  → Sauvegardé : {fpath}")

    plt.show()
    plt.close()


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # ── Sélection du fichier Excel ────────────────────────────────────────
    if CFG['excel_file'] and os.path.isfile(CFG['excel_file']):
        xl_path = CFG['excel_file']
    else:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        xl_path = filedialog.askopenfilename(
            title="Sélectionner analyse_biaxiale.xlsx",
            filetypes=[('Excel files', '*.xlsx'), ('All files', '*.*')],
            initialdir=os.path.expanduser('~')
        )
        root.destroy()
        if not xl_path:
            print("Aucun fichier sélectionné. Abandon."); return

    out_dir = os.path.dirname(xl_path)
    print(f"\n=== VIEWER 3D ===")
    print(f"  Fichier : {xl_path}")

    # ── Lecture de l'Excel ────────────────────────────────────────────────
    try:
        meta, sheets = parse_excel(xl_path)
    except Exception as e:
        print(f"  Erreur lecture Excel : {e}"); return

    if not sheets:
        print("  Aucune donnée trouvée."); return

    print(f"\n  Paramètres géométriques extraits :")
    print(f"    a_eff = {meta.get('a_eff','?')} µm")
    print(f"    b_eff = {meta.get('b_eff','?')} µm")
    print(f"    e0    = {meta.get('e0','?')} µm")

    # ── Boucle de visualisation (multi-pression possible) ─────────────────
    while True:
        dp_chosen, z_scale_str = choose_pressure(list(sheets.keys()))

        # Mettre à jour CFG avec le z_scale choisi
        CFG['z_scale'] = z_scale_str

        plot_3d(meta, sheets, dp_chosen, z_scale_str, out_dir)

        # Proposer une autre pression
        root2 = tk.Tk(); root2.withdraw(); root2.attributes('-topmost', True)
        again = tk.messagebox.askyesno(
            "Continuer ?",
            "Afficher une autre pression / exagération ?",
            parent=root2
        )
        root2.destroy()
        if not again:
            break

    print("\n  Terminé.")


# =============================================================================
# LANCEMENT — F5 dans Pyzo
# =============================================================================
main()