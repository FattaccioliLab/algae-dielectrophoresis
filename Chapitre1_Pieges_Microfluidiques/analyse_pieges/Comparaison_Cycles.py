"""
=============================================================================
COMPARAISON DES CYCLES DE CAPTURE / LIBERATION  v1.0
=============================================================================
Prend en entree l'Excel (_analyse.xlsx) et compare 7 cycles entre eux.

Interface :
  - Affiche la dynamique de remplissage totale (timeline complete)
  - 7 lignes : debut + fin (en secondes) de chaque cycle
  - La frame de liberation = point du MAX de chaque cycle (auto)

Sorties (dossier "comparaison_cycles" a cote de l'Excel) :
  1. Timeline complete annotee avec les 7 cycles
  2. Les 7 dynamiques superposees (alignees a t=0 relatif)
  3. Courbe moyenne + ecart-type (moyenne jusqu'au max de chaque cycle)
  4. Histogramme capturees vs liberees (max - reste apres liberation)
  5. Histogramme des vitesses de remplissage (particules/seconde)
  6. Boxplot comparatif des cycles
  7. Excel recapitulatif (donnees alignees + statistiques par cycle)

Frame rate : 20 fps

Dependances :
    pip install openpyxl numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import os
import sys

FPS       = 20.0
N_CYCLES  = 4


# Palette de 7 couleurs distinctes pour les cycles
CYCLE_COLORS = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6',
]


# =============================================================================
# CHARGEMENT TIMELINE
# =============================================================================

def load_timeline(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb['Timeline']
    rows = list(ws.iter_rows(values_only=True))[1:]
    frames, temps, totaux, oof = [], [], [], []
    for r in rows:
        if r[0] is None:
            continue
        frames.append(int(r[0]))
        temps.append(float(r[1]))
        totaux.append(int(r[2]))
        oof.append(str(r[3]).upper() == "OUI")
    return (np.array(frames), np.array(temps),
            np.array(totaux), np.array(oof))


# =============================================================================
# EXTRACTION D'UN CYCLE
# =============================================================================

def extract_cycle(temps, totaux, t_start, t_end):
    """
    Extrait un cycle sur la plage [t_start, t_end].

    Retourne un dict avec :
      t_rel      : temps relatif (t - t_start), commence a 0
      total      : nombre de particules
      t_abs      : temps absolu
      idx_max    : index du maximum (= liberation auto)
      t_max_rel  : temps relatif du max
      n_max      : nombre max de particules (pic de capture)
      n_end      : nombre a la fin (apres liberation)
      n_start    : nombre au debut
      captured   : particules capturees = n_max - n_start
      released   : particules liberees = n_max - n_end
      fill_rate  : vitesse de remplissage moyenne (particules/s)
                   = (n_max - n_start) / (t_max_rel) si t_max_rel > 0
    """
    mask = (temps >= t_start) & (temps <= t_end)
    t_abs = temps[mask]
    tot   = totaux[mask].astype(float)
    if len(t_abs) < 2:
        return None

    t_rel = t_abs - t_abs[0]
    idx_max = int(np.argmax(tot))

    n_max   = float(tot[idx_max])
    n_start = float(tot[0])
    n_end   = float(tot[-1])
    t_max_rel = float(t_rel[idx_max])

    captured = n_max - n_start
    released = n_max - n_end
    fill_rate = captured / t_max_rel if t_max_rel > 0 else 0.0

    return dict(
        t_rel=t_rel, total=tot, t_abs=t_abs,
        idx_max=idx_max, t_max_rel=t_max_rel,
        n_max=n_max, n_end=n_end, n_start=n_start,
        captured=captured, released=released, fill_rate=fill_rate,
    )


# =============================================================================
# COURBE MOYENNE (jusqu'au max de chaque cycle)
# =============================================================================

def compute_mean_curve(cycles, dt=1.0/FPS):
    """
    Calcule la courbe moyenne + ecart-type des phases de CAPTURE
    (de t=0 jusqu'au max de chaque cycle).

    Methode :
      1. Pour chaque cycle, ne garder que la phase de capture (0 -> idx_max)
      2. Reechantillonner chaque phase sur une grille de temps commune
         (pas = dt), par interpolation lineaire
      3. Moyenner et calculer l'ecart-type point par point
         (seuls les cycles encore "actifs" a un temps t contribuent)

    Retourne : t_grid, mean, std, n_contrib (nb de cycles par point)
    """
    # Duree max des phases de capture
    max_tcap = max(c['t_max_rel'] for c in cycles if c is not None)
    t_grid = np.arange(0, max_tcap + dt, dt)

    # Matrice (n_cycles, n_points) remplie de NaN
    resampled = np.full((len(cycles), len(t_grid)), np.nan)

    for i, c in enumerate(cycles):
        if c is None:
            continue
        # Phase de capture uniquement
        t_cap   = c['t_rel'][:c['idx_max']+1]
        tot_cap = c['total'][:c['idx_max']+1]
        if len(t_cap) < 2:
            continue
        # Interpoler sur la grille, uniquement dans la plage du cycle
        valid = t_grid <= t_cap[-1]
        resampled[i, valid] = np.interp(t_grid[valid], t_cap, tot_cap)

    mean = np.nanmean(resampled, axis=0)
    std  = np.nanstd(resampled, axis=0)
    n_contrib = np.sum(~np.isnan(resampled), axis=0)

    return t_grid, mean, std, n_contrib, resampled


# =============================================================================
# GENERATION DES FIGURES
# =============================================================================

def fig_timeline_annotated(temps, totaux, oof, cycles, ranges, out_dir):
    """Timeline complete avec les 7 cycles surlignes."""
    fig, ax = plt.subplots(figsize=(15, 5), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(temps, totaux, color='#333333', lw=1.0, alpha=0.6, zorder=1)

    for i, (c, (ts, te)) in enumerate(zip(cycles, ranges)):
        if c is None:
            continue
        col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        ax.axvspan(ts, te, color=col, alpha=0.12, zorder=0)
        ax.plot(c['t_abs'], c['total'], color=col, lw=1.8,
                label=f"Cycle {i+1}", zorder=2)
        # Marquer le max (liberation)
        t_max_abs = c['t_abs'][c['idx_max']]
        ax.plot(t_max_abs, c['n_max'], 'v', color=col, markersize=9,
                markeredgecolor='black', zorder=3)

    ax.set_xlabel("Temps (s)", fontsize=11)
    ax.set_ylabel("Particules totales piegees", fontsize=11)
    ax.set_title("Dynamique de remplissage totale — 7 cycles (v = max/liberation)",
                 fontsize=13)
    ax.legend(fontsize=8, ncol=7, loc='upper center')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "1_timeline_annotee.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_superposition(cycles, out_dir):
    """Les 7 dynamiques alignees a t=0 relatif."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    for i, c in enumerate(cycles):
        if c is None:
            continue
        col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        ax.plot(c['t_rel'], c['total'], color=col, lw=1.6,
                label=f"Cycle {i+1}", alpha=0.85)
        ax.plot(c['t_max_rel'], c['n_max'], 'v', color=col,
                markersize=8, markeredgecolor='black')

    ax.set_xlabel("Temps relatif depuis le debut du cycle (s)", fontsize=11)
    ax.set_ylabel("Particules totales piegees", fontsize=11)
    ax.set_title("Superposition des 7 cycles (alignes a t=0)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "2_superposition.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_mean_std(t_grid, mean, std, n_contrib, cycles, out_dir):
    """Courbe moyenne + bande d'ecart-type (phase de capture)."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')

    # Cycles individuels en gris clair (phase capture)
    for i, c in enumerate(cycles):
        if c is None:
            continue
        t_cap   = c['t_rel'][:c['idx_max']+1]
        tot_cap = c['total'][:c['idx_max']+1]
        ax.plot(t_cap, tot_cap, color='#cccccc', lw=0.9, zorder=1)

    # Bande ecart-type
    ax.fill_between(t_grid, mean - std, mean + std,
                    color='#4363d8', alpha=0.25, zorder=2,
                    label='+/- 1 ecart-type')
    # Courbe moyenne
    ax.plot(t_grid, mean, color='#4363d8', lw=2.5, zorder=3,
            label='Moyenne (phase de capture)')

    ax.set_xlabel("Temps relatif depuis le debut du cycle (s)", fontsize=11)
    ax.set_ylabel("Particules totales piegees", fontsize=11)
    ax.set_title("Dynamique moyenne de capture +/- ecart-type\n"
                 "(moyenne jusqu'au max de chaque cycle)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "3_moyenne_ecarttype.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_capture_release(cycles, out_dir):
    """Histogramme : capturees vs liberees par cycle."""
    idx = [i for i, c in enumerate(cycles) if c is not None]
    captured = [cycles[i]['captured'] for i in idx]
    released = [cycles[i]['released'] for i in idx]
    labels = [f"C{i+1}" for i in idx]

    x = np.arange(len(idx));  w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    ax.set_facecolor('white')
    b1 = ax.bar(x - w/2, captured, w, label='Capturees (max - debut)',
                color='#3cb44b', edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + w/2, released, w, label='Liberees (max - fin)',
                color='#e6194B', edgecolor='black', linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{int(bar.get_height())}", ha='center', va='bottom',
                    fontsize=9)
    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_ylabel("Nombre de particules", fontsize=11)
    ax.set_title("Particules capturees vs liberees par cycle", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "4_capture_vs_liberation.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_fill_rate(cycles, out_dir):
    """Histogramme des vitesses de remplissage (particules/seconde)."""
    idx = [i for i, c in enumerate(cycles) if c is not None]
    rates = [cycles[i]['fill_rate'] for i in idx]
    labels = [f"C{i+1}" for i in idx]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    ax.set_facecolor('white')
    bars = ax.bar(labels, rates, color=[CYCLE_COLORS[i % 7] for i in idx],
                  edgecolor='black', linewidth=0.6)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"{r:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    mean_rate = np.mean(rates)
    ax.axhline(mean_rate, color='red', ls='--', lw=1.5,
               label=f"Moyenne = {mean_rate:.2f} part/s")
    ax.set_ylabel("Vitesse de remplissage (particules / s)", fontsize=11)
    ax.set_title("Vitesse de remplissage moyenne par cycle\n"
                 "(capturees / temps jusqu'au max)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "5_vitesse_remplissage.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_boxplot(cycles, out_dir):
    """Boxplot comparatif : distribution des valeurs de chaque cycle."""
    idx = [i for i, c in enumerate(cycles) if c is not None]
    data = [cycles[i]['total'] for i in idx]
    labels = [f"C{i+1}" for i in idx]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    ax.set_facecolor('white')
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, i in zip(bp['boxes'], idx):
        patch.set_facecolor(CYCLE_COLORS[i % 7])
        patch.set_alpha(0.6)
    ax.set_ylabel("Particules totales (distribution)", fontsize=11)
    ax.set_title("Distribution des valeurs par cycle (boxplot)", fontsize=13)
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "6_boxplot.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


# =============================================================================
# EXPORT EXCEL RECAPITULATIF
# =============================================================================

def export_excel(cycles, ranges, t_grid, mean, std, n_contrib, out_dir):
    wb = openpyxl.Workbook()

    # ── Feuille 1 : Statistiques par cycle ───────────────────────────────────
    ws = wb.active
    ws.title = "Stats_cycles"
    headers = ["Cycle", "Debut (s)", "Fin (s)", "Duree (s)",
               "t_max_rel (s)", "N debut", "N max", "N fin",
               "Capturees", "Liberees", "Vitesse (part/s)",
               "Retention (%)"]
    for ci, h in enumerate(headers, 1):
        cell = ws.cell(1, ci, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E75B6")
        cell.alignment = Alignment(horizontal="center")

    ri = 2
    for i, (c, (ts, te)) in enumerate(zip(cycles, ranges)):
        if c is None:
            continue
        retention = 100 * c['n_end'] / c['n_max'] if c['n_max'] > 0 else 0
        vals = [f"Cycle {i+1}", round(ts,2), round(te,2), round(te-ts,2),
                round(c['t_max_rel'],2), int(c['n_start']), int(c['n_max']),
                int(c['n_end']), int(c['captured']), int(c['released']),
                round(c['fill_rate'],3), round(retention,1)]
        for ci, v in enumerate(vals, 1):
            ws.cell(ri, ci, v)
        ri += 1
    for ci in range(1, len(headers)+1):
        ws.column_dimensions[get_column_letter(ci)].width = 14

    # ── Feuille 2 : Courbe moyenne ───────────────────────────────────────────
    ws2 = wb.create_sheet("Courbe_moyenne")
    for ci, h in enumerate(["Temps rel (s)", "Moyenne", "Ecart-type",
                             "Nb cycles contrib"], 1):
        cell = ws2.cell(1, ci, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E75B6")
    for ri2, (t, m, s, n) in enumerate(zip(t_grid, mean, std, n_contrib), 2):
        ws2.cell(ri2, 1, round(float(t), 4))
        ws2.cell(ri2, 2, round(float(m), 3) if not np.isnan(m) else None)
        ws2.cell(ri2, 3, round(float(s), 3) if not np.isnan(s) else None)
        ws2.cell(ri2, 4, int(n))
    for ci in range(1, 5):
        ws2.column_dimensions[get_column_letter(ci)].width = 18

    # ── Feuilles 3+ : donnees brutes alignees par cycle ──────────────────────
    for i, c in enumerate(cycles):
        if c is None:
            continue
        ws_c = wb.create_sheet(f"Cycle_{i+1}"[:31])
        for ci, h in enumerate(["Temps rel (s)", "Temps abs (s)",
                                 "Particules"], 1):
            cell = ws_c.cell(1, ci, h)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="2E75B6")
        for ri3, (tr, ta, tot) in enumerate(
                zip(c['t_rel'], c['t_abs'], c['total']), 2):
            ws_c.cell(ri3, 1, round(float(tr), 4))
            ws_c.cell(ri3, 2, round(float(ta), 4))
            ws_c.cell(ri3, 3, int(tot))
        for ci in range(1, 4):
            ws_c.column_dimensions[get_column_letter(ci)].width = 15

    path = os.path.join(out_dir, "comparaison_cycles.xlsx")
    wb.save(path)
    print(f"  -> {path}")


# =============================================================================
# INTERFACE
# =============================================================================

class App:

    def __init__(self):
        self.excel_path = None
        self.out_dir = None
        self.frames = self.temps = self.totaux = self.oof = None

        self.root = tk.Tk()
        self.root.title("Comparaison des cycles")
        self.root.geometry("560x560")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        tk.Label(self.root, text="Comparaison des 7 cycles",
                 font=("Helvetica", 15, "bold"),
                 bg="white", fg="#1a5fa8").pack(pady=14)

        tk.Button(self.root, text="Charger l'Excel (_analyse.xlsx)",
                  command=self._load_excel, width=36,
                  font=("Helvetica", 10), bg="#4a90d9", fg="white",
                  relief="flat", padx=8, pady=6).pack(pady=4)
        self.lbl = tk.Label(self.root, text="(aucun fichier)", bg="white",
                            fg="#888888", font=("Helvetica", 9))
        self.lbl.pack()

        tk.Button(self.root, text="Afficher la dynamique totale",
                  command=self._show_timeline, width=36,
                  font=("Helvetica", 9), bg="#eeeeee", fg="#333333",
                  relief="flat", padx=8, pady=5).pack(pady=4)

        # Tableau des 7 cycles
        tframe = tk.Frame(self.root, bg="white")
        tframe.pack(pady=10)
        tk.Label(tframe, text="Cycle", bg="white", font=("Helvetica", 9, "bold"),
                 width=8).grid(row=0, column=0)
        tk.Label(tframe, text="Debut (s)", bg="white", font=("Helvetica", 9, "bold"),
                 width=12).grid(row=0, column=1)
        tk.Label(tframe, text="Fin (s)", bg="white", font=("Helvetica", 9, "bold"),
                 width=12).grid(row=0, column=2)

        self.v_starts = []
        self.v_ends   = []
        for i in range(N_CYCLES):
            tk.Label(tframe, text=f"Cycle {i+1}", bg="white",
                     fg=CYCLE_COLORS[i], font=("Helvetica", 10, "bold"),
                     width=8).grid(row=i+1, column=0, pady=2)
            vs = tk.DoubleVar(value=0.0)
            ve = tk.DoubleVar(value=0.0)
            tk.Spinbox(tframe, from_=0, to=100000, textvariable=vs,
                       width=10, increment=1.0, format="%.1f",
                       font=("Helvetica", 10)).grid(row=i+1, column=1, padx=4)
            tk.Spinbox(tframe, from_=0, to=100000, textvariable=ve,
                       width=10, increment=1.0, format="%.1f",
                       font=("Helvetica", 10)).grid(row=i+1, column=2, padx=4)
            self.v_starts.append(vs)
            self.v_ends.append(ve)

        self.btn_go = tk.Button(self.root, text="Generer la comparaison  ->",
                                command=self._process, width=36,
                                font=("Helvetica", 11, "bold"),
                                bg="#5cb85c", fg="white", relief="flat",
                                padx=8, pady=8, state="disabled")
        self.btn_go.pack(pady=12)

        self.root.mainloop()

    def _load_excel(self):
        path = filedialog.askopenfilename(
            title="Excel", filetypes=[("Excel","*.xlsx"),("Tous","*.*")])
        if not path:
            return
        try:
            self.frames, self.temps, self.totaux, self.oof = load_timeline(path)
            self.excel_path = path
            self.out_dir = os.path.join(os.path.dirname(path), "comparaison_cycles")
            os.makedirs(self.out_dir, exist_ok=True)
            self.lbl.config(
                text=f"OK : {len(self.temps)} points, "
                     f"{self.temps.min():.0f}-{self.temps.max():.0f} s",
                fg="#2a8a35")
            self.btn_go.config(state="normal")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _show_timeline(self):
        if self.temps is None:
            messagebox.showwarning("", "Chargez d'abord l'Excel.");  return
        plt.figure(figsize=(14, 5))
        plt.plot(self.temps, self.totaux, color='steelblue', lw=1.2)
        plt.xlabel("Temps (s)");  plt.ylabel("Particules totales")
        plt.title("Dynamique de remplissage totale — reperez vos 7 cycles")
        plt.grid(True, alpha=0.2);  plt.tight_layout()
        plt.show(block=False)

    def _process(self):
        # Extraire les cycles
        cycles = []
        ranges = []
        for i in range(N_CYCLES):
            ts = self.v_starts[i].get()
            te = self.v_ends[i].get()
            if te <= ts:
                cycles.append(None);  ranges.append((ts, te))
                continue
            c = extract_cycle(self.temps, self.totaux, ts, te)
            cycles.append(c);  ranges.append((ts, te))

        valid = [c for c in cycles if c is not None]
        if len(valid) < 2:
            messagebox.showwarning("", "Renseignez au moins 2 cycles valides "
                                       "(fin > debut).")
            return

        print(f"\n[Comparaison -> {self.out_dir}]")
        print(f"  {len(valid)} cycles valides")

        # Courbe moyenne
        t_grid, mean, std, n_contrib, resampled = compute_mean_curve(valid)

        # Toutes les figures
        fig_timeline_annotated(self.temps, self.totaux, self.oof,
                               cycles, ranges, self.out_dir)
        fig_superposition(cycles, self.out_dir)
        fig_mean_std(t_grid, mean, std, n_contrib, cycles, self.out_dir)
        fig_capture_release(cycles, self.out_dir)
        fig_fill_rate(cycles, self.out_dir)
        fig_boxplot(cycles, self.out_dir)

        # Excel
        export_excel(cycles, ranges, t_grid, mean, std, n_contrib, self.out_dir)

        messagebox.showinfo("Termine",
            f"{len(valid)} cycles compares.\n"
            f"6 figures SVG + 1 Excel generes dans :\n{self.out_dir}")


# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Comparaison des cycles de capture/liberation")
    print("=" * 60)
    for mod, pkg in {'openpyxl':'openpyxl','numpy':'numpy',
                     'matplotlib':'matplotlib'}.items():
        try:
            __import__(mod);  print(f"  [OK]  {mod}")
        except ImportError:
            print(f"  [MANQUANT]  {mod} -> pip install {pkg}");  sys.exit(1)
    print("\n  Demarrage...\n")
    App()


    ###



"""
=============================================================================
COMPARAISON DES PENTES DE CAPTURE  v1.0
=============================================================================
Reprend l'Excel "comparaison_cycles.xlsx" (feuilles Cycle_1 ... Cycle_N,
chacune avec : Temps rel (s), Temps abs (s), Particules).

Objectif : comparer les DYNAMIQUES (pentes) de capture entre cycles.

Chaque courbe :
  - part de 0 en Y      (on soustrait la valeur de depart N_start)
  - part de 0 en X      (temps relatif, deja le cas)
  - s'arrete au PREMIER MAXIMUM  (premier pic avant la premiere chute)

Ainsi toutes les phases de montee sont comparables directement.

Sorties (dossier "comparaison_pentes" a cote de l'Excel) :
  1. Superposition des montees (0 -> premier max), normalisees a 0 en Y
  2. Idem + regression lineaire par cycle (pente = vitesse reelle de montee)
  3. Histogramme des pentes : 2 methodes comparees
       - pente moyenne  = (Nmax - Nstart) / t_premier_max     [ancien calcul]
       - pente lineaire = coefficient directeur de la regression lineaire
  4. Excel recapitulatif des pentes

Dependances : pip install openpyxl numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import os
import sys

CYCLE_COLORS = ['#e6194B', '#3cb44b', '#4363d8', '#f58231',
                '#911eb4', '#42d4f4', '#f032e6', '#808000', '#000075']


# =============================================================================
# CHARGEMENT DES FEUILLES CYCLE_X
# =============================================================================

def load_cycles(path):
    """
    Charge toutes les feuilles Cycle_X.
    Retourne une liste de dicts : {name, t_rel, total}
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    cycles = []
    # Trier les feuilles Cycle_1, Cycle_2, ... dans l'ordre numerique
    cycle_sheets = [s for s in wb.sheetnames if s.startswith('Cycle_')]
    cycle_sheets.sort(key=lambda s: int(s.split('_')[1]))

    for sh in cycle_sheets:
        ws = wb[sh]
        rows = list(ws.iter_rows(values_only=True))[1:]  # sauter entete
        t_rel, total = [], []
        for r in rows:
            if r[0] is None:
                continue
            t_rel.append(float(r[0]))
            total.append(float(r[2]))
        if len(t_rel) < 2:
            continue
        cycles.append(dict(name=sh, t_rel=np.array(t_rel),
                           total=np.array(total)))
    return cycles


# =============================================================================
# DETECTION DU PREMIER MAXIMUM
# =============================================================================

def first_maximum(total, drop_frac=0.30):
    """
    Trouve l'index du PREMIER maximum : le premier pic suivi d'une chute
    significative (>= drop_frac de la valeur du pic).

    Pourquoi pas juste argmax ? Parce que le maximum GLOBAL peut arriver
    tard (ex : cycle 7 monte tres haut). On veut le premier vrai sommet,
    celui juste avant la premiere liberation.

    Methode :
      - on parcourt la courbe
      - on suit le maximum courant
      - des que la valeur retombe sous (1 - drop_frac) * max_courant,
        on considere que le premier max etait atteint -> on s'arrete
    """
    n = len(total)
    run_max = total[0]
    run_max_idx = 0
    for i in range(1, n):
        if total[i] > run_max:
            run_max = total[i]
            run_max_idx = i
        elif total[i] < run_max * (1 - drop_frac):
            # chute significative apres le pic -> premier max confirme
            return run_max_idx
    # Pas de chute detectee : le max global est le premier max
    return int(np.argmax(total))


# =============================================================================
# EXTRACTION DE LA PHASE DE MONTEE + PENTES
# =============================================================================

def extract_rise(cycle, drop_frac=0.30):
    """
    Extrait la phase de montee 0 -> premier max, normalisee a 0 en Y.

    Retourne un dict enrichi :
      t_rise    : temps 0 -> t_premier_max
      y_rise    : particules - N_start  (part de 0)
      idx_max   : index du premier max
      t_max     : temps du premier max
      n_start   : valeur de depart
      n_max     : valeur au premier max
      captured  : n_max - n_start
      slope_mean: pente moyenne = captured / t_max
      slope_lin : pente par regression lineaire (moindres carres)
      r2        : qualite du fit lineaire (coefficient de determination)
    """
    t   = cycle['t_rel']
    tot = cycle['total']

    idx_max = first_maximum(tot, drop_frac)
    t_rise  = t[:idx_max+1] - t[0]
    n_start = tot[0]
    y_rise  = tot[:idx_max+1] - n_start   # normalise a 0 en Y

    n_max   = tot[idx_max]
    t_max   = t_rise[-1]
    captured = n_max - n_start

    # Pente moyenne (ancien calcul)
    slope_mean = captured / t_max if t_max > 0 else 0.0

    # Pente par regression lineaire y = a*t + b (on force pas b=0,
    # mais y part deja de ~0). On prend a comme pente reelle de montee.
    if len(t_rise) >= 2:
        a, b = np.polyfit(t_rise, y_rise, 1)
        y_pred = a * t_rise + b
        ss_res = np.sum((y_rise - y_pred) ** 2)
        ss_tot = np.sum((y_rise - np.mean(y_rise)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        a, b, r2 = 0.0, 0.0, 0.0

    return dict(
        name=cycle['name'], t_rise=t_rise, y_rise=y_rise,
        idx_max=idx_max, t_max=t_max, n_start=n_start, n_max=n_max,
        captured=captured, slope_mean=slope_mean,
        slope_lin=a, intercept=b, r2=r2,
    )


# =============================================================================
# FIGURES
# =============================================================================

def fig_rise_superposition(rises, out_dir):
    """Toutes les montees superposees, partant de (0,0), coupees au 1er max."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    for i, r in enumerate(rises):
        col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        label = f"{r['name'].replace('_',' ')} (pente={r['slope_lin']:.2f}/s)"
        ax.plot(r['t_rise'], r['y_rise'], color=col, lw=2, label=label)
        # Marquer le premier max
        ax.plot(r['t_max'], r['captured'], 'o', color=col,
                markersize=8, markeredgecolor='black')

    ax.set_xlabel("Temps depuis le debut de la montee (s)", fontsize=11)
    ax.set_ylabel("Particules capturees (depart a 0)", fontsize=11)
    ax.set_title("Montees de capture superposees\n"
                 "(depart 0,0 -> premier maximum) — comparaison des pentes",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "1_montees_superposees.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_rise_with_fits(rises, out_dir):
    """Montees + droites de regression lineaire."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    for i, r in enumerate(rises):
        col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        ax.plot(r['t_rise'], r['y_rise'], color=col, lw=1.4, alpha=0.55)
        # Droite de fit
        t_fit = np.array([0, r['t_max']])
        y_fit = r['slope_lin'] * t_fit + r['intercept']
        ax.plot(t_fit, y_fit, color=col, lw=2.5, ls='--',
                label=f"{r['name'].replace('_',' ')}: "
                      f"{r['slope_lin']:.2f}/s (R2={r['r2']:.2f})")

    ax.set_xlabel("Temps depuis le debut de la montee (s)", fontsize=11)
    ax.set_ylabel("Particules capturees (depart a 0)", fontsize=11)
    ax.set_title("Regression lineaire des montees (pente = vitesse reelle)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "2_montees_regression.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


def fig_slopes_comparison(rises, out_dir):
    """Histogramme comparant pente moyenne vs pente lineaire."""
    labels = [r['name'].replace('Cycle_', 'C') for r in rises]
    slopes_mean = [r['slope_mean'] for r in rises]
    slopes_lin  = [r['slope_lin']  for r in rises]

    x = np.arange(len(rises));  w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    b1 = ax.bar(x - w/2, slopes_mean, w,
                label='Pente moyenne (Nmax-Nstart)/t_max',
                color='#f58231', edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + w/2, slopes_lin, w,
                label='Pente reelle (regression lineaire)',
                color='#4363d8', edgecolor='black', linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha='center', va='bottom',
                    fontsize=9)

    m_mean = np.mean(slopes_mean);  m_lin = np.mean(slopes_lin)
    ax.axhline(m_lin, color='#4363d8', ls=':', lw=1.2, alpha=0.7)

    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_ylabel("Pente de capture (particules / s)", fontsize=11)
    ax.set_title("Comparaison des pentes : moyenne vs regression lineaire\n"
                 f"(moyenne lineaire globale = {m_lin:.2f} part/s)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "3_comparaison_pentes.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


# =============================================================================
# EXPORT EXCEL
# =============================================================================

def export_excel(rises, out_dir):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Pentes"
    headers = ["Cycle", "N debut", "N premier max", "Capturees",
               "t premier max (s)", "Pente moyenne (part/s)",
               "Pente lineaire (part/s)", "R2 fit", "Ecart (%)"]
    for ci, h in enumerate(headers, 1):
        cell = ws.cell(1, ci, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E75B6")
        cell.alignment = Alignment(horizontal="center")

    for ri, r in enumerate(rises, 2):
        ecart = (100 * (r['slope_mean'] - r['slope_lin']) / r['slope_lin']
                 if r['slope_lin'] else 0)
        vals = [r['name'].replace('_', ' '), int(r['n_start']),
                int(r['n_max']), int(r['captured']), round(r['t_max'], 2),
                round(r['slope_mean'], 3), round(r['slope_lin'], 3),
                round(r['r2'], 3), round(ecart, 1)]
        for ci, v in enumerate(vals, 1):
            ws.cell(ri, ci, v)
    for ci in range(1, len(headers)+1):
        ws.column_dimensions[get_column_letter(ci)].width = 18

    # Feuille avec les courbes de montee brutes
    ws2 = wb.create_sheet("Montees_brutes")
    col = 1
    for r in rises:
        ws2.cell(1, col, f"{r['name']} t(s)").font = Font(bold=True)
        ws2.cell(1, col+1, f"{r['name']} capt").font = Font(bold=True)
        for ri, (t, y) in enumerate(zip(r['t_rise'], r['y_rise']), 2):
            ws2.cell(ri, col, round(float(t), 4))
            ws2.cell(ri, col+1, round(float(y), 1))
        col += 3

    path = os.path.join(out_dir, "comparaison_pentes.xlsx")
    wb.save(path)
    print(f"  -> {path}")


# =============================================================================
# INTERFACE
# =============================================================================

class App:

    def __init__(self):
        self.excel_path = None
        self.out_dir = None
        self.cycles = None

        self.root = tk.Tk()
        self.root.title("Comparaison des pentes de capture")
        self.root.geometry("500x360")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        tk.Label(self.root, text="Comparaison des pentes de capture",
                 font=("Helvetica", 14, "bold"),
                 bg="white", fg="#1a5fa8").pack(pady=16)
        tk.Label(self.root,
                 text="Reprend comparaison_cycles.xlsx (feuilles Cycle_X).\n"
                      "Aligne les montees a (0,0) et coupe au premier maximum.",
                 font=("Helvetica", 9), bg="white", fg="#555555",
                 justify="center").pack()

        tk.Button(self.root, text="Charger l'Excel (Cycle_X)",
                  command=self._load, width=34,
                  font=("Helvetica", 10), bg="#4a90d9", fg="white",
                  relief="flat", padx=8, pady=7).pack(pady=10)
        self.lbl = tk.Label(self.root, text="(aucun fichier)", bg="white",
                            fg="#888888", font=("Helvetica", 9))
        self.lbl.pack()

        # Reglage du seuil de chute
        f = tk.Frame(self.root, bg="white");  f.pack(pady=8)
        tk.Label(f, text="Seuil de chute pour 1er max (%) :",
                 bg="white", font=("Helvetica", 9)).pack(side="left", padx=4)
        self.v_drop = tk.IntVar(value=30)
        tk.Spinbox(f, from_=5, to=90, textvariable=self.v_drop,
                   width=5, increment=5, font=("Helvetica", 10)).pack(side="left")

        self.btn_go = tk.Button(self.root, text="Generer les comparaisons  ->",
                                command=self._process, width=34,
                                font=("Helvetica", 11, "bold"),
                                bg="#5cb85c", fg="white", relief="flat",
                                padx=8, pady=8, state="disabled")
        self.btn_go.pack(pady=14)

        self.root.mainloop()

    def _load(self):
        path = filedialog.askopenfilename(
            title="Excel comparaison_cycles.xlsx",
            filetypes=[("Excel","*.xlsx"),("Tous","*.*")])
        if not path:
            return
        try:
            self.cycles = load_cycles(path)
            if not self.cycles:
                messagebox.showerror("Erreur",
                    "Aucune feuille Cycle_X trouvee dans cet Excel.")
                return
            self.excel_path = path
            self.out_dir = os.path.join(os.path.dirname(path),
                                        "comparaison_pentes")
            os.makedirs(self.out_dir, exist_ok=True)
            self.lbl.config(text=f"OK : {len(self.cycles)} cycles charges",
                            fg="#2a8a35")
            self.btn_go.config(state="normal")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _process(self):
        drop = self.v_drop.get() / 100.0
        rises = [extract_rise(c, drop_frac=drop) for c in self.cycles]

        print(f"\n[Comparaison pentes -> {self.out_dir}]")
        for r in rises:
            print(f"  {r['name']}: 1er max a t={r['t_max']:.1f}s, "
                  f"capt={r['captured']:.0f}, "
                  f"pente_moy={r['slope_mean']:.2f}, "
                  f"pente_lin={r['slope_lin']:.2f} (R2={r['r2']:.2f})")

        fig_rise_superposition(rises, self.out_dir)
        fig_rise_with_fits(rises, self.out_dir)
        fig_slopes_comparison(rises, self.out_dir)
        export_excel(rises, self.out_dir)

        # Afficher la superposition a l'ecran
        plt.figure(figsize=(11, 6))
        for i, r in enumerate(rises):
            col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
            plt.plot(r['t_rise'], r['y_rise'], color=col, lw=2,
                     label=f"{r['name'].replace('_',' ')} ({r['slope_lin']:.2f}/s)")
        plt.xlabel("Temps depuis debut montee (s)")
        plt.ylabel("Particules capturees (depart 0)")
        plt.title("Montees superposees (0,0 -> premier max)")
        plt.legend(fontsize=9);  plt.grid(True, alpha=0.2);  plt.tight_layout()
        plt.show(block=False)

        messagebox.showinfo("Termine",
            f"{len(rises)} cycles compares.\n"
            f"3 figures SVG + 1 Excel dans :\n{self.out_dir}")


# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Comparaison des pentes de capture")
    print("=" * 60)
    for mod, pkg in {'openpyxl':'openpyxl','numpy':'numpy',
                     'matplotlib':'matplotlib'}.items():
        try:
            __import__(mod);  print(f"  [OK]  {mod}")
        except ImportError:
            print(f"  [MANQUANT]  {mod} -> pip install {pkg}");  sys.exit(1)
    print("\n  Demarrage...\n")
    App()


    ###


"""
=============================================================================
PENTE REELLE + DATA EN POINTS  v1.0
=============================================================================
Reprend l'Excel "comparaison_cycles.xlsx" (feuilles Cycle_1 ... Cycle_N).

Pour chaque cycle, sur la phase de montee (0 -> premier max) :
  - les donnees sont tracees comme des POINTS (nuage de mesures)
  - la droite de regression lineaire est superposee (pente reelle)

Sorties (dossier "pente_reelle" a cote de l'Excel) :
  1. Figure globale : tous les cycles (points + droites) superposes
  2. Une figure INDIVIDUELLE par cycle (points + fit + equation + R2)
  3. Excel avec les pentes reelles

Dependances : pip install openpyxl numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import os
import sys

CYCLE_COLORS = ['#e6194B', '#3cb44b', '#4363d8', '#f58231',
                '#911eb4', '#42d4f4', '#f032e6', '#808000', '#000075']


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_cycles(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    cycles = []
    sheets = [s for s in wb.sheetnames if s.startswith('Cycle_')]
    sheets.sort(key=lambda s: int(s.split('_')[1]))
    for sh in sheets:
        ws = wb[sh]
        rows = list(ws.iter_rows(values_only=True))[1:]
        t_rel, total = [], []
        for r in rows:
            if r[0] is None:
                continue
            t_rel.append(float(r[0]))
            total.append(float(r[2]))
        if len(t_rel) >= 2:
            cycles.append(dict(name=sh, t_rel=np.array(t_rel),
                               total=np.array(total)))
    return cycles


def first_maximum(total, drop_frac=0.30):
    """Premier pic suivi d'une chute >= drop_frac du pic."""
    n = len(total)
    run_max = total[0];  run_max_idx = 0
    for i in range(1, n):
        if total[i] > run_max:
            run_max = total[i];  run_max_idx = i
        elif total[i] < run_max * (1 - drop_frac):
            return run_max_idx
    return int(np.argmax(total))


def extract_rise(cycle, drop_frac=0.30):
    """Phase de montee normalisee a 0 en Y + regression lineaire."""
    t = cycle['t_rel'];  tot = cycle['total']
    idx_max = first_maximum(tot, drop_frac)
    t_rise  = t[:idx_max+1] - t[0]
    n_start = tot[0]
    y_rise  = tot[:idx_max+1] - n_start
    n_max   = tot[idx_max]
    t_max   = t_rise[-1]
    captured = n_max - n_start

    if len(t_rise) >= 2:
        a, b = np.polyfit(t_rise, y_rise, 1)
        y_pred = a * t_rise + b
        ss_res = np.sum((y_rise - y_pred) ** 2)
        ss_tot = np.sum((y_rise - np.mean(y_rise)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        a, b, r2 = 0.0, 0.0, 0.0

    return dict(name=cycle['name'], t_rise=t_rise, y_rise=y_rise,
                t_max=t_max, n_start=n_start, n_max=n_max,
                captured=captured, slope=a, intercept=b, r2=r2)


# =============================================================================
# FIGURE GLOBALE : POINTS + DROITES
# =============================================================================

def fig_all_points_fits(rises, out_dir):
    """Tous les cycles : donnees en points + droite de regression."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    for i, r in enumerate(rises):
        col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        # Donnees = points seuls (pas de ligne)
        ax.scatter(r['t_rise'], r['y_rise'], color=col, s=18,
                   alpha=0.55, edgecolors='none', zorder=2)
        # Droite de regression
        t_fit = np.array([0, r['t_max']])
        y_fit = r['slope'] * t_fit + r['intercept']
        ax.plot(t_fit, y_fit, color=col, lw=2.5, zorder=3,
                label=f"{r['name'].replace('_',' ')} : "
                      f"{r['slope']:.2f} part/s  (R2={r['r2']:.3f})")

    ax.set_xlabel("Temps depuis le debut de la montee (s)", fontsize=11)
    ax.set_ylabel("Particules capturees (depart a 0)", fontsize=11)
    ax.set_title("Pente reelle : donnees (points) + regression lineaire (droites)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, "1_tous_points_fits.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


# =============================================================================
# FIGURES INDIVIDUELLES : 1 par cycle
# =============================================================================

def fig_individual(r, index, out_dir):
    """Un cycle : points + droite + equation + R2."""
    col = CYCLE_COLORS[index % len(CYCLE_COLORS)]
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    ax.set_facecolor('white')

    # Donnees = points
    ax.scatter(r['t_rise'], r['y_rise'], color=col, s=32,
               alpha=0.7, edgecolors='black', linewidths=0.4,
               zorder=2, label='Donnees (mesures)')
    # Droite de fit
    t_fit = np.linspace(0, r['t_max'], 100)
    y_fit = r['slope'] * t_fit + r['intercept']
    ax.plot(t_fit, y_fit, color='black', lw=2.2, zorder=3,
            label=f"Fit lineaire : y = {r['slope']:.2f} t + {r['intercept']:.1f}")

    # Encadre avec les infos
    txt = (f"Pente reelle = {r['slope']:.3f} part/s\n"
           f"R2 = {r['r2']:.4f}\n"
           f"Capturees = {r['captured']:.0f}\n"
           f"Duree montee = {r['t_max']:.1f} s")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=10,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=col, alpha=0.9))

    ax.set_xlabel("Temps depuis le debut de la montee (s)", fontsize=11)
    ax.set_ylabel("Particules capturees (depart a 0)", fontsize=11)
    ax.set_title(f"{r['name'].replace('_',' ')} — pente reelle de capture",
                 fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out_dir, f"2_{r['name']}_points_fit.svg")
    fig.savefig(p, bbox_inches='tight', facecolor='white');  plt.close(fig)
    print(f"  -> {p}")


# =============================================================================
# EXPORT EXCEL
# =============================================================================

def export_excel(rises, out_dir):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Pentes_reelles"
    headers = ["Cycle", "Pente reelle (part/s)", "Ordonnee origine",
               "R2", "Capturees", "Duree montee (s)"]
    for ci, h in enumerate(headers, 1):
        cell = ws.cell(1, ci, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E75B6")
        cell.alignment = Alignment(horizontal="center")
    for ri, r in enumerate(rises, 2):
        vals = [r['name'].replace('_', ' '), round(r['slope'], 3),
                round(r['intercept'], 2), round(r['r2'], 4),
                int(r['captured']), round(r['t_max'], 2)]
        for ci, v in enumerate(vals, 1):
            ws.cell(ri, ci, v)
    for ci in range(1, len(headers)+1):
        ws.column_dimensions[get_column_letter(ci)].width = 18

    # Feuille points bruts
    ws2 = wb.create_sheet("Points_bruts")
    col = 1
    for r in rises:
        ws2.cell(1, col, f"{r['name']} t(s)").font = Font(bold=True)
        ws2.cell(1, col+1, f"{r['name']} capt").font = Font(bold=True)
        for ri, (t, y) in enumerate(zip(r['t_rise'], r['y_rise']), 2):
            ws2.cell(ri, col, round(float(t), 4))
            ws2.cell(ri, col+1, round(float(y), 1))
        col += 3

    path = os.path.join(out_dir, "pentes_reelles.xlsx")
    wb.save(path)
    print(f"  -> {path}")


# =============================================================================
# INTERFACE
# =============================================================================

class App:

    def __init__(self):
        self.cycles = None;  self.out_dir = None

        self.root = tk.Tk()
        self.root.title("Pente reelle + points")
        self.root.geometry("480x330")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        tk.Label(self.root, text="Pente reelle + donnees en points",
                 font=("Helvetica", 14, "bold"),
                 bg="white", fg="#1a5fa8").pack(pady=16)
        tk.Label(self.root,
                 text="Trace les mesures comme des points\n"
                      "et superpose la droite de regression (pente reelle).",
                 font=("Helvetica", 9), bg="white", fg="#555555",
                 justify="center").pack()

        tk.Button(self.root, text="Charger l'Excel (Cycle_X)",
                  command=self._load, width=34,
                  font=("Helvetica", 10), bg="#4a90d9", fg="white",
                  relief="flat", padx=8, pady=7).pack(pady=10)
        self.lbl = tk.Label(self.root, text="(aucun fichier)", bg="white",
                            fg="#888888", font=("Helvetica", 9))
        self.lbl.pack()

        f = tk.Frame(self.root, bg="white");  f.pack(pady=8)
        tk.Label(f, text="Seuil de chute pour 1er max (%) :",
                 bg="white", font=("Helvetica", 9)).pack(side="left", padx=4)
        self.v_drop = tk.IntVar(value=30)
        tk.Spinbox(f, from_=5, to=90, textvariable=self.v_drop,
                   width=5, increment=5, font=("Helvetica", 10)).pack(side="left")

        self.btn_go = tk.Button(self.root, text="Generer  ->",
                                command=self._process, width=34,
                                font=("Helvetica", 11, "bold"),
                                bg="#5cb85c", fg="white", relief="flat",
                                padx=8, pady=8, state="disabled")
        self.btn_go.pack(pady=14)
        self.root.mainloop()

    def _load(self):
        path = filedialog.askopenfilename(
            title="Excel comparaison_cycles.xlsx",
            filetypes=[("Excel","*.xlsx"),("Tous","*.*")])
        if not path:
            return
        try:
            self.cycles = load_cycles(path)
            if not self.cycles:
                messagebox.showerror("Erreur", "Aucune feuille Cycle_X trouvee.")
                return
            self.out_dir = os.path.join(os.path.dirname(path), "pente_reelle")
            os.makedirs(self.out_dir, exist_ok=True)
            self.lbl.config(text=f"OK : {len(self.cycles)} cycles", fg="#2a8a35")
            self.btn_go.config(state="normal")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _process(self):
        drop = self.v_drop.get() / 100.0
        rises = [extract_rise(c, drop_frac=drop) for c in self.cycles]

        print(f"\n[Pente reelle -> {self.out_dir}]")
        for r in rises:
            print(f"  {r['name']}: pente={r['slope']:.3f} part/s, R2={r['r2']:.4f}")

        # Figure globale
        fig_all_points_fits(rises, self.out_dir)
        # Figures individuelles
        for i, r in enumerate(rises):
            fig_individual(r, i, self.out_dir)
        # Excel
        export_excel(rises, self.out_dir)

        # Afficher la figure globale a l'ecran
        plt.figure(figsize=(11, 6))
        for i, r in enumerate(rises):
            col = CYCLE_COLORS[i % len(CYCLE_COLORS)]
            plt.scatter(r['t_rise'], r['y_rise'], color=col, s=16, alpha=0.5)
            t_fit = np.array([0, r['t_max']])
            plt.plot(t_fit, r['slope']*t_fit + r['intercept'], color=col, lw=2.2,
                     label=f"{r['name'].replace('_',' ')} ({r['slope']:.2f}/s)")
        plt.xlabel("Temps depuis debut montee (s)")
        plt.ylabel("Particules capturees (depart 0)")
        plt.title("Pente reelle : points + regression")
        plt.legend(fontsize=9);  plt.grid(True, alpha=0.2);  plt.tight_layout()
        plt.show(block=False)

        messagebox.showinfo("Termine",
            f"{len(rises)} cycles.\n"
            f"1 figure globale + {len(rises)} figures individuelles + Excel\n"
            f"dans : {self.out_dir}")


# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Pente reelle + donnees en points")
    print("=" * 60)
    for mod, pkg in {'openpyxl':'openpyxl','numpy':'numpy',
                     'matplotlib':'matplotlib'}.items():
        try:
            __import__(mod);  print(f"  [OK]  {mod}")
        except ImportError:
            print(f"  [MANQUANT]  {mod} -> pip install {pkg}");  sys.exit(1)
    print("\n  Demarrage...\n")
    App()