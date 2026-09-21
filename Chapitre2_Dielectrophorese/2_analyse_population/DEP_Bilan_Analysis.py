"""
DEP_Bilan_Analysis.py
=====================
Reads a folder containing ##CONDITION-Sx-PRE/POST-fMHz##.txt result files,
extracts the redistribution signal (numerator) per frequency, computes the
tanh-normalised DEP index I_DEP per replicate, and generates four summary
figures plus a structured text report.

All outputs are saved to a BILAN/ subfolder inside the selected directory.

Usage: run the script and select the folder containing the .txt files.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import filedialog
from datetime import datetime


# ================================================================
# CONFIGURATION
# ================================================================

FREQ_ORDER = [1, 10, 20, 30, 40, 50]   # analysed frequencies in MHz
ALPHA_TANH = np.arctanh(0.9)            # tanh scaling constant (= 1.4722)
                                         # chosen so that I_DEP saturates at
                                         # ±0.9 when numerator = C_ref_global

plt.rcParams.update({
    'font.size':          11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
})

# Colour palette — consistent across all figures
C_PRE  = '#2166AC'   # blue  — PRE exposure
C_POST = '#D6604D'   # red   — POST exposure


# ================================================================
# STEP 1 — SELECT INPUT FOLDER
# ================================================================

def select_folder():
    """
    Open a directory chooser dialog and return the selected path.
    The folder should contain ##CONDITION-Sx-PRE/POST-fMHz##.txt files.
    """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select folder containing ##...##.txt result files")
    root.destroy()
    return folder


# ================================================================
# STEP 2 — PARSE FILENAME
# ================================================================

def parse_filename(filename):
    # """
    # Extract (condition, sample, exposure, freq_mhz) from the file name.
    # Accepted formats:
    # ##CONDITION-Sx-PRE-fMHz##.txt
    # __CONDITION-Sx-PRE-fMHz__.txt   (alternative delimiter)
    #
    # Returns a metadata dict, or None if the name does not match.
    # """
    base = os.path.splitext(filename)[0]
    base = base.strip('#').strip('_').strip()

    pattern = r'^(TEMOIN|TEST)-(S\d+)-(PRE|POST)-(\d+)MHz$'
    m = re.match(pattern, base, re.IGNORECASE)
    if not m:
        return None
    return {
        'condition': m.group(1).upper(),
        'sample':    m.group(2).upper(),
        'exposure':  m.group(3).upper(),
        'freq':      int(m.group(4)),
    }


# ================================================================
# STEP 3 — PARSE TEXT FILE CONTENT
# ================================================================

def parse_txt_file(filepath):
    """
    Extract numerical values from a per-video result file using regex.

    Parsed fields: numerator, C_center, C_inner, C_outer, C_rings,
    sigma_bruit, C_ref, SNR, DEP_center_local, DEP_inner_local,
    DEP_outer_local.

    Returns a dict, or None if the file cannot be read or the
    critical 'numerator' field is missing.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Cannot read {filepath}: {e}")
        return None

    def extract(pattern, text, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    data = {}
    data['numerator']        = extract(
        r'numerator \(C_center - C_rings\)\s*=\s*([-\d.]+)', content)
    data['C_center']         = extract(r'C_center\s*=\s*([-\d.]+)', content)
    data['C_inner']          = extract(r'C_inner\s*=\s*([-\d.]+)', content)
    data['C_outer']          = extract(r'C_outer\s*=\s*([-\d.]+)', content)
    data['C_rings']          = extract(r'C_rings\s*=\s*([-\d.]+)', content)
    data['sigma_bruit']      = extract(r'sigma_bruit\s*=\s*([\d.]+)', content)
    data['C_ref']            = extract(r'C_ref\s*=\s*([\d.]+)', content)
    data['SNR']              = extract(
        r'SNR = \|numerator\| / sigma\s*=\s*([\d.]+)', content)
    data['DEP_center_local'] = extract(r'Center\s*:\s*([-\d.]+)\s*%', content)
    data['DEP_inner_local']  = extract(r'Inner\s*:\s*([-\d.]+)\s*%', content)
    data['DEP_outer_local']  = extract(r'Outer\s*:\s*([-\d.]+)\s*%', content)

    if data['numerator'] is None:
        print(f"[WARN] numerator not found in {os.path.basename(filepath)}")
        return None

    return data


# ================================================================
# STEP 4 — LOAD ALL FILES FROM FOLDER
# ================================================================

def load_experiment(folder):
    """
    Scan the folder for all valid ##...##.txt files, parse each one,
    and organise the data into a nested dict:

      results[condition][sample][exposure][freq_mhz] = {numerator, ...}

    Unrecognised filenames are skipped and reported.
    """
    results     = {}
    files_found = 0
    files_skip  = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.txt'):
            continue
        meta = parse_filename(filename)
        if meta is None:
            files_skip.append(filename)
            continue

        filepath = os.path.join(folder, filename)
        parsed   = parse_txt_file(filepath)
        if parsed is None:
            continue

        c, s, e, f = (meta['condition'], meta['sample'],
                      meta['exposure'],  meta['freq'])
        results.setdefault(c, {})
        results[c].setdefault(s, {})
        results[c][s].setdefault(e, {})
        results[c][s][e][f] = parsed
        files_found += 1
        print(f"  [OK] {filename:<48}  numerator = {parsed['numerator']:+.5f}")

    if files_skip:
        print(f"\n  [SKIP] {len(files_skip)} unrecognized file(s): {files_skip}")

    print(f"\n  => {files_found} files loaded.\n")
    return results


# ================================================================
# STEP 5 — COMPUTE I_DEP PER REPLICATE
# ================================================================

def compute_I_dep(results, condition):
    """
    For each replicate and each exposure condition (PRE / POST):
      1. Collect the numerator values across the 6 frequencies.
      2. Compute C_ref_global = max|numerator| over the 6 frequencies
         (specific to each replicate × exposure combination).
      3. Apply I_DEP = tanh(alpha * numerator / C_ref_global).

    Missing frequency data is replaced by NaN and a warning is printed.

    Returns
    -------
    I_dep[sample][exposure]      : np.array of shape (6,)
    numerators[sample][exposure] : np.array of shape (6,)
    """
    cond_data  = results.get(condition, {})
    I_dep      = {}
    numerators = {}

    for sample in sorted(cond_data.keys()):
        I_dep[sample]      = {}
        numerators[sample] = {}

        for exposure in ['PRE', 'POST']:
            if exposure not in cond_data[sample]:
                print(f"  [WARN] {condition}-{sample}: no {exposure} data")
                continue

            freq_data = cond_data[sample][exposure]
            nums = []
            for f in FREQ_ORDER:
                if f in freq_data and freq_data[f]['numerator'] is not None:
                    nums.append(freq_data[f]['numerator'])
                else:
                    print(f"  [WARN] Missing {condition}-{sample}-{exposure}-{f}MHz")
                    nums.append(np.nan)

            nums    = np.array(nums, dtype=float)
            valid   = nums[~np.isnan(nums)]
            C_ref_g = np.max(np.abs(valid)) if len(valid) > 0 else 1.0

            numerators[sample][exposure] = nums
            I_dep[sample][exposure]      = np.tanh(ALPHA_TANH * nums / C_ref_g)

    return I_dep, numerators


# ================================================================
# STEP 6 — INTER-REPLICATE STATISTICS
# ================================================================

def compute_stats(I_dep, samples):
    """
    Compute descriptive statistics and paired t-tests across replicates.

    For each frequency:
      - mean and SEM of I_DEP (PRE and POST)
      - mean and SEM of delta I_DEP = POST - PRE
      - two-tailed paired t-test p-value (PRE vs POST)

    Returns a dict of numpy arrays indexed by statistic name.
    """
    pre_matrix  = np.array([I_dep[s]['PRE']  for s in samples])
    post_matrix = np.array([I_dep[s]['POST'] for s in samples])
    n           = len(samples)

    mean_pre  = np.nanmean(pre_matrix,  axis=0)
    mean_post = np.nanmean(post_matrix, axis=0)
    std_pre   = np.nanstd(pre_matrix,   axis=0, ddof=1)
    std_post  = np.nanstd(post_matrix,  axis=0, ddof=1)
    sem_pre   = std_pre  / np.sqrt(n)
    sem_post  = std_post / np.sqrt(n)

    delta_matrix = post_matrix - pre_matrix
    mean_delta   = np.nanmean(delta_matrix, axis=0)
    std_delta    = np.nanstd(delta_matrix,  axis=0, ddof=1)
    sem_delta    = std_delta / np.sqrt(n)

    # Two-tailed paired t-test per frequency
    p_values = []
    for i in range(len(FREQ_ORDER)):
        try:
            _, p = stats.ttest_rel(pre_matrix[:, i], post_matrix[:, i],
                                   nan_policy='omit')
        except Exception:
            p = np.nan
        p_values.append(p)

    return {
        'pre_matrix':   pre_matrix,
        'post_matrix':  post_matrix,
        'delta_matrix': delta_matrix,
        'mean_pre':     mean_pre,
        'mean_post':    mean_post,
        'std_pre':      std_pre,
        'std_post':     std_post,
        'sem_pre':      sem_pre,
        'sem_post':     sem_post,
        'mean_delta':   mean_delta,
        'std_delta':    std_delta,
        'sem_delta':    sem_delta,
        'p_values':     np.array(p_values),
        'n':            n,
    }


def significance_stars(p):
    """Return a significance annotation string from a p-value."""
    if np.isnan(p): return '?'
    if p < 0.001:   return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'


# ================================================================
# STEP 7 — FIGURES
# ================================================================

freq_labels = [str(f) for f in FREQ_ORDER]
x           = np.arange(len(FREQ_ORDER))
w           = 0.15   # horizontal jitter for individual replicate points


def make_fig1_individual(I_dep, samples, condition):
    """
    Figure 1 — Individual DEP spectra.
    One panel per replicate showing PRE and POST I_DEP across frequencies.
    """
    n     = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f'Individual DEP spectra - {condition}\n'
        f'(PRE vs POST, per replicate)',
        fontsize=13, fontweight='bold')

    for idx, s in enumerate(samples):
        ax = axes[idx]
        ax.axhline(0, color='black', linewidth=0.8)
        ax.plot(freq_labels, I_dep[s]['PRE'],
                'o-',  color=C_PRE,  linewidth=2, markersize=7, label='PRE')
        ax.plot(freq_labels, I_dep[s]['POST'],
                's--', color=C_POST, linewidth=2, markersize=7,
                label='POST exposure')
        ax.set_title(f'Replicate {s}', fontweight='bold')
        ax.set_xlabel('Frequency (MHz)')
        if idx == 0:
            ax.set_ylabel('$I_{DEP}$ (a.u.)')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(framealpha=0.9)

    plt.tight_layout()
    return fig


def make_fig2_mean(st, condition):
    """
    Figure 2 — Mean DEP response.
    Mean ± SEM of I_DEP for PRE and POST, with significance annotations
    (two-tailed paired t-test) per frequency.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='black', linewidth=0.8)

    ax.errorbar(x - 0.05, st['mean_pre'],  yerr=st['sem_pre'],
                fmt='o-',  color=C_PRE,  linewidth=2.5, markersize=8,
                capsize=5, capthick=2,
                label=f'PRE exposure (n={st["n"]})')
    ax.errorbar(x + 0.05, st['mean_post'], yerr=st['sem_post'],
                fmt='s--', color=C_POST, linewidth=2.5, markersize=8,
                capsize=5, capthick=2,
                label=f'POST exposure (n={st["n"]})')

    for i, p in enumerate(st['p_values']):
        stars = significance_stars(p)
        y_top = max(st['mean_pre'][i]  + st['sem_pre'][i],
                    st['mean_post'][i] + st['sem_post'][i]) + 0.08
        ax.text(i, y_top, stars, ha='center', va='bottom', fontsize=12,
                color='black' if stars != 'ns' else '#999999')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l} MHz' for l in freq_labels])
    ax.set_ylabel('$I_{DEP}$ (a.u.)', fontsize=12)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(
        f'Mean DEP response +/- SEM - {condition}\n'
        f'(n={st["n"]} independent replicates)',
        fontweight='bold')
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    return fig


def make_fig3_delta(I_dep, st, samples, condition):
    """
    Figure 3 — Net DEP shift.
    Delta I_DEP = POST - PRE, showing individual replicate points
    (jittered) overlaid with the mean ± SEM and significance annotations.
    """
    ind_colors = ['#6BAED6', '#FC8D59', '#74C476', '#9E9AC8', '#FD8D3C']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='black', linewidth=0.8)

    for idx, s in enumerate(samples):
        delta_ind = I_dep[s]['POST'] - I_dep[s]['PRE']
        ax.scatter(x + (idx - len(samples) // 2) * w, delta_ind,
                   color=ind_colors[idx % len(ind_colors)],
                   s=55, zorder=3, alpha=0.85, label=f'Replicate {s}')

    ax.errorbar(x, st['mean_delta'], yerr=st['sem_delta'],
                fmt='D-', color='black', linewidth=2.5, markersize=9,
                capsize=6, capthick=2, zorder=5, label='Mean +/- SEM')
    ax.fill_between(x,
                    st['mean_delta'] - st['sem_delta'],
                    st['mean_delta'] + st['sem_delta'],
                    alpha=0.12, color='black')

    for i, p in enumerate(st['p_values']):
        stars = significance_stars(p)
        y_pos = st['mean_delta'][i] + st['sem_delta'][i] + 0.06
        ax.text(i, y_pos, stars, ha='center', fontsize=12,
                color='#CC0000' if stars != 'ns' else '#999999')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l} MHz' for l in freq_labels])
    ax.set_ylabel('Delta I_DEP = POST - PRE (a.u.)', fontsize=12)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(
        f'Net effect of exposure - {condition}\n'
        f'Delta I_DEP per frequency (n={st["n"]})',
        fontweight='bold')
    ax.legend(framealpha=0.9, loc='lower right')
    plt.tight_layout()
    return fig


def make_fig4_heatmap(st, samples, condition):
    """
    Figure 4 — Synthetic heatmap.
    Three panels showing I_DEP values for PRE, POST, and Delta
    across all replicates and frequencies.
    Coloured according to RdBu_r (PRE/POST) and PuOr (Delta) diverging maps.
    """
    n_s  = len(samples)
    fig, axes4 = plt.subplots(1, 3, figsize=(14, max(3.5, n_s * 1.1)))
    fig.suptitle(f'DEP index heatmap - {condition}', fontweight='bold')

    titles_h = ['PRE exposure', 'POST exposure', 'Delta (POST - PRE)']
    matrices = [st['pre_matrix'], st['post_matrix'], st['delta_matrix']]
    cmaps    = ['RdBu_r', 'RdBu_r', 'PuOr']

    for ax, mat, title, cmap in zip(axes4, matrices, titles_h, cmaps):
        vmax = 1.0 if 'Delta' not in title else 0.8
        im   = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(freq_labels)))
        ax.set_xticklabels([f'{l}M' for l in freq_labels], fontsize=9)
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(samples)
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)
        # Annotate each cell with its numerical value
        for i in range(n_s):
            for j in range(len(FREQ_ORDER)):
                ax.text(j, i, f'{mat[i, j]:.2f}',
                        ha='center', va='center', fontsize=8)

    plt.tight_layout()
    return fig


# ================================================================
# STEP 8 — SAVE TEXT BILAN
# ================================================================

def save_bilan_txt(bilan_folder, condition, samples, I_dep, st, numerators):
    """
    Write a structured plain-text summary to BILAN-<condition>.txt.

    Sections:
      1. Raw numerators (C_center - C_rings) per replicate and frequency
      2. Tanh-normalised I_DEP per replicate (PRE, POST, delta)
      3. Inter-replicate statistics: mean, SEM, paired t-test p-value
         and significance annotation per frequency
    """
    filepath  = os.path.join(bilan_folder, f"BILAN-{condition}.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep       = "=" * 72

    L = [sep,
         f"  DEP BILAN - {condition}",
         f"  Generated  : {timestamp}",
         f"  Replicates : {', '.join(samples)}   (n = {len(samples)})",
         sep, ""]

    # --- Raw numerators table ---
    L.append("--- Raw numerators (C_center - C_rings) ---")
    col_w  = 14
    header = f"  {'Freq (MHz)':>10} |"
    for s in samples:
        header += f" {'PRE-'+s:>{col_w}} | {'POST-'+s:>{col_w}} |"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))

    for fi, f in enumerate(FREQ_ORDER):
        row = f"  {str(f)+' MHz':>10} |"
        for s in samples:
            pv = numerators[s].get('PRE',  np.full(6, np.nan))[fi]
            ov = numerators[s].get('POST', np.full(6, np.nan))[fi]
            row += f" {pv:>{col_w}.5f} | {ov:>{col_w}.5f} |"
        L.append(row)
    L.append("")

    # --- I_DEP per replicate ---
    L.append("--- I_DEP per replicate (tanh-normalized, C_ref_global per replicate) ---")
    for s in samples:
        L.append(f"\n  Replicate {s} :")
        L.append(f"  {'Freq':>8} | {'PRE':>8} | {'POST':>8} | {'Delta':>8}")
        L.append("  " + "-" * 42)
        for fi, f in enumerate(FREQ_ORDER):
            pre_v  = I_dep[s]['PRE'][fi]
            post_v = I_dep[s]['POST'][fi]
            L.append(f"  {str(f)+' MHz':>8} | {pre_v:>8.4f} | "
                     f"{post_v:>8.4f} | {post_v - pre_v:>8.4f}")
    L.append("")

    # --- Inter-replicate statistics ---
    L.append("--- Inter-replicate statistics ---")
    L.append(
        f"  {'Freq':>8} | {'mean PRE':>9} | {'SEM PRE':>8} |"
        f" {'mean POST':>10} | {'SEM POST':>9} |"
        f" {'mean D':>8} | {'SEM D':>7} | {'p-value':>9} | {'sig':>4}")
    L.append("  " + "-" * 98)

    for fi, f in enumerate(FREQ_ORDER):
        p = st['p_values'][fi]
        L.append(
            f"  {str(f)+' MHz':>8} |"
            f" {st['mean_pre'][fi]:>9.4f} | {st['sem_pre'][fi]:>8.4f} |"
            f" {st['mean_post'][fi]:>10.4f} | {st['sem_post'][fi]:>9.4f} |"
            f" {st['mean_delta'][fi]:>8.4f} | {st['sem_delta'][fi]:>7.4f} |"
            f" {p:>9.4f} | {significance_stars(p):>4}")

    L += ["", sep, ""]

    with open(filepath, 'w', encoding='utf-8') as fout:
        fout.write("\n".join(L))

    print(f"  [SAVED] Bilan text -> {filepath}")
    return filepath


# ================================================================
# MAIN LOOP
# ================================================================

if __name__ == "__main__":

    # --- Step 1: select input folder ---
    folder = select_folder()
    if not folder:
        print("[ERROR] No folder selected.")
    else:
        print(f"\n{'='*60}")
        print(f"  Folder : {folder}")
        print(f"{'='*60}\n")

        # --- Step 4: load all result files ---
        results = load_experiment(folder)

        if not results:
            print("[ERROR] No valid result files found.")
        else:
            # Create BILAN/ output subfolder
            bilan_folder = os.path.join(folder, "BILAN")
            os.makedirs(bilan_folder, exist_ok=True)
            print(f"  Output folder : {bilan_folder}\n")

            all_stats = {}
            all_I_dep = {}

            # --- Process each condition (TEMOIN / TEST) ---
            for condition in sorted(results.keys()):
                print(f"\n{'='*60}")
                print(f"  CONDITION : {condition}")
                print(f"{'='*60}")

                cond_data = results[condition]

                # Retain only replicates with both PRE and POST data complete
                valid_samples = []
                for s in sorted(cond_data.keys()):
                    if 'PRE' in cond_data[s] and 'POST' in cond_data[s]:
                        valid_samples.append(s)
                    else:
                        missing = [e for e in ['PRE', 'POST']
                                   if e not in cond_data[s]]
                        print(f"  [WARN] {condition}-{s} : missing {missing} - excluded")

                if not valid_samples:
                    print(f"  [ERROR] No complete PRE/POST pairs - skipping {condition}.")
                    continue

                print(f"  Replicates : {valid_samples}  (n={len(valid_samples)})")

                # --- Step 5: compute I_DEP ---
                I_dep, numerators = compute_I_dep(results, condition)
                I_dep_valid = {s: I_dep[s] for s in valid_samples}
                num_valid   = {s: numerators[s] for s in valid_samples}

                # --- Step 6: inter-replicate statistics ---
                st = compute_stats(I_dep_valid, valid_samples)
                all_stats[condition] = st
                all_I_dep[condition] = I_dep_valid

                # Console summary table
                print(f"\n  {'Freq':>8} | {'mean PRE':>9} | {'mean POST':>10} |"
                      f" {'mean D':>8} | {'p-value':>9} | {'sig':>4}")
                print("  " + "-" * 58)
                for fi, f in enumerate(FREQ_ORDER):
                    p = st['p_values'][fi]
                    print(f"  {str(f)+' MHz':>8} | {st['mean_pre'][fi]:>9.4f} |"
                          f" {st['mean_post'][fi]:>10.4f} |"
                          f" {st['mean_delta'][fi]:>8.4f} |"
                          f" {p:>9.4f} | {significance_stars(p):>4}")

                # --- Step 7: generate figures ---
                print(f"\n  Generating figures...")
                fig1 = make_fig1_individual(I_dep_valid, valid_samples, condition)
                fig2 = make_fig2_mean(st, condition)
                fig3 = make_fig3_delta(I_dep_valid, st, valid_samples, condition)
                fig4 = make_fig4_heatmap(st, valid_samples, condition)

                for fig, suffix in [
                    (fig1, "Fig1-Individual"),
                    (fig2, "Fig2-Mean"),
                    (fig3, "Fig3-Delta"),
                    (fig4, "Fig4-Heatmap"),
                ]:
                    path = os.path.join(bilan_folder,
                                        f"BILAN-{condition}-{suffix}.svg")
                    fig.savefig(path, dpi=150, bbox_inches='tight')
                    print(f"  [SAVED] {os.path.basename(path)}")

                # --- Step 8: save text bilan ---
                save_bilan_txt(bilan_folder, condition, valid_samples,
                               I_dep_valid, st, num_valid)

                plt.show(block=False)

            # --- Comparison figure: TEST vs TEMOIN (if both conditions exist) ---
            if 'TEST' in all_stats and 'TEMOIN' in all_stats:
                print(f"\n{'='*60}")
                print("  COMPARISON: TEST vs TEMOIN")
                print(f"{'='*60}")

                st_t = all_stats['TEST']
                st_c = all_stats['TEMOIN']

                fig_comp, ax = plt.subplots(figsize=(9, 5))
                ax.axhline(0, color='black', linewidth=0.8)

                ax.errorbar(x - 0.08, st_t['mean_delta'], yerr=st_t['sem_delta'],
                            fmt='D-', color=C_POST, linewidth=2.5, markersize=8,
                            capsize=5, capthick=2,
                            label=f'TEST (n={st_t["n"]})')
                ax.errorbar(x + 0.08, st_c['mean_delta'], yerr=st_c['sem_delta'],
                            fmt='s--', color=C_PRE, linewidth=2.5, markersize=8,
                            capsize=5, capthick=2,
                            label=f'TEMOIN (n={st_c["n"]})')

                ax.set_xticks(x)
                ax.set_xticklabels([f'{l} MHz' for l in freq_labels])
                ax.set_ylabel('Delta I_DEP = POST - PRE (a.u.)', fontsize=12)
                ax.set_xlabel('Frequency', fontsize=12)
                ax.set_title('TEST vs TEMOIN - Net DEP shift (mean +/- SEM)',
                             fontweight='bold')
                ax.legend(framealpha=0.9)
                plt.tight_layout()

                comp_path = os.path.join(bilan_folder,
                                         "BILAN-COMPARISON-TEST-vs-TEMOIN.svg")
                fig_comp.savefig(comp_path, dpi=150, bbox_inches='tight')
                print(f"  [SAVED] {os.path.basename(comp_path)}")
                plt.show(block=False)

            print(f"\n{'='*60}")
            print(f"  All done. Results in : {bilan_folder}")
            print(f"{'='*60}\n")

            plt.show(block=True)   # keep all figures open until manually closed