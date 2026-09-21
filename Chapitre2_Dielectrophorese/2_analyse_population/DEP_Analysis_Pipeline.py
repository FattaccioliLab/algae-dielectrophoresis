import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, box
import os
import math
from datetime import datetime


# ================================================================
# GLOBAL CONFIGURATION
# ================================================================

SCALE = 3.06                  # spatial calibration factor (pixels per micrometre)
radius       = 50  * SCALE    # inner electrode radius in pixels (~50 um)
radius_outer = 140 * SCALE    # outer electrode radius in pixels (~140 um)
scale_factor = SCALE


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

# A single persistent Tk root instance is kept throughout the session.
# This avoids conflicts between successive filedialog / simpledialog calls.
_tk_root = None

def get_tk_root():
    """Return the global Tk root, creating it once if necessary."""
    global _tk_root
    if _tk_root is None:
        _tk_root = tk.Tk()
        _tk_root.withdraw()
    return _tk_root


def on_key(event):
    """Close the current matplotlib figure when the user presses Enter."""
    if event.key == 'enter':
        plt.close(event.canvas.figure)


# ================================================================
# STEP 1 — SELECT RESULT FOLDER
# ================================================================

def select_result_folder():
    """
    Open a directory chooser dialog.
    All output files (figures, text results) will be written here.
    Raises RuntimeError if no folder is selected.
    """
    root = get_tk_root()
    folder = filedialog.askdirectory(title="Select RESULT folder")
    if not folder:
        raise RuntimeError("No result folder selected — aborting.")
    return folder


# ================================================================
# STEP 2 — SELECT VIDEO FILE
# ================================================================

def select_video():
    """Open a file chooser and return the path to the selected video."""
    root = get_tk_root()
    path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    return path


# ================================================================
# STEP 3 — EXPERIMENT METADATA GUI
# ================================================================

def ask_metadata(video_path):
    """
    Display a modal dialog to collect experiment metadata:
      - Condition : TEMOIN (control) or TEST
      - Sample    : S1 to S6 (biological replicate)
      - Exposure  : PRE or POST (light exposure)
      - Frequency : 1, 10, 20, 30, 40 or 50 MHz

    The video filename is displayed for visual confirmation.
    Returns a dict with keys 'condition', 'sample', 'exposure', 'freq',
    or None if the dialog is cancelled.

    StringVar instances are explicitly bound to the Toplevel window (master=win)
    to ensure correct value retrieval on repeated calls within the same session.
    """
    root = get_tk_root()
    result = {}

    # Use Toplevel rather than a new Tk() instance to avoid
    # invalidating the persistent global root.
    win = tk.Toplevel(root)
    win.title("Experiment metadata")
    win.resizable(False, False)
    win.lift()
    win.focus_force()
    win.grab_set()    # make the dialog modal

    pad = dict(padx=10, pady=6)

    # Video filename — read-only, shown for confirmation
    tk.Label(win, text="Video:", font=("Helvetica", 10, "bold")).grid(
        row=0, column=0, sticky="w", **pad)
    tk.Label(win, text=os.path.basename(video_path),
             font=("Helvetica", 10), fg="#444444").grid(
        row=0, column=1, columnspan=3, sticky="w", **pad)

    # Experimental condition
    tk.Label(win, text="Condition:", font=("Helvetica", 10, "bold")).grid(
        row=1, column=0, sticky="w", **pad)
    condition_var = tk.StringVar(master=win, value="TEST")
    for idx, val in enumerate(["TEMOIN", "TEST"]):
        tk.Radiobutton(win, text=val, variable=condition_var,
                       value=val).grid(row=1, column=idx+1, sticky="w")

    # Biological replicate identifier
    tk.Label(win, text="Sample:", font=("Helvetica", 10, "bold")).grid(
        row=2, column=0, sticky="w", **pad)
    sample_var = tk.StringVar(master=win, value="S3")
    sample_cb = ttk.Combobox(win, textvariable=sample_var,
                              values=["S1","S2","S3","S4","S5","S6"],
                              width=6, state="readonly")
    sample_cb.grid(row=2, column=1, sticky="w", **pad)

    # Light exposure condition
    tk.Label(win, text="Exposure:", font=("Helvetica", 10, "bold")).grid(
        row=3, column=0, sticky="w", **pad)
    exposure_var = tk.StringVar(master=win, value="PRE")
    for idx, val in enumerate(["PRE", "POST"]):
        tk.Radiobutton(win, text=val, variable=exposure_var,
                       value=val).grid(row=3, column=idx+1, sticky="w")

    # Applied AC frequency
    tk.Label(win, text="Frequency:", font=("Helvetica", 10, "bold")).grid(
        row=4, column=0, sticky="w", **pad)
    freq_var = tk.StringVar(master=win, value="1")
    freq_cb = ttk.Combobox(win, textvariable=freq_var,
                            values=["1","10","20","30","40","50"],
                            width=6, state="readonly")
    freq_cb.grid(row=4, column=1, sticky="w", **pad)
    tk.Label(win, text="MHz").grid(row=4, column=2, sticky="w")

    def on_ok():
        # Read all widget values at the moment of the click,
        # not at widget creation time.
        c = condition_var.get()
        s = sample_var.get()
        e = exposure_var.get()
        f = freq_var.get()
        print(f"[DEBUG RAW] condition={c!r} sample={s!r} exposure={e!r} freq={f!r}")
        result['condition'] = c
        result['sample']    = s
        result['exposure']  = e
        result['freq']      = f + "MHz"
        print(f"[DEBUG] Metadata captured: {result}")
        win.grab_release()
        win.destroy()

    def on_cancel():
        win.grab_release()
        win.destroy()

    tk.Button(win, text="OK", command=on_ok,
              bg="#2166AC", fg="white", width=10).grid(row=5, column=1, pady=12)
    tk.Button(win, text="Cancel", command=on_cancel, width=10).grid(
        row=5, column=2, pady=12)

    win.wait_window()    # block until the window is closed, without mainloop()
    return result if result else None


def build_basename(meta):
    """
    Build the output file base name from the experiment metadata dict.
    Example output: 'TEST-S1-PRE-1MHz'
    Must be called immediately after ask_metadata() returns, and never
    cached across loop iterations.
    """
    return (f"{meta['condition']}-"
            f"{meta['sample']}-"
            f"{meta['exposure']}-"
            f"{meta['freq']}")


# ================================================================
# STEP 4a — VIDEO PREVIEW AND FRAME SELECTION
# ================================================================

def preview_video_with_slider(video_path):
    """
    Open an OpenCV window with a trackbar to browse the video frame by frame.
    The user presses Enter or Q (or closes the window) to proceed.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("Video preview", cv2.WINDOW_NORMAL)

    def on_slider(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        ret, frame = cap.read()
        if ret:
            display = frame.copy()
            cv2.putText(display, f"Frame {val}/{total_frames}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video preview", display)

    cv2.createTrackbar("Frame", "Video preview", 0, total_frames - 1, on_slider)
    on_slider(0)

    while True:
        key = cv2.waitKey(30)
        if key == 13 or key == ord('q'):
            break
        if cv2.getWindowProperty("Video preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def get_frame_numbers(max_frames):
    """
    Ask the user to enter two frame indices via input dialog boxes:
      f1 — reference frame (before the electric field is applied)
      f2 — steady-state frame (after the field reaches equilibrium)

    Uses the persistent Tk root to avoid invalidating the global
    instance that backs all StringVar objects in the session.
    """
    root = get_tk_root()    # must NOT create a new tk.Tk() here
    f1 = simpledialog.askinteger(
        "Frames", f"First frame (0-{max_frames-1})",
        parent=root)
    f2 = simpledialog.askinteger(
        "Frames", f"Second frame (0-{max_frames-1})",
        parent=root)
    return f1, f2


def average_frames(cap, frame_number, window=5):
    """
    Return the temporal average of frames in the interval
    [frame_number - window, frame_number + window].
    Temporal averaging reduces shot noise and improves the reliability
    of mean intensity measurements.
    """
    imgs = []
    for k in range(frame_number - window, frame_number + window):
        if k < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame = cap.read()
        if ret:
            imgs.append(frame.astype(np.float32))
    return np.mean(imgs, axis=0).astype(np.uint8)


# ================================================================
# STEP 4b — CELL DETECTION
# ================================================================

def detect_particles(frame):
    """
    Detect approximately circular cells using the Hough gradient transform
    on the grayscale image.

    Returns the annotated frame (circles drawn in red) and a list of
    (x, y) centre coordinates for each detected cell.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=0.1, minDist=20,
        param1=20, param2=20,
        minRadius=2, maxRadius=20)
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            centers.append((c[0], c[1]))
            cv2.circle(frame, (c[0], c[1]), c[2], (0, 0, 255), 2)
    return frame, centers


# ================================================================
# STEP 4c — MANUAL ELECTRODE CENTRE SELECTION
# ================================================================

_center_global = None

def _mouse_click(event, x, y, flags, param):
    """Mouse callback: store the clicked pixel as the electrode centre."""
    global _center_global
    if event == cv2.EVENT_LBUTTONDOWN:
        _center_global = (x, y)


def manual_electrode_center(frame, scale=0.5):
    """
    Display a downscaled frame and let the user click on the electrode centre.
    Inner and outer electrode boundaries are overlaid as red circles
    for visual guidance.  Press Enter to confirm the selection.

    Returns the centre coordinates rescaled to full-resolution pixels.
    """
    global _center_global
    _center_global = None
    resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    cv2.namedWindow("Electrode selection")
    cv2.setMouseCallback("Electrode selection", _mouse_click)

    while True:
        temp = resized.copy()
        if _center_global:
            cv2.circle(temp, _center_global, int(radius * scale), (0, 0, 255), 2)
            cv2.circle(temp, _center_global, int(radius_outer * scale), (0, 0, 255), 2)
        cv2.imshow("Electrode selection", temp)
        key = cv2.waitKey(1)
        if key == 13:
            break

    cv2.destroyAllWindows()
    return (int(_center_global[0] / scale), int(_center_global[1] / scale))


# ================================================================
# DEP ANALYSIS — ROI FUNCTIONS
# ================================================================

def circular_annulus_mask(shape, center, R, W):
    """
    Return a boolean mask selecting pixels within a circular annulus
    of mean radius R and radial width W, centred on `center`.
    """
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return (dist >= R - W/2) & (dist <= R + W/2)


def cumulative_intensity_shift_circular(img1, img2, center, R, W):
    """
    Compute mean pixel intensities and full intensity histograms for a
    circular annulus ROI in the reference frame (img1) and the
    steady-state frame (img2).

    Returns: mean1, mean2, hist1, hist2
    """
    mask = circular_annulus_mask(img1.shape, center, R, W)
    roi1, roi2 = img1[mask], img2[mask]
    hist1 = cv2.calcHist([roi1], [0], None, [256], [0, 256]).ravel()
    hist2 = cv2.calcHist([roi2], [0], None, [256], [0, 256]).ravel()
    return np.mean(roi1), np.mean(roi2), hist1, hist2


def cumulative_intensity_shift_square(img1, img2, roi):
    """
    Compute mean pixel intensities and full intensity histograms for a
    rectangular ROI defined by pixel coordinates (x0, y0, x1, y1).

    Returns: mean1, mean2, hist1, hist2
    """
    x0, y0, x1, y1 = roi
    roi1, roi2 = img1[y0:y1, x0:x1], img2[y0:y1, x0:x1]
    hist1 = cv2.calcHist([roi1], [0], None, [256], [0, 256]).ravel()
    hist2 = cv2.calcHist([roi2], [0], None, [256], [0, 256]).ravel()
    return np.mean(roi1), np.mean(roi2), hist1, hist2


def crown_intersection_area(radius_outer, radius_inner, electrode_center, frame_shape):
    """
    Compute the effective area of a circular annulus clipped by the image
    boundary, using Shapely polygon intersection.

    Applied when an annulus partially extends beyond the field of view,
    ensuring unbiased area-based density normalisation.
    """
    cx, cy   = electrode_center
    h, w     = frame_shape
    img_box  = box(0, 0, w, h)
    outer_circle = Point(cx, cy).buffer(radius_outer)
    inner_circle = Point(cx, cy).buffer(radius_inner)
    crown    = outer_circle.difference(inner_circle)
    return crown.intersection(img_box).area


def plot_normalized_distances(centers_matrix, electrode_center, frame_shape, ax, title, scale=1):
    """
    Compute and plot the radial cell surface density as a function of
    distance from the electrode centre.

    Cells are binned by radial distance (in um).  Each bin count is
    divided by the corresponding annular area to yield a surface density
    in cells per 400 um^2.  Annuli partially outside the image boundary
    are corrected using Shapely-based area clipping.

    Parameters
    ----------
    centers_matrix   : list of (x, y) pixel coordinates of detected cells
    electrode_center : (cx, cy) in pixels
    frame_shape      : (height, width[, channels]) of the source frame
    ax               : matplotlib Axes to draw on
    title            : subplot title string
    scale            : pixel-to-micrometre conversion factor (pass 1/SCALE)

    Returns
    -------
    normalized_counts : raw density values (before x400 scaling)
    bin_edges         : array of bin edges in um
    """
    distances = []
    for center in centers_matrix:
        # Convert pixel distance to micrometres
        dist = np.linalg.norm(np.array(center) - np.array(electrode_center)) * scale
        # Retain only cells whose enclosing annulus fits within the image
        if dist < min(electrode_center[0], electrode_center[1], frame_shape[1] - electrode_center[0], frame_shape[0] - electrode_center[1]):
            distances.append(dist)

    # 50 equally spaced bins from 0 to the maximum observed distance
    bin_edges =  np.linspace(0, max(distances), 50)
    bin_counts, _ = np.histogram(distances, bins=bin_edges)

    # Convert coordinates to micrometres for area calculations
    electrode_center = (electrode_center[0] * scale, electrode_center[1] * scale)
    frame_shape = ( frame_shape[0]*scale, frame_shape[1]*scale)

    normalized_counts = []
    for i in range(len(bin_edges) - 1):
        radius_outer = bin_edges[i + 1]
        radius_inner = bin_edges[i]

        # First bin: full disk area (no inner hole)
        if i == 0:
            crown_area = np.pi * (radius_outer ** 2)

        else:
            if radius_outer < min(electrode_center[0], electrode_center[1], frame_shape[1] - electrode_center[0], frame_shape[0] - electrode_center[1]):
                # Full annulus fits inside the image
                R=radius_outer
                A=min(electrode_center[0], electrode_center[1], frame_shape[1] - electrode_center[0], frame_shape[0] - electrode_center[1])
                crown_area = np.pi * (radius_outer ** 2 - radius_inner ** 2)
            else:
                # Annulus is partially outside the image: use clipped area
                crown_area = crown_intersection_area(radius_outer, radius_inner, electrode_center, frame_shape)

        if crown_area > 0:
            normalized_counts.append(bin_counts[i] / crown_area)
        else:
            normalized_counts.append(0)

    # Scale to cells per 100 um^2 and cells per 400 um^2 (reference area for display)
    normalized_counts_per_100um = [count * 100 for count in normalized_counts]
    normalized_counts_per_400um = [count * 400 for count in normalized_counts]

    ax.bar(bin_edges[:-1], normalized_counts_per_400um, width=np.diff(bin_edges), align='edge')

    ax.set_xlim([0, 500])

    ax.set_title(title)
    ax.set_xlabel('Distance to the center of the electrode (um)')
    ax.set_ylabel('Density of cells per 400 um^2')

    return normalized_counts, bin_edges


# ================================================================
# STEP 5 — CONTRAST FIGURE
# ================================================================

def make_contrast_figure(m1_c, m2_c, m1_i, m2_i, m1_o, m2_o,
                          h1_c, h2_c, h1_i, h2_i, h1_o, h2_o,
                          C_center, C_inner, C_outer,
                          DEP_center_local, DEP_inner_local, DEP_outer_local,
                          DEP_index, C_ref, C_floor, numerator, sigma_bruit,
                          basename):
    """
    Generate a 2-row diagnostic figure:
      Row 1 — pixel intensity histograms (before vs. steady-state) per ROI
      Row 2 — summary tables: mean intensities, background-corrected
               contrasts, and local DEP index for each ROI

    Returns the matplotlib Figure object.
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 3, height_ratios=[3, 1])

    titles = ["Center square", "Inner annulus", "Outer annulus"]
    data   = [(h1_c, h2_c), (h1_i, h2_i), (h1_o, h2_o)]

    for j, (title, (A, B)) in enumerate(zip(titles, data)):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(A, label="Before")
        ax.plot(B, label="Steady-state")
        ax.set_title(title)
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Pixel count")
        ax.legend()

    tables_data = [
        [["Mean before",           f"{m1_c:.2f}"],
         ["Mean after",            f"{m2_c:.2f}"],
         ["Contrast C (bg corr.)", f"{C_center:.4f}"],
         ["DEP local %",           f"{DEP_center_local:.2f}"]],
        [["Mean before",           f"{m1_i:.2f}"],
         ["Mean after",            f"{m2_i:.2f}"],
         ["Contrast C (bg corr.)", f"{C_inner:.4f}"],
         ["DEP local %",           f"{DEP_inner_local:.2f}"]],
        [["Mean before",           f"{m1_o:.2f}"],
         ["Mean after",            f"{m2_o:.2f}"],
         ["Contrast C (bg corr.)", f"{C_outer:.4f}"],
         ["DEP local %",           f"{DEP_outer_local:.2f}"]],
    ]

    for j in range(3):
        ax = fig.add_subplot(gs[1, j])
        ax.axis('off')
        tbl = ax.table(cellText=tables_data[j],
                       colLabels=["Parameter", "Value"], loc='center')
        tbl.scale(1, 2)

    fig.suptitle(basename + " — Contrast analysis", fontsize=12, fontweight='bold')
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    return fig


# ================================================================
# STEP 6 — SAVE TEXT RESULTS
# ================================================================

def save_text_results(result_folder, basename, meta,
                       sigma_bruit, bg_drift,
                       C_center, C_inner, C_outer, C_rings,
                       numerator, C_floor, C_ref, DEP_index,
                       m1_c, m2_c, m1_i, m2_i, m1_o, m2_o,
                       DEP_center_local, DEP_inner_local, DEP_outer_local,
                       hist_before=None, hist_after=None,
                       bin_edges=None):
    # """
    # Write all numerical results to a structured plain-text file named
    # ##<basename>##.txt in the result folder.
    #
    # Saved quantities:
    #   - Background noise estimation (sigma, drift)
    #   - Background-corrected optical contrasts per ROI
    #   - Redistribution signal (numerator) and SNR
    #   - Reference normalisation values (C_floor, C_ref)
    #   - Local DEP intensity (%) per ROI
    #   - Mean pixel intensities before and after field application
    #   - Radial cell density distribution (before/after, if available)
    #
    # Note: the final tanh-normalised I_DEP index is intentionally omitted.
    # It is computed in a separate post-processing script once all frequencies
    # for a given replicate are available, using a frequency-global C_ref
    # for consistent normalisation across the DEP spectrum.
    # """
    filepath = os.path.join(result_folder, f"##{basename}##.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 60)
    lines.append(f"  DEP ANALYSIS RESULTS — {basename}")
    lines.append(f"  Generated : {timestamp}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Condition : {meta['condition']}")
    lines.append(f"  Sample    : {meta['sample']}")
    lines.append(f"  Exposure  : {meta['exposure']}")
    lines.append(f"  Frequency : {meta['freq']}")
    lines.append("")
    lines.append("--- Background noise estimation ---")
    lines.append(f"  sigma_bruit = {sigma_bruit:.5f}")
    lines.append(f"  bg_drift    = {bg_drift:.5f}")
    lines.append("")
    lines.append("--- Corrected contrasts ---")
    lines.append(f"  C_center = {C_center:.5f}  ({'accumulation => nDEP' if C_center < 0 else 'depletion'})")
    lines.append(f"  C_inner  = {C_inner:.5f}")
    lines.append(f"  C_outer  = {C_outer:.5f}")
    lines.append(f"  C_rings  = {C_rings:.5f}")
    lines.append("")
    lines.append("--- Raw redistribution signal ---")
    lines.append(f"  numerator (C_center - C_rings) = {numerator:.5f}")
    lines.append(f"  SNR = |numerator| / sigma       = {abs(numerator)/sigma_bruit:.3f}")
    lines.append("")
    lines.append("--- Reference values ---")
    lines.append(f"  C_floor    = {C_floor:.5f}")
    lines.append(f"  C_ref      = {C_ref:.5f}")
    lines.append("")
    lines.append("--- Local DEP intensity (%) ---")
    lines.append(f"  Center : {DEP_center_local:.2f} %")
    lines.append(f"  Inner  : {DEP_inner_local:.2f} %")
    lines.append(f"  Outer  : {DEP_outer_local:.2f} %")
    lines.append("")
    lines.append("--- Mean pixel intensities ---")
    lines.append(f"  Center — before: {m1_c:.2f}  after: {m2_c:.2f}")
    lines.append(f"  Inner  — before: {m1_i:.2f}  after: {m2_i:.2f}")
    lines.append(f"  Outer  — before: {m1_o:.2f}  after: {m2_o:.2f}")
    lines.append("")

    # Radial cell density distribution (optional section)
    if hist_before is not None and hist_after is not None and bin_edges is not None:
        lines.append("--- Cell radial distribution (cells/400 um^2) ---")
        lines.append(f"  {'Bin center (um)':>18} | {'Before':>10} | {'After':>10}")
        lines.append("  " + "-" * 44)
        centers_bin = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for bc, vb, va in zip(centers_bin, hist_before, hist_after):
            lines.append(f"  {bc:>18.2f} | {vb:>10.4f} | {va:>10.4f}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[SAVED] Text results -> {filepath}")
    return filepath


# ================================================================
# STEPS 8-9 — CELL DENSITY FIGURE
# ================================================================

def make_density_figure(frames, Original, processed, centers1, centers2,
                         f1, f2, electrode_center, basename):
    """
    Generate a 3-row x 4-column figure:
      Row 0    — column headers
      Col 0    — row labels (Pre / Post Electric Field)
      Col 1    — original (unprocessed) frame
      Col 2    — processed frame with ROI overlays and scale annotations
      Col 3    — radial cell density histogram

    Both histogram axes share the same Y scale, rounded to the nearest
    multiple of 0.2 cells/400 um^2 for clean tick labels.
    Vertical dashed blue lines mark the inner and outer electrode edges.

    Returns the Figure and a list of (norm_counts, bin_edges) per row.
    """
    frame_data = [
        (Original[0], processed[0], centers1, f1),
        (Original[1], processed[1], centers2, f2),
    ]

    # --- Pre-compute a common Y axis maximum across both histograms ---
    y_max_global = 0
    for i, (_, _, centers, _) in enumerate(frame_data):
        distances = []
        lim_px = min(electrode_center[0], electrode_center[1],
                     frames[i].shape[1] - electrode_center[0],
                     frames[i].shape[0] - electrode_center[1])
        for c in centers:
            dist = np.linalg.norm(np.array(c) - np.array(electrode_center)) / SCALE
            if dist < lim_px / SCALE:
                distances.append(dist)
        if not distances:
            continue
        bin_edges = np.linspace(0, max(distances), 50)
        bin_counts, _ = np.histogram(distances, bins=bin_edges)
        ec_um = (electrode_center[0] / SCALE, electrode_center[1] / SCALE)
        fs_um = (frames[i].shape[0] / SCALE, frames[i].shape[1] / SCALE)
        for k in range(len(bin_edges) - 1):
            r_out, r_in = bin_edges[k+1], bin_edges[k]
            if k == 0:
                area = np.pi * r_out**2
            else:
                lim = min(ec_um[0], ec_um[1], fs_um[1]-ec_um[0], fs_um[0]-ec_um[1])
                area = np.pi*(r_out**2-r_in**2) if r_out < lim else \
                       crown_intersection_area(r_out, r_in, ec_um, fs_um)
            if area > 0:
                y_max_global = max(y_max_global, bin_counts[k] / area * 400)

    # Round up to the next multiple of 0.2 for clean tick marks
    tick_step  = 0.2
    y_max_plot = math.ceil(y_max_global / tick_step) * tick_step
    y_max_plot = max(y_max_plot, tick_step)

    # --- Figure layout ---
    fig = plt.figure(figsize=(26, 14))
    gs  = fig.add_gridspec(3, 4,
                            height_ratios=[0.07, 1, 1],
                            width_ratios=[0.08, 1, 1, 1.3],
                            hspace=0.04, wspace=0.35)

    # Column header labels
    col_titles = ['', 'Frame before analysis',
                  'Frame post analysis', 'Cell radial distribution']
    for j, title in enumerate(col_titles):
        ax_t = fig.add_subplot(gs[0, j])
        ax_t.axis('off')
        if title:
            ax_t.text(0.5, 0.5, title, ha='center', va='center',
                      fontsize=13, fontweight='bold', transform=ax_t.transAxes)

    # Row labels
    row_labels = ['Pre\nElectric Field',
                  'Stationary State\nPost Electric Field']
    for i, label in enumerate(row_labels):
        ax_l = fig.add_subplot(gs[i+1, 0])
        ax_l.axis('off')
        ax_l.text(0.5, 0.5, label, ha='center', va='center',
                  fontsize=10, fontweight='bold', rotation=90,
                  transform=ax_l.transAxes)

    hist_data = []    # stores (norm_counts, bin_edges) per row

    for i, (orig, proc, centers, frame_idx) in enumerate(frame_data):
        row = i + 1

        # Original frame
        ax_orig = fig.add_subplot(gs[row, 1])
        ax_orig.imshow(orig)
        ax_orig.axis('off')

        # Processed frame with overlays
        ax_proc = fig.add_subplot(gs[row, 2])
        ax_proc.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
        ax_proc.axis('off')

        img_shape = frames[i].shape
        max_y, max_x = img_shape[:2]

        # Horizontal scale bar — 50 um
        bar_length_px = 50 * scale_factor
        bar_x, bar_y  = 80, max_y - 80
        ax_proc.add_patch(patches.Rectangle(
            (bar_x, bar_y - 20), bar_length_px, 20,
            linewidth=0, facecolor='black'))
        ax_proc.text(bar_x + bar_length_px/2, bar_y - 28,
                     '50 um', color='black', ha='center', va='bottom', fontsize=9)

        # Reference area square — 20x20 um = 400 um^2
        sq_size = 20
        sq_x = max_x - sq_size * scale_factor - 180
        sq_y = max_y - sq_size * scale_factor - 90
        ax_proc.add_patch(patches.Rectangle(
            (sq_x, sq_y), sq_size*scale_factor, sq_size*scale_factor,
            linewidth=1, edgecolor='black', facecolor='none'))
        ax_proc.text(sq_x + sq_size*scale_factor/2, sq_y - 8,
                     '400 um^2', color='black', ha='center', va='bottom', fontsize=9)

        # Radial density histogram
        ax_hist = fig.add_subplot(gs[row, 3])
        ret = plot_normalized_distances(
            centers, electrode_center, frames[i].shape,
            ax_hist, '', 1/SCALE)

        if ret:
            norm_counts, bin_edges = ret
            hist_data.append((norm_counts, bin_edges))
        else:
            hist_data.append(([], []))

        # Apply the shared Y scale
        ax_hist.set_ylim(0, y_max_plot)
        ax_hist.set_yticks(np.arange(0, y_max_plot + tick_step/2, tick_step))

        # Show X label only on the bottom histogram to avoid overlap
        if i == 0:
            ax_hist.set_xlabel('')
            ax_hist.tick_params(labelbottom=False)
        else:
            ax_hist.set_xlabel('Distance to electrode center (um)', fontsize=9)

        # Vertical markers for electrode inner and outer boundaries
        r_um       = radius       / SCALE
        r_outer_um = radius_outer / SCALE
        ax_hist.axvline(r_um,       color='blue', linestyle='--', linewidth=1.5,
                        label=f'Inner edge ({r_um:.0f} um)')
        ax_hist.axvline(r_outer_um, color='blue', linestyle=':', linewidth=1.5,
                        label=f'Outer edge ({r_outer_um:.0f} um)')
        ax_hist.legend(fontsize=8, framealpha=0.9, loc='upper right')

    fig.suptitle(basename + " — Cell density", fontsize=12, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.03, right=0.98, top=0.97, bottom=0.03)
    fig.canvas.mpl_connect('key_press_event', on_key)

    return fig, hist_data


# ================================================================
# STEP 11 — CONTINUE / STOP DIALOG
# ================================================================

def ask_continue():
    """
    Display a yes/no message box after each analysis cycle.
    Returns True to process another video, False to exit the loop.
    """
    root = get_tk_root()
    answer = messagebox.askyesno(
        "Continue?",
        "Analysis saved.\n\nDo you want to analyse another video?",
        icon='question'
    )
    return answer


# ================================================================
## MAIN LOOP
# ================================================================

if __name__ == "__main__":

    # Step 1: select the result folder once for the entire session
    result_folder = select_result_folder()
    print(f"[INFO] Result folder: {result_folder}")

    while True:

        # Step 2: select video file
        video = select_video()
        if not video:
            print("[INFO] No video selected — stopping.")
            break

        # Step 3: collect experiment metadata
        meta = ask_metadata(video)

        if meta is None:
            print("[INFO] Metadata cancelled — stopping.")
            break

        # basename MUST be built here, immediately after ask_metadata(),
        # and must never be cached or reused from a previous iteration.
        basename = build_basename(meta)
        print(f"\n[INFO] Analysis: {basename}")
        print(f"       Condition : {meta['condition']}")
        print(f"       Sample    : {meta['sample']}")
        print(f"       Exposure  : {meta['exposure']}")
        print(f"       Frequency : {meta['freq']}")

        # Step 4: load video, average frames, detect electrode and cells
        preview_video_with_slider(video)

        cap   = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f1, f2 = get_frame_numbers(total)

        frames = [
            average_frames(cap, f1, window=5),
            average_frames(cap, f2, window=5),
        ]
        cap.release()

        # Manual electrode centre identification
        electrode_center = manual_electrode_center(frames[0])

        # Convert frames to RGB for matplotlib display
        Original  = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        processed = []
        centers1, centers2 = [], []

        for idx, frame in enumerate(frames):
            pf, c = detect_particles(frame.copy())
            processed.append(pf)
            if idx == 0:
                centers1 = c
            else:
                centers2 = c

        # Overlay semi-transparent electrode ROI masks on the processed frames
        cx, cy   = electrode_center
        roi_half = int(radius / 4)
        roi      = (cx - roi_half, cy - roi_half, cx + roi_half, cy + roi_half)

        mask_inner = circular_annulus_mask(
            frames[0].shape[:2], electrode_center, radius, 20*SCALE)
        mask_outer = circular_annulus_mask(
            frames[0].shape[:2], electrode_center, radius_outer, 20*SCALE)

        for pf in processed:
            for mask in [mask_inner, mask_outer]:
                overlay     = pf.copy()
                color       = np.zeros_like(pf)
                color[mask] = (255, 0, 0)
                pf[:]       = np.where(mask[..., None],
                                       cv2.addWeighted(overlay, 0.6, color, 0.4, 0), pf)
            cv2.rectangle(pf, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 3)

        # --- DEP contrast index computation ---
        gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)

        m1_c, m2_c, h1_c, h2_c = cumulative_intensity_shift_square(gray1, gray2, roi)
        m1_i, m2_i, h1_i, h2_i = cumulative_intensity_shift_circular(
            gray1, gray2, electrode_center, radius, 20*SCALE)
        m1_o, m2_o, h1_o, h2_o = cumulative_intensity_shift_circular(
            gray1, gray2, electrode_center, radius_outer, 20*SCALE)

        # Background illumination drift correction.
        # The top-left 50x50 pixel region is assumed free of cells and electrode.
        bg1 = gray1[1:50, 1:50].astype(np.float64)
        bg2 = gray2[1:50, 1:50].astype(np.float64)
        bg_contrast = (bg2 - bg1) / (bg1 + 1e-9)
        sigma_bruit = max(np.std(bg_contrast), 0.005)
        bg_drift    = np.mean(bg_contrast)

        eps       = 1e-9
        # Background-corrected optical contrast per ROI.
        # Sign convention: C < 0 => image darkening => cell accumulation (nDEP).
        C_center  = (m2_c - m1_c) / (m1_c + eps) - bg_drift
        C_inner   = (m2_i - m1_i) / (m1_i + eps) - bg_drift
        C_outer   = (m2_o - m1_o) / (m1_o + eps) - bg_drift
        C_rings   = 0.5 * (C_inner + C_outer)
        # Redistribution signal: negative => nDEP, positive => pDEP
        numerator = C_center - C_rings

        # Normalisation floor to prevent saturation on pure noise
        C_floor    = max(3.0 * sigma_bruit, 0.10)
        C_observed = max(abs(C_center), abs(C_rings))
        C_ref      = max(C_observed, C_floor)
        DEP_index  = np.clip(numerator / C_ref, -1.0, 1.0)

        DEP_center_local = C_center * 100.0
        DEP_inner_local  = -C_inner * 100.0
        DEP_outer_local  = -C_outer * 100.0

        print(f"\n[RESULTS] C_center={C_center:.4f} | C_rings={C_rings:.4f} | "
              f"numerator={numerator:.4f} | DEP_index={DEP_index:.4f}")

        # Step 5: contrast figure — save to disk before displaying
        fig_contrast = make_contrast_figure(
            m1_c, m2_c, m1_i, m2_i, m1_o, m2_o,
            h1_c, h2_c, h1_i, h2_i, h1_o, h2_o,
            C_center, C_inner, C_outer,
            DEP_center_local, DEP_inner_local, DEP_outer_local,
            DEP_index, C_ref, C_floor, numerator, sigma_bruit,
            basename)

        contrast_path = os.path.join(result_folder, f"{basename}-CONTRASTE.svg")
        fig_contrast.savefig(contrast_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Contrast figure -> {contrast_path}")

        plt.show(block=True)    # blocking: closes when user presses Enter

        # Steps 8-9: density figure — save to disk before displaying
        fig_density, hist_data = make_density_figure(
            frames, Original, processed, centers1, centers2,
            f1, f2, electrode_center, basename)

        density_path = os.path.join(result_folder, f"{basename}-DENSITY.svg")
        fig_density.savefig(density_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Density figure -> {density_path}")

        plt.show(block=True)    # blocking: closes when user presses Enter

        # Steps 6 + 10: extract histogram arrays and write the text file
        def extract_hist(hd):
            """Return (counts, bin_edges) or (None, None) if the data is empty."""
            if not hd or len(hd) < 2:
                return None, None
            counts, bin_edges = hd[0], hd[1]
            if counts is None or len(counts) == 0:
                return None, None
            if bin_edges is None or len(bin_edges) == 0:
                return None, None
            return counts, bin_edges

        counts_before, bins_before = extract_hist(hist_data[0])
        counts_after,  _           = extract_hist(hist_data[1])

        save_text_results(
            result_folder, basename, meta,
            sigma_bruit, bg_drift,
            C_center, C_inner, C_outer, C_rings,
            numerator, C_floor, C_ref, DEP_index,
            m1_c, m2_c, m1_i, m2_i, m1_o, m2_o,
            DEP_center_local, DEP_inner_local, DEP_outer_local,
            hist_before=counts_before,
            hist_after=counts_after,
            bin_edges=bins_before
        )

        # Step 11: continue or stop
        if not ask_continue():
            print("[INFO] Analysis complete — stopping.")
            break

    print("[INFO] Pipeline finished.")