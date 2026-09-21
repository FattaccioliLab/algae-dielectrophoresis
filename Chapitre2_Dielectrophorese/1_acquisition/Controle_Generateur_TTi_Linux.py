"""
Controleur GBF - TTi TGF4162 (TGF4000 Series)
Version LINUX

Communication : pyserial direct sur port USB virtuel TTi
Port typique : /dev/ttyUSB0 ou /dev/ttyACM0

Droits serie requis (une seule fois) :
  sudo usermod -aG dialout $USER
  puis deconnexion/reconnexion

Commandes TTi TGF4000 :
  CHN <1|2>          selectionne le canal actif
  WAVE <type>        SINE SQUARE RAMP PULSE NOISE ARB
  FREQ <Hz>          frequence en Hz
  AMPL <Vpp>         amplitude peak-to-peak
  OUTPUT ON|OFF      active/desactive la sortie
  *IDN?              identification
"""

import serial
import serial.tools.list_ports
import tkinter as tk
import time
import threading
import queue
import os
import glob
import subprocess

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
PURPLE   = "#d2a8ff"
DANGER   = "#f85149"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
BTN_BG   = "#21262d"
BTN_HOV  = "#30363d"
MON_BG   = "#0a1a0a"

ACCENT_CH  = {1: ACCENT, 2: PURPLE}
WAVES      = ["SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "ARB"]
PRESET_MHZ = [1, 10, 20, 30, 40, 50]

# ── Etat global ───────────────────────────────────────────────────────────────
ser       = None
cmd_queue = queue.Queue()


# ── Detection automatique du port TTi sous Linux ──────────────────────────────
def _detect_tti_port():
    """
    Cherche le port USB TTi automatiquement.
    Le driver TTi cree un port /dev/ttyUSB* ou /dev/ttyACM*.
    Retourne le premier trouve, ou /dev/ttyUSB0 par defaut.
    """
    # Cherche via lsusb les peripheriques TTi (Vendor ID 0x103E = Thurlby Thandar)
    candidates = []
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=3)
        tti_found = any("103e" in line.lower() or "thurlby" in line.lower() or
                        "tti" in line.lower() or "thandar" in line.lower()
                        for line in result.stdout.splitlines())
    except Exception:
        tti_found = False

    # Parcourt tous les ports serie disponibles
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        # TTi Virtual COM : vendor ID 0x103E
        if "103e" in hwid or "thurlby" in desc or "tti" in desc or "thandar" in desc:
            candidates.insert(0, p.device)   # priorite maximale
        elif p.device.startswith("/dev/ttyUSB") or p.device.startswith("/dev/ttyACM"):
            candidates.append(p.device)

    return candidates[0] if candidates else "/dev/ttyUSB0"


# ── Thread serie unique ───────────────────────────────────────────────────────
def _worker():
    while True:
        item = cmd_queue.get()
        if item is None:
            break
        cmd, want_reply, holder, evt = item
        try:
            if ser is not None and ser.is_open:
                ser.reset_input_buffer()
                ser.write((cmd + "\n").encode("ascii"))
                ser.flush()
                if want_reply:
                    time.sleep(0.12)
                    raw = ser.readline()
                    holder["v"] = raw.decode("ascii", errors="replace").strip()
                else:
                    time.sleep(0.08)
        except Exception as e:
            holder["v"] = "ERR"
            root.after(0, log, f"[SERIE] {cmd!r} -> {e}")
        finally:
            if evt is not None:
                evt.set()
        cmd_queue.task_done()


def _send(cmd):
    cmd_queue.put((cmd, False, {}, None))

def _ask(cmd, timeout=5.0):
    h = {"v": "ERR"}
    e = threading.Event()
    cmd_queue.put((cmd, True, h, e))
    e.wait(timeout)
    return h["v"]


# ── Connexion ─────────────────────────────────────────────────────────────────
def connect():
    global ser
    port = port_var.get().strip()
    log(f"Connexion sur {port}...")

    # Verifier les droits d'acces au port
    if not os.access(port, os.R_OK | os.W_OK):
        log(f"ERREUR : pas les droits sur {port}")
        log("  Executez dans un terminal :")
        log(f"  sudo chmod a+rw {port}")
        log("  Ou de facon permanente :")
        log("  sudo usermod -aG dialout $USER  (puis deconnexion)")
        return

    try:
        ser = serial.Serial(
            port     = port,
            baudrate = 9600,       # ignore par le driver TTi USB, valeur conventionnelle
            bytesize = serial.EIGHTBITS,
            parity   = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE,
            timeout  = 2.0,
            xonxoff  = False,
            rtscts   = False,
            dsrdtr   = False,
        )
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        idn = _ask("*IDN?", timeout=5.0)
        if idn and idn != "ERR":
            log(f"Connecte : {idn}")
            connect_btn.config(text="Connecte", bg=ACCENT2)
        else:
            log("ATTENTION : pas de reponse a *IDN?")
            log("  Verifiez que le GBF est allume et le cable USB branche")
            connect_btn.config(text="Connecte?", bg="#b08020")

    except serial.SerialException as e:
        ser = None
        log(f"ERREUR : {e}")
        log("  -> Verifiez le port avec 'Lister ports'")
        log("  -> dmesg | grep tty  pour voir le port assigne")


def disconnect():
    global ser
    while not cmd_queue.empty():
        try: cmd_queue.get_nowait()
        except: break
    if ser and ser.is_open:
        try: ser.close()
        except: pass
    ser = None
    connect_btn.config(text="Connecter", bg=BTN_BG)
    log("Deconnecte.")


def list_ports():
    """Affiche les ports serie disponibles avec leur description."""
    ports = serial.tools.list_ports.comports()
    if ports:
        log("Ports serie detectes :")
        for p in sorted(ports):
            log(f"  {p.device:<20}  {p.description}  [{p.hwid}]")
    else:
        log("Aucun port serie detecte.")
    # Afficher aussi les tty bruts
    tty_list = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if tty_list:
        log("Ports /dev/tty* presents : " + "  ".join(tty_list))


def auto_detect():
    """Detecte automatiquement le port TTi et le met dans le champ."""
    port = _detect_tti_port()
    port_var.set(port)
    log(f"Port detecte : {port}")
    # Afficher les droits actuels
    if os.path.exists(port):
        readable = os.access(port, os.R_OK)
        writable = os.access(port, os.W_OK)
        log(f"  Droits : lecture={'OK' if readable else 'MANQUANT'}  "
            f"ecriture={'OK' if writable else 'MANQUANT'}")
        if not (readable and writable):
            log(f"  Corriger : sudo chmod a+rw {port}")


# ── Commandes canal ───────────────────────────────────────────────────────────
def apply_ch(ch):
    if not ser or not ser.is_open:
        log("Non connecte !"); return
    wave = wave_var[ch].get()
    try:
        vpp  = float(vpp_var[ch].get())
        freq = float(freq_var[ch].get())
    except ValueError:
        log(f"CH{ch} : valeur invalide"); return

    _send(f"CHN {ch}")
    _send(f"WAVE {wave}")
    _send(f"FREQ {freq:.6f}")
    _send(f"AMPL {vpp:.4f}")

    fmhz = freq / 1e6
    log(f"CH{ch} -> {wave}  {fmhz:.6f} MHz  {vpp:.3f} Vpp")
    _update_status_local(ch, wave, freq, vpp)


def set_out(ch, state):
    if not ser or not ser.is_open:
        log("Non connecte !"); return
    _send(f"CHN {ch}")
    _send(f"OUTPUT {'ON' if state else 'OFF'}")
    _btn_color(ch, state)
    log(f"CH{ch} -> OUTPUT {'ON' if state else 'OFF'}")


def toggle_out(ch):
    current = out_btn[ch].cget("text").startswith("ON")
    set_out(ch, not current)


def preset_freq(ch, mhz):
    freq_var[ch].set(str(int(mhz * 1e6)))
    apply_ch(ch)


def _btn_color(ch, state):
    c = ACCENT2 if state else DANGER
    out_btn[ch].config(bg=c, text="ON  *" if state else "OFF o")


# ── Affichage statut ──────────────────────────────────────────────────────────
def _update_status_local(ch, wave, freq, vpp):
    try:
        fhz = float(freq)
        if   fhz >= 1e6: fstr = f"{fhz/1e6:.6f} MHz"
        elif fhz >= 1e3: fstr = f"{fhz/1e3:.3f} kHz"
        else:            fstr = f"{fhz:.3f} Hz"
    except:
        fstr = str(freq)
    _write_status(ch, [
        f"  Frequence  :  {fstr}",
        f"  Amplitude  :  {vpp} Vpp",
        f"  Forme      :  {wave}",
        f"  Sortie     :  (cliquer Lire pour verifier)",
    ])


def read_ch(ch):
    if not ser or not ser.is_open:
        log("Non connecte !"); return
    log(f"CH{ch} : lecture...")
    _write_status(ch, ["  Lecture en cours..."])

    def _do():
        _send(f"CHN {ch}")
        time.sleep(0.25)
        freq = _ask("FREQ?")
        ampl = _ask("AMPL?")
        wave = _ask("WAVE?")
        outp = _ask("OUTPUT?")
        root.after(0, _show_read, ch, freq, ampl, wave, outp)

    threading.Thread(target=_do, daemon=True).start()


def _show_read(ch, freq, ampl, wave, outp):
    try:
        fhz = float(freq)
        if   fhz >= 1e6: fstr = f"{fhz/1e6:.6f} MHz"
        elif fhz >= 1e3: fstr = f"{fhz/1e3:.3f} kHz"
        else:            fstr = f"{fhz:.3f} Hz"
    except:
        fstr = str(freq)
    _write_status(ch, [
        f"  Frequence  :  {fstr}",
        f"  Amplitude  :  {ampl} Vpp",
        f"  Forme      :  {wave}",
        f"  Sortie     :  {outp}",
    ])
    state = str(outp).strip().upper() in ("ON", "1")
    _btn_color(ch, state)
    log(f"CH{ch} lu -> {fstr}  {ampl} Vpp  {wave}  OUT={outp}")


def _write_status(ch, lines):
    status_text[ch].config(state="normal")
    status_text[ch].delete("1.0", "end")
    status_text[ch].insert("end", "\n".join(lines))
    status_text[ch].config(state="disabled")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    log_box.config(state="normal")
    log_box.insert("end", f"[{ts}]  {msg}\n")
    log_box.see("end")
    log_box.config(state="disabled")


# ════════════════════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("TTi TGF4162 -- Controller  [Linux]")
root.configure(bg=BG)
root.minsize(860, 640)

F_HEAD  = ("Monospace", 11, "bold")
F_LABEL = ("Monospace", 9)
F_SMALL = ("Monospace", 8)
F_MONO  = ("Monospace", 9)


def dframe(parent, **kw):
    return tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                    highlightthickness=1, **kw)

def lbl(parent, text, fg=TEXT_DIM, f=("Monospace", 8)):
    return tk.Label(parent, text=text, bg=PANEL, fg=fg, font=f)

def hdiv(parent):
    return tk.Frame(parent, bg=BORDER, height=1)

def mkbtn(parent, text, cmd, bg=BTN_BG, fg=TEXT, w=None, f=("Monospace", 8)):
    kw2 = {"width": w} if w else {}
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=f,
                  relief="flat", bd=0, padx=6, pady=4, cursor="hand2",
                  activebackground=BTN_HOV, activeforeground=TEXT, **kw2)
    b.bind("<Enter>", lambda e, _b=b: _b.config(bg=BTN_HOV))
    b.bind("<Leave>", lambda e, _b=b, _bg=bg: _b.config(bg=_bg))
    return b


# ── Top bar ───────────────────────────────────────────────────────────────────
top = dframe(root)
top.pack(fill="x", padx=8, pady=(8, 4))
tk.Label(top, text="  TTi TGF4162  CONTROLLER",
         bg=PANEL, fg=ACCENT, font=("Monospace", 12, "bold")).pack(side="left", padx=8, pady=8)

cr = tk.Frame(top, bg=PANEL)
cr.pack(side="right", padx=10, pady=6)

tk.Label(cr, text="Port :", bg=PANEL, fg=TEXT_DIM, font=F_SMALL).pack(side="left")
port_var = tk.StringVar(value="/dev/ttyUSB0")
tk.Entry(cr, textvariable=port_var, bg=BTN_BG, fg=TEXT, font=F_MONO,
         width=16, insertbackground=TEXT, relief="flat").pack(side="left", padx=(4, 6))

connect_btn = mkbtn(cr, "Connecter", connect, bg=BTN_BG, fg=ACCENT, f=F_LABEL)
connect_btn.pack(side="left", padx=2)
mkbtn(cr, "Deconnecter", disconnect, bg=BTN_BG, fg=DANGER, f=F_LABEL).pack(side="left", padx=2)
mkbtn(cr, "Lister",      list_ports,  f=F_SMALL).pack(side="left", padx=2)
mkbtn(cr, "Auto-detect", auto_detect, bg=BTN_BG, fg=PURPLE, f=F_SMALL).pack(side="left", padx=2)


# ── Channels ──────────────────────────────────────────────────────────────────
ch_area = tk.Frame(root, bg=BG)
ch_area.pack(fill="both", expand=True, padx=8, pady=4)

wave_var    = {}
vpp_var     = {}
freq_var    = {}
out_btn     = {}
status_text = {}

for ch in (1, 2):
    col = ch - 1
    acc = ACCENT_CH[ch]

    cf = dframe(ch_area)
    cf.grid(row=0, column=col,
            padx=(0, 4) if ch == 1 else (4, 0), sticky="nsew")
    ch_area.columnconfigure(col, weight=1)
    ch_area.rowconfigure(0, weight=1)

    hdr = tk.Frame(cf, bg=PANEL)
    hdr.pack(fill="x", padx=10, pady=(10, 4))
    tk.Label(hdr, text=f"CHANNEL  {ch}", bg=PANEL, fg=acc,
             font=("Monospace", 11, "bold")).pack(side="left")
    out_btn[ch] = tk.Button(hdr, text="OFF o", bg=DANGER, fg=TEXT,
                             font=("Monospace", 9, "bold"), relief="flat", bd=0,
                             padx=10, pady=3, cursor="hand2",
                             command=lambda c=ch: toggle_out(c))
    out_btn[ch].pack(side="right")

    hdiv(cf).pack(fill="x", padx=10, pady=4)
    body = tk.Frame(cf, bg=PANEL)
    body.pack(fill="both", expand=True, padx=14, pady=4)

    # Waveform
    lbl(body, "FORME D'ONDE").pack(anchor="w")
    wave_var[ch] = tk.StringVar(value="SINE")
    wf = tk.Frame(body, bg=PANEL)
    wf.pack(fill="x", pady=(2, 8))
    for w in WAVES:
        rb = tk.Radiobutton(wf, text=w, variable=wave_var[ch], value=w,
                            bg=PANEL, fg=TEXT, selectcolor=BTN_BG,
                            activebackground=PANEL, font=F_SMALL,
                            indicatoron=False, relief="flat", bd=0,
                            padx=7, pady=3, cursor="hand2",
                            highlightthickness=0)
        rb.pack(side="left", padx=1)
        def _style(b=rb, v=wave_var[ch], val=w, a=acc):
            def _t(*_):
                b.config(fg=a     if v.get() == val else TEXT,
                         bg=BTN_BG if v.get() == val else PANEL)
            v.trace_add("write", _t)
        _style()

    hdiv(body).pack(fill="x", pady=4)

    # Amplitude
    lbl(body, "AMPLITUDE PEAK-TO-PEAK").pack(anchor="w", pady=(4, 0))
    vrow = tk.Frame(body, bg=PANEL)
    vrow.pack(fill="x", pady=(2, 4))
    vpp_var[ch] = tk.StringVar(value="1.000")
    tk.Entry(vrow, textvariable=vpp_var[ch], bg=BTN_BG, fg=acc,
             font=("Monospace", 16, "bold"), width=7,
             insertbackground=TEXT, relief="flat", justify="right").pack(side="left", padx=(0, 4))
    tk.Label(vrow, text="Vpp", bg=PANEL, fg=TEXT_DIM, font=F_LABEL).pack(side="left")

    sl_var = tk.DoubleVar(value=1.0)
    sl = tk.Scale(body, from_=0, to=20, resolution=0.01, orient="horizontal",
                   variable=sl_var, bg=PANEL, fg=TEXT_DIM, troughcolor=BTN_BG,
                   highlightthickness=0, sliderrelief="flat",
                   activebackground=acc, font=F_SMALL,
                   command=lambda v, c=ch: vpp_var[c].set(f"{float(v):.3f}"))
    sl.pack(fill="x", pady=(0, 8))
    def _vsync(v, c=ch, s=sl_var):
        try: s.set(float(vpp_var[c].get()))
        except: pass
    vpp_var[ch].trace_add("write", _vsync)

    hdiv(body).pack(fill="x", pady=4)

    # Presets MHz
    lbl(body, "FREQUENCES RAPIDES (MHz)").pack(anchor="w", pady=(4, 2))
    pf = tk.Frame(body, bg=PANEL)
    pf.pack(fill="x", pady=(0, 8))
    for mhz in PRESET_MHZ:
        mkbtn(pf, f"{mhz} MHz",
              lambda c=ch, m=mhz: preset_freq(c, m), w=6).pack(side="left", padx=2)

    # Frequence libre
    lbl(body, "FREQUENCE (Hz)  --  [Entree] pour valider").pack(anchor="w")
    frow = tk.Frame(body, bg=PANEL)
    frow.pack(fill="x", pady=(2, 8))
    freq_var[ch] = tk.StringVar(value="1000000")
    fe = tk.Entry(frow, textvariable=freq_var[ch], bg=BTN_BG, fg=acc,
                   font=("Monospace", 13, "bold"), width=14,
                   insertbackground=TEXT, relief="flat")
    fe.pack(side="left", padx=(0, 4))
    fe.bind("<Return>", lambda e, c=ch: apply_ch(c))
    tk.Label(frow, text="Hz", bg=PANEL, fg=TEXT_DIM, font=F_LABEL).pack(side="left", padx=(0, 8))
    mkbtn(frow, "Appliquer", lambda c=ch: apply_ch(c),
          bg=acc, fg=BG, w=10).pack(side="left")

    hdiv(body).pack(fill="x", pady=4)

    # Status
    lbl(body, "ETAT  (local apres Appliquer  /  reel apres Lire)").pack(anchor="w", pady=(4, 2))
    st = tk.Text(body, height=4, bg=MON_BG, fg=ACCENT2,
                  font=("Monospace", 9), relief="flat",
                  wrap="none", state="disabled")
    st.pack(fill="x", pady=(0, 4))
    status_text[ch] = st

    mkbtn(body, "Lire l'etat reel depuis le GBF",
          lambda c=ch: read_ch(c),
          bg=BTN_BG, fg=TEXT_DIM, f=F_SMALL).pack(anchor="e", pady=(0, 8))


# ── Log ───────────────────────────────────────────────────────────────────────
lf = dframe(root)
lf.pack(fill="both", padx=8, pady=(4, 8))
lhdr = tk.Frame(lf, bg=PANEL)
lhdr.pack(fill="x", padx=10, pady=(6, 2))
tk.Label(lhdr, text="LOG", bg=PANEL, fg=TEXT_DIM, font=F_SMALL).pack(side="left")
mkbtn(lhdr, "Effacer", lambda: (
    log_box.config(state="normal"),
    log_box.delete("1.0", "end"),
    log_box.config(state="disabled")
), f=F_SMALL).pack(side="right")
log_box = tk.Text(lf, height=5, bg=BG, fg=TEXT_DIM,
                   font=("Monospace", 8), relief="flat",
                   wrap="word", state="disabled")
log_box.pack(fill="both", padx=10, pady=(0, 8))


# ── Demarrage ─────────────────────────────────────────────────────────────────
threading.Thread(target=_worker, daemon=True).start()

log("TTi TGF4162 Controller -- Version Linux")
log("Port par defaut : /dev/ttyUSB0")
log("Si erreur de droits : sudo usermod -aG dialout $USER  (puis deconnexion)")
log("Cliquez 'Auto-detect' pour trouver le port automatiquement.")

# Detection automatique au demarrage
root.after(500, auto_detect)

root.mainloop()

cmd_queue.put(None)
if ser and ser.is_open:
    try: ser.close()
    except: pass