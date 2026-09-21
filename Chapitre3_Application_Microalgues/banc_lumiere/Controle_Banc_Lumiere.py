import tkinter as tk
import serial
import serial.tools.list_ports
import threading
import time

BAUD = 115200

WAVELENGTH_COLORS = {
    410: "#8B00FF", 435: "#6A0DAD", 460: "#0000FF",
    485: "#00BFFF", 510: "#00FF00", 535: "#7FFF00",
    560: "#ADFF2F", 585: "#FFD700", 610: "#FF8C00",
    645: "#FF4500", 680: "#FF0000", 705: "#DC143C",
    730: "#B22222", 760: "#8B0000", 810: "#800000",
    860: "#4B0000", 900: "#300000", 940: "#1a0000"
}

PAR_CHANNELS = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680]
CONV_FACTOR  = 0.0185

class ArduinoController:
    def __init__(self, root):
        self.root = root
        self.root.title("Contrôle Arduino — Lumière & Spectre")
        self.root.geometry("950x620")
        self.root.configure(bg="#1e1e2e")

        self.ser       = None
        self.led_on    = False
        self.running   = True
        self.port_var  = tk.StringVar()
        self.integ_var = tk.IntVar(value=10)
        self.last_wl   = []
        self.last_vals = []

        self._build_ui()
        self._refresh_ports()
        threading.Thread(target=self._read_serial, daemon=True).start()

    def _build_ui(self):
        left = tk.Frame(self.root, bg="#1e1e2e", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)
        left.pack_propagate(False)

        tk.Label(left, text="Contrôle Arduino",
                 font=("Helvetica", 14, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

        fp = tk.Frame(left, bg="#1e1e2e")
        fp.pack(pady=5)
        tk.Label(fp, text="Port :", bg="#1e1e2e",
                 fg="#cdd6f4", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.port_menu = tk.OptionMenu(fp, self.port_var, "")
        self.port_menu.config(width=10, bg="#313244", fg="#cdd6f4",
                              font=("Helvetica", 10), bd=0)
        self.port_menu.pack(side=tk.LEFT, padx=5)
        tk.Button(fp, text="↺", command=self._refresh_ports,
                  bg="#45475a", fg="#cdd6f4",
                  bd=0, padx=6, cursor="hand2").pack(side=tk.LEFT)

        tk.Button(left, text="Se connecter", command=self._connect,
                  bg="#89b4fa", fg="#1e1e2e",
                  font=("Helvetica", 11, "bold"),
                  bd=0, padx=15, pady=6, cursor="hand2").pack(pady=8)

        self.status_lbl = tk.Label(left, text="⚪ Non connecté",
                                    bg="#1e1e2e", fg="#6c7086",
                                    font=("Helvetica", 9))
        self.status_lbl.pack()

        tk.Frame(left, bg="#45475a", height=1).pack(fill=tk.X, pady=8)

        self.led_btn = tk.Button(left, text="💡 LED : OFF",
                                  command=self._toggle_led,
                                  bg="#f38ba8", fg="#1e1e2e",
                                  font=("Helvetica", 12, "bold"),
                                  bd=0, width=18, height=2, cursor="hand2")
        self.led_btn.pack(pady=4)

        tk.Button(left, text="📊 Mesurer Lux", command=self._measure_lux,
                  bg="#89dceb", fg="#1e1e2e",
                  font=("Helvetica", 12, "bold"),
                  bd=0, width=18, height=2, cursor="hand2").pack(pady=4)

        self.lux_lbl = tk.Label(left, text="— lux",
                                 font=("Helvetica", 14, "bold"),
                                 bg="#1e1e2e", fg="#a6e3a1")
        self.lux_lbl.pack(pady=3)

        tk.Frame(left, bg="#45475a", height=1).pack(fill=tk.X, pady=8)

        tk.Label(left, text="Intégration (× 2.78ms) :",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 9)).pack()

        self.integ_lbl = tk.Label(
            left,
            text=f"{self.integ_var.get()} cycles = {self.integ_var.get()*2.78:.1f}ms",
            bg="#1e1e2e", fg="#fab387", font=("Helvetica", 9))
        self.integ_lbl.pack()

        tk.Scale(left, from_=1, to=100, orient=tk.HORIZONTAL,
                 variable=self.integ_var,
                 command=self._update_integ_lbl,
                 bg="#1e1e2e", fg="#cdd6f4",
                 highlightthickness=0, troughcolor="#313244",
                 length=220).pack()

        tk.Button(left, text="✅ Appliquer", command=self._send_integration,
                  bg="#fab387", fg="#1e1e2e",
                  font=("Helvetica", 10, "bold"),
                  bd=0, padx=10, pady=4, cursor="hand2").pack(pady=4)

        tk.Button(left, text="🌈 Mesurer Spectre",
                  command=self._measure_spectre,
                  bg="#cba6f7", fg="#1e1e2e",
                  font=("Helvetica", 12, "bold"),
                  bd=0, width=18, height=2, cursor="hand2").pack(pady=4)

        self.ppfd_lbl = tk.Label(left, text="PPFD : — µmol/m²/s",
                                  font=("Helvetica", 10),
                                  bg="#1e1e2e", fg="#fab387")
        self.ppfd_lbl.pack(pady=2)

        tk.Frame(left, bg="#45475a", height=1).pack(fill=tk.X, pady=6)
        tk.Label(left, text="Console :", bg="#1e1e2e",
                 fg="#6c7086", font=("Helvetica", 9)).pack(anchor=tk.W)
        self.log = tk.Text(left, height=5, width=32,
                           bg="#313244", fg="#cdd6f4",
                           font=("Courier", 8), bd=0, state=tk.DISABLED)
        self.log.pack()

        right = tk.Frame(self.root, bg="#1e1e2e")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=15, padx=10)

        tk.Label(right, text="Spectre lumineux",
                 font=("Helvetica", 13, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=5)

        self.canvas = tk.Canvas(right, bg="#313244", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self._draw_empty()

    def _draw_empty(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()  or 600
        h = self.canvas.winfo_height() or 400
        self.canvas.create_text(w//2, h//2,
                                 text="En attente de mesure...",
                                 fill="#6c7086", font=("Helvetica", 12))

    def _redraw(self):
        if self.last_vals:
            self._plot_spectre(self.last_wl, self.last_vals)
        else:
            self._draw_empty()

    def _plot_spectre(self, wavelengths, values):
        self.last_wl   = wavelengths
        self.last_vals = values
        self.canvas.delete("all")

        w = self.canvas.winfo_width()  or 600
        h = self.canvas.winfo_height() or 400
        pad_l, pad_r, pad_t, pad_b = 55, 20, 20, 45
        graph_w = w - pad_l - pad_r
        graph_h = h - pad_t - pad_b
        max_val = max(values) if max(values) > 0 else 1
        n       = len(wavelengths)
        bar_w   = max(4, graph_w // n - 3)

        # Axes
        self.canvas.create_line(pad_l, pad_t, pad_l, pad_t + graph_h,
                                fill="#6c7086", width=1)
        self.canvas.create_line(pad_l, pad_t + graph_h,
                                pad_l + graph_w, pad_t + graph_h,
                                fill="#6c7086", width=1)

        # Graduations Y
        for i in range(5):
            y_val = max_val * i / 4
            y_px  = pad_t + graph_h - int(graph_h * i / 4)
            self.canvas.create_line(pad_l - 4, y_px, pad_l, y_px,
                                    fill="#6c7086", width=1)
            self.canvas.create_text(pad_l - 6, y_px,
                                    text=f"{y_val:.1f}",
                                    fill="#6c7086",
                                    font=("Helvetica", 7), anchor="e")

        # Barres
        for i, (wl, val) in enumerate(zip(wavelengths, values)):
            x_center = pad_l + int((i + 0.5) * graph_w / n)
            bar_h    = int(graph_h * val / max_val)
            x0 = x_center - bar_w // 2
            x1 = x_center + bar_w // 2
            y0 = pad_t + graph_h - bar_h
            y1 = pad_t + graph_h
            color = WAVELENGTH_COLORS.get(wl, "#ffffff")
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         fill=color, outline="#1e1e2e", width=1)
            if i % 2 == 0:
                self.canvas.create_text(x_center, pad_t + graph_h + 12,
                                        text=str(wl), fill="#6c7086",
                                        font=("Helvetica", 7),
                                        angle=45, anchor="ne")

        # Ligne de connexion
        pts = []
        for i, (wl, val) in enumerate(zip(wavelengths, values)):
            x = pad_l + int((i + 0.5) * graph_w / n)
            y = pad_t + graph_h - int(graph_h * val / max_val)
            pts.extend([x, y])
        if len(pts) >= 4:
            self.canvas.create_line(*pts, fill="#cdd6f4", width=1, smooth=True)

        # Annotation max
        max_idx = values.index(max(values))
        x_max   = pad_l + int((max_idx + 0.5) * graph_w / n)
        y_max   = pad_t + graph_h - int(graph_h * values[max_idx] / max_val)
        self.canvas.create_oval(x_max-4, y_max-4, x_max+4, y_max+4,
                                fill="#f9e2af", outline="")
        self.canvas.create_text(x_max + 8, y_max - 10,
                                text=f"Max: {wavelengths[max_idx]}nm\n{values[max_idx]:.2f}",
                                fill="#f9e2af", font=("Helvetica", 8), anchor="w")

        # Titres
        self.canvas.create_text(w//2, h - 5,
                                 text="Longueur d'onde (nm)",
                                 fill="#6c7086", font=("Helvetica", 9))
        self.canvas.create_text(10, h//2, text="Intensité",
                                 fill="#6c7086", font=("Helvetica", 9), angle=90)

    def _update_integ_lbl(self, val):
        c = int(val)
        self.integ_lbl.config(text=f"{c} cycles = {c*2.78:.1f}ms")

    def _send_integration(self):
        if not self.ser:
            self.status_lbl.config(text="❌ Non connecté !", fg="#f38ba8")
            return
        c = self.integ_var.get()
        self.ser.write(f"I{c}\n".encode())
        self._log(f"Intégration : {c} cycles ({c*2.78:.1f}ms)")

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        menu  = self.port_menu["menu"]
        menu.delete(0, tk.END)
        if ports:
            for p in ports:
                menu.add_command(label=p,
                                  command=lambda v=p: self.port_var.set(v))
            self.port_var.set(ports[0])
        else:
            self.port_var.set("Aucun port")

    def _connect(self):
        port = self.port_var.get()
        if not port or port == "Aucun port":
            self.status_lbl.config(text="❌ Aucun port", fg="#f38ba8")
            return
        try:
            if self.ser:
                self.ser.close()
            self.ser = serial.Serial(port, BAUD, timeout=1)
            time.sleep(2)
            self.status_lbl.config(text=f"🟢 Connecté {port}", fg="#a6e3a1")
            self._log(f"Connecté sur {port}")
        except Exception as e:
            self.status_lbl.config(text=f"❌ {e}", fg="#f38ba8")

    def _toggle_led(self):
        if not self.ser:
            self.status_lbl.config(text="❌ Non connecté !", fg="#f38ba8")
            return
        if self.led_on:
            self.ser.write(b'0')
            self.led_on = False
            self.led_btn.config(text="💡 LED : OFF", bg="#f38ba8")
        else:
            self.ser.write(b'1')
            self.led_on = True
            self.led_btn.config(text="💡 LED : ON", bg="#a6e3a1")

    def _measure_lux(self):
        if not self.ser:
            self.status_lbl.config(text="❌ Non connecté !", fg="#f38ba8")
            return
        self.lux_lbl.config(text="Mesure...")
        self.ser.write(b'3')

    def _measure_spectre(self):
        if not self.ser:
            self.status_lbl.config(text="❌ Non connecté !", fg="#f38ba8")
            return
        w = self.canvas.winfo_width() or 600
        h = self.canvas.winfo_height() or 400
        self.canvas.delete("all")
        self.canvas.create_text(w//2, h//2,
                                 text="Mesure en cours...",
                                 fill="#fab387", font=("Helvetica", 12))
        self.ser.write(b'4')

    def _parse_spectre(self, line):
        try:
            data  = line.replace("SPECTRE:", "")
            pairs = data.split("|")
            wls, vals = [], []
            for pair in pairs:
                wl, val = pair.split(":")
                wls.append(int(wl))
                vals.append(float(val))
            ppfd = sum(v for wl, v in zip(wls, vals)
                       if wl in PAR_CHANNELS) * CONV_FACTOR
            self.root.after(0, lambda p=ppfd:
                self.ppfd_lbl.config(text=f"PPFD : {p:.2f} µmol/m²/s"))
            self.root.after(0, lambda w=wls, v=vals:
                self._plot_spectre(w, v))
        except Exception as e:
            self._log(f"Erreur parsing: {e}")

    def _read_serial(self):
        while self.running:
            try:
                if self.ser and self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8').strip()
                    if not line:
                        continue
                    self._log(line)

                    # ✅ Format souple — capture tout ce qui contient "lux"
                    if "lux" in line.lower() and "spectre" not in line.lower():
                        # Extrait le nombre dans la ligne
                        import re
                        match = re.search(r"[\d]+\.?[\d]*", line)
                        if match:
                            val = match.group()
                            self.root.after(0, lambda v=val:
                                self.lux_lbl.config(text=f"🌟 {v} lux"))

                    elif line.startswith("SPECTRE:"):
                        self.root.after(0, lambda l=line:
                            self._parse_spectre(l))

                    elif line.startswith("LUX:"):
                        val = line.replace("LUX:", "")
                        self.root.after(0, lambda v=val:
                            self.lux_lbl.config(text=f"🌟 {v} lux"))

            except Exception:
                pass
            time.sleep(0.05)

    def _log(self, msg):
        def _upd():
            self.log.config(state=tk.NORMAL)
            self.log.insert(tk.END, f"{msg}\n")
            self.log.see(tk.END)
            self.log.config(state=tk.DISABLED)
        self.root.after(0, _upd)

    def on_close(self):
        self.running = False
        if self.ser:
            self.ser.close()
        self.root.destroy()

root = tk.Tk()
app  = ArduinoController(root)
root.protocol("WM_DELETE_WINDOW", app.on_close)
root.mainloop()