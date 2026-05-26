# -*- coding: utf-8 -*-
"""
coronary_pipeline_gui.py
Coronary artery segmentation pipeline GUI
Author: Hattori (2026)
"""

import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from postprocess import run_postprocess

MODELS = {
    "nnUNet": {
        "trainer": None,
        "extra_flags": [],
    },
    "U-Mamba (Bot)": {
        "trainer": "nnUNetTrainerUMambaBot",
        "extra_flags": ["--disable_tta"],
    },
    "U-Mamba (Enc)": {
        "trainer": "nnUNetTrainerUMambaEnc",
        "extra_flags": ["--disable_tta"],
    },
}


def build_predict_command(input_dir, raw_output_dir, dataset_id, config,
                          fold, model_key, checkpoint, save_probabilities):
    info = MODELS[model_key]
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", raw_output_dir,
        "-d", dataset_id,
        "-c", config,
        "-f", fold,
        "-chk", checkpoint,
    ]
    if info["trainer"]:
        cmd += ["-tr", info["trainer"]]
    cmd += info["extra_flags"]
    if save_probabilities:
        cmd.append("--save_probabilities")
    return cmd


def pipeline_worker(input_dir, output_dir, dataset_id, config, fold,
                    model_key, checkpoint, save_probabilities,
                    run_postproc, log_queue):
    def emit(msg):
        log_queue.put(msg)

    try:
        raw_out = str(Path(output_dir) / "raw_predictions")
        Path(raw_out).mkdir(parents=True, exist_ok=True)

        cmd = build_predict_command(input_dir, raw_out, dataset_id, config,
                                    fold, model_key, checkpoint,
                                    save_probabilities)
        emit("=" * 50)
        emit("STEP 1: nnUNetv2_predict")
        emit("Command: " + " ".join(cmd))
        emit("=" * 50)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            emit(line.rstrip())
        proc.wait()

        if proc.returncode != 0:
            emit(f"[ERROR] nnUNetv2_predict failed (code {proc.returncode})")
            return

        emit("nnUNetv2_predict finished.\n")

        if run_postproc:
            processed_out = str(Path(output_dir) / "postprocessed")
            emit("=" * 50)
            emit("STEP 2: Post-processing")
            emit(f"Input : {raw_out}")
            emit(f"Output: {processed_out}")
            emit("=" * 50)
            run_postprocess(raw_out, processed_out, top_n=2,
                            log_callback=emit)
        else:
            emit("Post-processing skipped.")

        emit("\nDone.")

    except FileNotFoundError:
        emit("[ERROR] nnUNetv2_predict not found. "
             "Check that nnUNet is installed and the environment is active.")
    except Exception as e:
        emit(f"[ERROR] {e}")
    finally:
        log_queue.put(None)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coronary Segmentation Pipeline")
        self.resizable(True, True)
        self._log_queue = queue.Queue()
        self._build_ui()
        self._poll()

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # --- Input folder ---
        tk.Label(self, text="Input folder:").grid(
            row=0, column=0, sticky="w", **pad)
        self._input_var = tk.StringVar()
        tk.Entry(self, textvariable=self._input_var, width=50).grid(
            row=0, column=1, sticky="ew", **pad)
        tk.Button(self, text="Browse",
                  command=lambda: self._browse(self._input_var)).grid(
            row=0, column=2, **pad)

        # --- Output folder ---
        tk.Label(self, text="Output folder:").grid(
            row=1, column=0, sticky="w", **pad)
        self._output_var = tk.StringVar()
        tk.Entry(self, textvariable=self._output_var, width=50).grid(
            row=1, column=1, sticky="ew", **pad)
        tk.Button(self, text="Browse",
                  command=lambda: self._browse(self._output_var)).grid(
            row=1, column=2, **pad)

        # --- Model ---
        tk.Label(self, text="Model:").grid(
            row=2, column=0, sticky="w", **pad)
        self._model_var = tk.StringVar(value=list(MODELS.keys())[0])
        model_frame = tk.Frame(self)
        model_frame.grid(row=2, column=1, sticky="w", **pad)
        for name in MODELS:
            tk.Radiobutton(model_frame, text=name,
                           variable=self._model_var,
                           value=name).pack(side="left", padx=4)

        # --- Dataset ID ---
        tk.Label(self, text="Dataset ID:").grid(
            row=3, column=0, sticky="w", **pad)
        self._dataset_var = tk.StringVar(value="101")
        tk.Entry(self, textvariable=self._dataset_var, width=8).grid(
            row=3, column=1, sticky="w", **pad)

        # --- Config ---
        tk.Label(self, text="Config:").grid(
            row=4, column=0, sticky="w", **pad)
        self._config_var = tk.StringVar(value="3d_fullres")
        cfg_frame = tk.Frame(self)
        cfg_frame.grid(row=4, column=1, sticky="w", **pad)
        for c in ["3d_fullres", "3d_lowres", "2d"]:
            tk.Radiobutton(cfg_frame, text=c,
                           variable=self._config_var,
                           value=c).pack(side="left", padx=4)

        # --- Fold ---
        tk.Label(self, text="Fold:").grid(
            row=5, column=0, sticky="w", **pad)
        self._fold_var = tk.StringVar(value="0")
        tk.Entry(self, textvariable=self._fold_var, width=8).grid(
            row=5, column=1, sticky="w", **pad)

        # --- Checkpoint ---
        tk.Label(self, text="Checkpoint:").grid(
            row=6, column=0, sticky="w", **pad)
        self._chk_var = tk.StringVar(value="checkpoint_best.pth")
        chk_frame = tk.Frame(self)
        chk_frame.grid(row=6, column=1, sticky="w", **pad)
        for c in ["checkpoint_best.pth", "checkpoint_final.pth"]:
            tk.Radiobutton(chk_frame, text=c,
                           variable=self._chk_var,
                           value=c).pack(side="left", padx=4)

        # --- Options ---
        self._save_prob_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Save probability maps (--save_probabilities)",
                       variable=self._save_prob_var).grid(
            row=7, column=0, columnspan=3, sticky="w", **pad)

        self._postproc_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Run post-processing (keep top-2 connected components)",
                       variable=self._postproc_var).grid(
            row=8, column=0, columnspan=3, sticky="w", **pad)

        # --- Buttons ---
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=6)
        self._run_btn = tk.Button(btn_frame, text="Run",
                                  width=12, command=self._on_run)
        self._run_btn.pack(side="left", padx=6)
        tk.Button(btn_frame, text="Clear log",
                  width=10, command=self._clear_log).pack(side="left", padx=6)

        # --- Log ---
        tk.Label(self, text="Log:").grid(
            row=10, column=0, sticky="nw", **pad)
        self._log = scrolledtext.ScrolledText(
            self, width=70, height=18, state="disabled",
            font=("Courier", 9))
        self._log.grid(row=10, column=1, columnspan=2,
                       sticky="nsew", **pad)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(10, weight=1)

    def _browse(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _emit(self, msg):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg is None:
                    self._run_btn.configure(state="normal")
                else:
                    self._emit(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _on_run(self):
        input_dir = self._input_var.get().strip()
        output_dir = self._output_var.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            messagebox.showerror("Error", "Please select a valid input folder.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        self._run_btn.configure(state="disabled")
        self._emit(f"--- Start [{self._model_var.get()}] ---")

        threading.Thread(
            target=pipeline_worker,
            kwargs=dict(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset_id=self._dataset_var.get().strip(),
                config=self._config_var.get(),
                fold=self._fold_var.get(),
                model_key=self._model_var.get(),
                checkpoint=self._chk_var.get(),
                save_probabilities=self._save_prob_var.get(),
                run_postproc=self._postproc_var.get(),
                log_queue=self._log_queue,
            ),
            daemon=True,
        ).start()


if __name__ == "__main__":
    App().mainloop()
