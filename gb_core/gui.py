from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .artifacts import get_resource_path, load_bundle
from .contracts import (
    CLASSIFICATION_MODELS,
    PREPROCESSING_METHODS,
    REGRESSION_MODELS,
    SELECTOR_METHODS,
    EvaluationMode,
    PipelineSpec,
    RunConfig,
    TaskType,
    enumerate_specs,
)
from .data import DatasetSplit, load_dataset, split_dataset
from .engine import EvaluationEngine, RunOutputs
from .plotting import compare_models_plot, create_visualization_suite
from .reporting import DEEPSEEK_MODELS, build_deepseek_prompt, call_deepseek, structured_summary, write_report_files


class RedirectText:
    def __init__(self, widget: ScrolledText):
        self.widget = widget
        self.queue: queue.Queue[str] = queue.Queue()
        self.widget.after(100, self._poll)

    def write(self, text: str) -> None:
        if text:
            self.queue.put(text)

    def flush(self) -> None:
        return None

    def _poll(self) -> None:
        try:
            while True:
                text = self.queue.get_nowait()
                self.widget.insert("end", text)
                self.widget.see("end")
        except queue.Empty:
            pass
        try:
            self.widget.after(100, self._poll)
        except tk.TclError:
            pass


class GBApplication:
    def __init__(self, root: tk.Tk, task: TaskType):
        self.root = root
        self.task = TaskType(task)
        self.task_title = "Classification" if self.task is TaskType.CLASSIFICATION else "Regression"
        self.root.title(f"Grand Banquet {self.task_title} v0.0")
        width = max(int(root.winfo_screenwidth() * 0.8), 1000)
        height = max(int(root.winfo_screenheight() * 0.8), 700)
        self.root.geometry(f"{width}x{height}+{(root.winfo_screenwidth()-width)//2}+{(root.winfo_screenheight()-height)//2}")
        self.root.minsize(1000, 700)
        self._setup_theme()
        self._setup_variables()
        self.dataset_split: Optional[DatasetSplit] = None
        self.outputs: Optional[RunOutputs] = None
        self.last_specs: List[PipelineSpec] = []
        self.last_run_data_file: Optional[Path] = None
        self._run_busy = False
        self._export_busy = False
        self._ai_busy = False
        self.stop_requested = False
        self.worker_queue: queue.Queue = queue.Queue()
        self.project_step = "Data loading"
        self.last_saved = "None"
        self._create_gui()
        self.original_stdout = sys.stdout
        sys.stdout = RedirectText(self.console_text)
        self.root.after(100, self._poll_worker_queue)
        self._update_status()
        print(f"Welcome to Grand Banquet {self.task_title} v0.0")

    @property
    def models(self) -> List[str]:
        return list(CLASSIFICATION_MODELS if self.task is TaskType.CLASSIFICATION else REGRESSION_MODELS)

    def _setup_theme(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#E0F7FA")
        style.configure(".", background="#E0F7FA", foreground="#37474F", font=("Segoe UI", 10))
        style.configure("TFrame", background="#FFFFFF")
        style.configure("TLabel", background="#FFFFFF", foreground="#37474F")
        style.configure("TLabelFrame", background="#FFFFFF", bordercolor="#B0BEC5")
        style.configure("TLabelFrame.Label", background="#FFFFFF", foreground="#00796B", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", background="#FF7043", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"), padding=5)
        style.map("TButton", background=[("active", "#F4511E"), ("disabled", "#CFD8DC")], foreground=[("disabled", "#78909C")])
        style.configure("TEntry", fieldbackground="#FFFDE7", foreground="#000000")
        style.configure("TCombobox", fieldbackground="#FFFDE7", foreground="#000000")
        style.configure("TCheckbutton", background="#FFFFFF", foreground="#37474F")
        style.configure("TRadiobutton", background="#FFFFFF", foreground="#37474F")
        style.configure("TNotebook", background="#E0F7FA", borderwidth=0)
        style.configure("TNotebook.Tab", background="#B2EBF2", foreground="#00796B", padding=[12, 6])
        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")], foreground=[("selected", "#EF5350")])
        style.configure("Horizontal.TProgressbar", troughcolor="#B0BEC5", background="#4FC3F7")

    def _setup_variables(self) -> None:
        self.data_file = tk.StringVar()
        self.base_dir = tk.StringVar(value=str(Path.cwd() / f"gb_{self.task.value}_output"))
        self.validation_ratio = tk.StringVar(value="0.2")
        self.mode_var = tk.StringVar(value=EvaluationMode.STANDARD.value)
        self.outer_folds = tk.StringVar(value="5")
        self.inner_folds = tk.StringVar(value="3")
        self.seed_var = tk.StringVar(value="42")
        self.api_key_var = tk.StringVar()
        self.deepseek_model_var = tk.StringVar(value=DEEPSEEK_MODELS[0])
        self.export_mode_var = tk.StringVar(value="id")
        self.export_id_var = tk.StringVar()
        self.export_metric_var = tk.StringVar(value="CV_MCC" if self.task is TaskType.CLASSIFICATION else "CV_RMSE")
        self.export_direction_var = tk.StringVar(value="auto")
        self.export_reason_var = tk.StringVar()
        self.preprocessing_vars: Dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=name == "Raw") for name in PREPROCESSING_METHODS
        }
        self.selector_vars: Dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=name == "None") for name in SELECTOR_METHODS
        }
        defaults = {"PLS-DA", "Random Forest"} if self.task is TaskType.CLASSIFICATION else {"PLS Regression", "Random Forest Regression"}
        self.model_vars: Dict[str, tk.BooleanVar] = {name: tk.BooleanVar(value=name in defaults) for name in self.models}

    def _create_gui(self) -> None:
        self.paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, bg="#E0F7FA")
        self.paned.pack(fill="both", expand=True)
        top = ttk.Frame(self.paned)
        bottom = ttk.Frame(self.paned)
        self.paned.add(top)
        self.paned.add(bottom)
        self.notebook = ttk.Notebook(top, padding="10")
        self.notebook.pack(fill="both", expand=True)
        self._create_data_tab()
        self._create_preprocessing_tab()
        self._create_selector_tab()
        self._create_model_tab()
        self._create_results_tab()
        self._create_ai_tab()
        self._create_console(bottom)
        self._create_status_bar()
        self.root.update_idletasks()
        self.paned.sash_place(0, 0, self.root.winfo_height() // 2)

    def _scrollable_tab(self, title: str):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text=title)
        canvas = tk.Canvas(tab, highlightthickness=0, bg="#FFFFFF")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units")))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        return frame

    def _create_data_tab(self) -> None:
        frame = self._scrollable_tab("Data Loading")
        file_box = ttk.LabelFrame(frame, text="Data File Selection", padding=5)
        file_box.pack(fill="x", pady=(0, 5))
        ttk.Label(file_box, text="Raw data file:").pack(anchor="w")
        row = ttk.Frame(file_box)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.data_file).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(row, text="Browse", command=self._browse_data).pack(side="right")
        project = ttk.LabelFrame(frame, text="Project Directory", padding=5)
        project.pack(fill="x", pady=(0, 5))
        ttk.Label(project, text="Base directory:").pack(anchor="w")
        prow = ttk.Frame(project)
        prow.pack(fill="x")
        ttk.Entry(prow, textvariable=self.base_dir).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(prow, text="Browse", command=self._browse_output).pack(side="right")
        status = ttk.LabelFrame(frame, text="Project Status", padding=5)
        status.pack(fill="x", pady=(0, 5))
        self.progress_status_label = ttk.Label(status)
        self.progress_status_label.pack(anchor="w")
        self.next_step_label = ttk.Label(status)
        self.next_step_label.pack(anchor="w")
        self.last_saved_label = ttk.Label(status)
        self.last_saved_label.pack(anchor="w")
        srow = ttk.Frame(status)
        srow.pack(fill="x", pady=(5, 0))
        ttk.Button(srow, text="Save State", command=self._save_state).pack(side="left", padx=(0, 5))
        ttk.Button(srow, text="Load State", command=self._load_state).pack(side="left", padx=(0, 5))
        ttk.Button(srow, text="Reset Project", command=self._reset).pack(side="left", padx=(0, 5))
        ttk.Button(srow, text="Refresh Status", command=self._update_status).pack(side="left")
        split = ttk.LabelFrame(frame, text="Data Split and Evaluation Mode", padding=5)
        split.pack(fill="x", pady=(0, 5))
        ratio = ttk.Frame(split)
        ratio.pack(fill="x", pady=2)
        ttk.Label(ratio, text="Internal holdout ratio:").pack(side="left")
        ttk.Combobox(ratio, textvariable=self.validation_ratio, values=["0.1", "0.15", "0.2", "0.25", "0.3"], width=6, state="readonly").pack(side="left", padx=5)
        ttk.Radiobutton(split, text="Exploration (quick test)", variable=self.mode_var, value="exploration", command=self._mode_changed).pack(anchor="w")
        ttk.Radiobutton(split, text="Standard", variable=self.mode_var, value="standard", command=self._mode_changed).pack(anchor="w")
        ttk.Radiobutton(split, text="Full", variable=self.mode_var, value="audit", command=self._mode_changed).pack(anchor="w")
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=5)
        ttk.Button(buttons, text="Load and Split Data", command=self._load_data).pack(side="left", padx=5)
        self.stop_button = ttk.Button(buttons, text="Stop Processing", command=self._stop, state="disabled")
        self.stop_button.pack(side="right", padx=5)

    def _create_preprocessing_tab(self) -> None:
        frame = self._scrollable_tab("Preprocessing")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = ttk.Frame(frame)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        box = ttk.LabelFrame(left, text="High-Dimensional Data Preprocessing Options", padding=5)
        box.pack(fill="x")
        for name in PREPROCESSING_METHODS:
            ttk.Checkbutton(box, text=name, variable=self.preprocessing_vars[name]).pack(anchor="w")
        ttk.Button(left, text="Confirm Preprocessing", command=self._confirm_preprocessing).pack(anchor="w", pady=5)
        result = ttk.LabelFrame(right, text="Preprocessing Results", padding=5)
        result.pack(fill="both", expand=True)
        self.preprocessing_text = ScrolledText(result, height=15, bg="#FFFDE7", font=("Segoe UI", 10))
        self.preprocessing_text.pack(fill="both", expand=True)

    def _create_selector_tab(self) -> None:
        frame = self._scrollable_tab("Variable Selection")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = ttk.Frame(frame)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        box = ttk.LabelFrame(left, text="Variable Selection Methods", padding=5)
        box.pack(fill="x")
        labels = {"None": "Raw data (no variable selection)", "VIP": "VIP", "GA": "GA", "iPLS": "iPLS", "CARS": "CARS", "SFLA": "SFLA"}
        for name in SELECTOR_METHODS:
            ttk.Checkbutton(box, text=labels[name], variable=self.selector_vars[name]).pack(anchor="w")
        ttk.Button(left, text="Confirm Variable Selection", command=self._confirm_selectors).pack(anchor="w", pady=5)
        result = ttk.LabelFrame(right, text="Variable Selection Results", padding=5)
        result.pack(fill="both", expand=True)
        self.selector_text = ScrolledText(result, height=15, bg="#FFFDE7", font=("Segoe UI", 10))
        self.selector_text.pack(fill="both", expand=True)

    def _create_model_tab(self) -> None:
        frame = self._scrollable_tab("Model Training")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = ttk.Frame(frame)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        box = ttk.LabelFrame(left, text=f"{self.task_title} Models", padding=5)
        box.pack(fill="x")
        for name in self.models:
            ttk.Checkbutton(box, text=name, variable=self.model_vars[name]).pack(anchor="w")
        advanced = ttk.LabelFrame(left, text="Advanced Options", padding=5)
        advanced.pack(fill="x", pady=5)
        for label, variable, values in [
            ("Outer CV folds:", self.outer_folds, ["3", "5", "7", "10"]),
            ("Inner CV folds:", self.inner_folds, ["2", "3", "5"]),
        ]:
            row = ttk.Frame(advanced)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Combobox(row, textvariable=variable, values=values, width=5, state="readonly").pack(side="left", padx=5)
        seed = ttk.Frame(advanced)
        seed.pack(fill="x")
        ttk.Label(seed, text="Random seed:").pack(side="left")
        ttk.Entry(seed, textvariable=self.seed_var, width=8).pack(side="left", padx=5)
        ttk.Button(left, text="Train Models", command=self._train).pack(anchor="w", pady=5)
        result = ttk.LabelFrame(right, text="Model Training Results", padding=5)
        result.pack(fill="both", expand=True)
        self.training_text = ScrolledText(result, height=15, bg="#FFFDE7", font=("Segoe UI", 10))
        self.training_text.pack(fill="both", expand=True)

    def _create_results_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Results Analysis")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = ttk.Frame(frame)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        options_area = ttk.Frame(left)
        options_area.pack(fill="both", expand=True)
        options_canvas = tk.Canvas(options_area, height=220, highlightthickness=0, bg="#FFFFFF")
        options_scroll = ttk.Scrollbar(options_area, orient="vertical", command=options_canvas.yview)
        options_content = ttk.Frame(options_canvas)
        options_window = options_canvas.create_window((0, 0), window=options_content, anchor="nw")
        options_content.bind("<Configure>", lambda _event: options_canvas.configure(scrollregion=options_canvas.bbox("all")))
        options_canvas.bind("<Configure>", lambda event: options_canvas.itemconfigure(options_window, width=event.width))
        options_canvas.configure(yscrollcommand=options_scroll.set)
        options_canvas.pack(side="left", fill="both", expand=True)
        options_scroll.pack(side="right", fill="y")
        buttons = ttk.LabelFrame(options_content, text="Analysis Options", padding=5)
        buttons.pack(fill="x")
        actions = [
            ("Generate Performance Report", self._performance_report),
            ("Create Visualization Plots", self._create_plots),
            ("Compare All Models", self._compare_models),
            ("Export Results", self._export_results),
            ("Export Candidate", self._export_candidate),
            ("Generate Leaderboard (CSV/HTML)", self._open_leaderboard),
        ]
        for label, command in actions:
            ttk.Button(buttons, text=label, command=command).pack(fill="x", pady=2)
        ttk.Separator(buttons).pack(fill="x", pady=5)
        ttk.Button(buttons, text="Select All Options", command=self._select_all).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Load Pipeline / Predict New Data", command=self._predict_new).pack(fill="x", pady=2)
        choice = ttk.LabelFrame(options_content, text="Candidate Export Choice", padding=5)
        choice.pack(fill="x", pady=5)
        ttk.Radiobutton(choice, text="Pipeline ID", variable=self.export_mode_var, value="id").pack(anchor="w")
        self.export_id_combo = ttk.Combobox(choice, textvariable=self.export_id_var, state="readonly")
        self.export_id_combo.pack(fill="x", pady=2)
        ttk.Radiobutton(choice, text="Calibration CV metric", variable=self.export_mode_var, value="metric").pack(anchor="w")
        self.export_metric_combo = ttk.Combobox(choice, textvariable=self.export_metric_var, state="readonly")
        self.export_metric_combo.pack(fill="x", pady=2)
        ttk.Label(choice, text="Metric direction:").pack(anchor="w")
        ttk.Combobox(choice, textvariable=self.export_direction_var, values=["auto", "higher", "lower"], state="readonly").pack(fill="x")
        ttk.Label(choice, text="Selection reason (optional):").pack(anchor="w")
        ttk.Entry(choice, textvariable=self.export_reason_var).pack(fill="x")
        ttk.Label(choice, text="Default order is descriptive. Choosing after inspecting holdout results uses it for selection.", wraplength=350).pack(anchor="w", pady=4)
        summary = ttk.LabelFrame(left, text="Model Performance Summary", padding=5)
        summary.pack(fill="both", expand=True, pady=(5, 0))
        self.summary_text = ScrolledText(summary, height=10, bg="#FFFDE7", font=("Segoe UI", 9))
        self.summary_text.pack(fill="both", expand=True)
        detail = ttk.LabelFrame(right, text="Detailed Results", padding=5)
        detail.pack(fill="both", expand=True)
        self.detail_text = ScrolledText(detail, height=20, bg="#FFFDE7", font=("Consolas", 8))
        self.detail_text.pack(fill="both", expand=True)

    def _create_ai_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="AI Report Generation")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = ttk.Frame(frame)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        options = ttk.LabelFrame(left, text="AI Report Options", padding=5)
        options.pack(fill="x")
        ttk.Button(options, text="1. Analyze Model Performance", command=self._analyze).pack(fill="x", pady=2)
        api = ttk.LabelFrame(left, text="LLM API Configuration", padding=5)
        api.pack(fill="x", pady=(5, 0))
        ttk.Label(api, text="API Key (DeepSeek Only):").pack(anchor="w")
        ttk.Entry(api, textvariable=self.api_key_var, show="*").pack(fill="x", pady=2)
        ttk.Label(api, text="Leave blank to use DEEPSEEK_API_KEY.").pack(anchor="w")
        ttk.Label(api, text="Model Selection:").pack(anchor="w")
        ttk.Combobox(api, textvariable=self.deepseek_model_var, values=DEEPSEEK_MODELS, state="readonly").pack(fill="x", pady=2)
        ttk.Button(api, text="2. Generate AI Reports (Word/PDF)", command=self._generate_ai_report).pack(fill="x", pady=(8, 2))
        result = ttk.LabelFrame(right, text="AI Report Results", padding=5)
        result.pack(fill="both", expand=True)
        self.ai_text = ScrolledText(result, height=20, bg="#FFFDE7", font=("Segoe UI", 10))
        self.ai_text.pack(fill="both", expand=True)

    def _create_console(self, parent) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)
        console = ttk.LabelFrame(frame, text="Console Output")
        console.pack(side="left", fill="both", expand=True, padx=(0, 5))
        image_box = ttk.LabelFrame(frame, text="Visualization")
        image_box.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.console_text = ScrolledText(console, height=15, width=60, bg="#FFFDE7", font=("Segoe UI", 10))
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Button(console, text="Clear Console", command=lambda: self.console_text.delete("1.0", "end")).pack(side="bottom", pady=3)
        self.canvas = tk.Canvas(image_box, bg="white")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self._load_image()

    def _create_status_bar(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(frame, text="Ready")
        self.status_label.pack(side="left", padx=5)
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(frame, variable=self.progress_var, maximum=100).pack(side="right", fill="x", expand=True, padx=5)

    def _load_image(self) -> None:
        try:
            from PIL import Image, ImageTk

            path = get_resource_path("123.png")
            if not Path(path).exists():
                self.canvas.create_text(200, 120, text="Image '123.png' not found", fill="gray")
                return
            self.current_image = Image.open(path)

            def resize(_event=None):
                width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
                if width <= 1 or height <= 1:
                    return
                scale = min(width / self.current_image.width, height / self.current_image.height) * 0.9
                image = self.current_image.resize((max(1, int(self.current_image.width * scale)), max(1, int(self.current_image.height * scale))), Image.Resampling.LANCZOS)
                self.current_photo = ImageTk.PhotoImage(image)
                self.canvas.delete("all")
                self.canvas.create_image(width // 2, height // 2, image=self.current_photo)

            self.canvas.bind("<Configure>", resize)
            self.root.after(200, resize)
        except Exception as exc:
            print(f"Image loading warning: {exc}")

    def _mode_changed(self) -> None:
        if self.mode_var.get() == "exploration":
            self.outer_folds.set("3")
        else:
            self.outer_folds.set("5")

    def _browse_data(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        if path:
            self.data_file.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.base_dir.set(path)

    def _load_data(self) -> None:
        if self._run_busy or self._ai_busy:
            messagebox.showwarning("Run in progress", "Wait for processing to finish before loading another dataset.")
            return
        try:
            config = self._config()
            dataset = load_dataset(self.data_file.get(), self.task)
            self.dataset_split = split_dataset(dataset, config)
            self.project_step = "Preprocessing"
            self._update_status()
            print(f"Loaded {len(dataset.target)} samples and {dataset.features.shape[1]} features")
            print(f"Calibration={len(self.dataset_split.y_cal)}, internal holdout={len(self.dataset_split.y_validation)}")
            messagebox.showinfo("Success", "Data loaded and split successfully.")
        except Exception as exc:
            messagebox.showerror("Data loading failed", str(exc))
            print(traceback.format_exc())

    def _selected(self, mapping: Dict[str, tk.BooleanVar]) -> List[str]:
        return [name for name, variable in mapping.items() if variable.get()]

    def _confirm_preprocessing(self) -> None:
        selected = self._selected(self.preprocessing_vars)
        if not selected:
            messagebox.showwarning("Selection required", "Select at least one preprocessing method.")
            return
        self.preprocessing_text.delete("1.0", "end")
        self.preprocessing_text.insert("end", "Preprocessing fitted within calibration training boundaries:\n\n" + "\n".join(selected))
        self.project_step = "Variable selection"
        self._update_status()

    def _confirm_selectors(self) -> None:
        selected = self._selected(self.selector_vars)
        if not selected:
            messagebox.showwarning("Selection required", "Select at least one variable-selection method.")
            return
        self.selector_text.delete("1.0", "end")
        self.selector_text.insert("end", "Selectors fitted on calibration training subsets:\n\n" + "\n".join(selected))
        self.project_step = "Model training"
        self._update_status()

    def _config(self) -> RunConfig:
        return RunConfig(
            task=self.task,
            mode=EvaluationMode(self.mode_var.get()),
            validation_ratio=float(self.validation_ratio.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            random_state=int(self.seed_var.get()),
            n_jobs=0,
            output_dir=Path(self.base_dir.get()),
            strict=False,
            resume=True,
        )

    def _train(self) -> None:
        if self._run_busy or self._export_busy or self._ai_busy:
            messagebox.showwarning("Processing in progress", "Wait for the current run or export to finish.")
            return
        if self.dataset_split is None:
            messagebox.showwarning("Missing data", "Load and split data first.")
            return
        preprocessing = self._selected(self.preprocessing_vars)
        selectors = self._selected(self.selector_vars)
        models = self._selected(self.model_vars)
        if not preprocessing or not selectors or not models:
            messagebox.showwarning("Selection required", "Select preprocessing, variable selection, and model options.")
            return
        try:
            config = self._config()
            data_path = Path(self.data_file.get()).resolve()
            dataset = load_dataset(data_path, self.task)
            run_split = split_dataset(dataset, config)
        except Exception as exc:
            messagebox.showerror("Run setup failed", str(exc))
            return
        specs = enumerate_specs(self.task, preprocessing, selectors, models, seed=config.random_state)
        self.dataset_split = run_split
        self.outputs = None
        self.last_specs = specs
        self.last_run_data_file = data_path
        self._run_busy = True
        self.stop_requested = False
        self.stop_button.configure(state="normal")
        self.status_label.configure(text=f"Running {len(specs)} pipelines")

        def work():
            try:
                engine = EvaluationEngine(
                    run_split,
                    config,
                    logger=lambda message: self.worker_queue.put(("log", message)),
                    should_stop=lambda: self.stop_requested,
                )
                outputs = engine.run(
                    specs,
                    progress=lambda done, total, spec: self.worker_queue.put(("progress", done, total, spec)),
                )
                self.worker_queue.put(("done", outputs))
            except Exception as exc:
                self.worker_queue.put(("error", exc, traceback.format_exc(), "run"))

        threading.Thread(target=work, daemon=True).start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                event = self.worker_queue.get_nowait()
                if event[0] == "log":
                    print(event[1])
                elif event[0] == "progress":
                    _, done, total, spec = event
                    self.progress_var.set(100 * done / max(total, 1))
                    self.status_label.configure(text=f"{done}/{total}: {spec.model}")
                elif event[0] == "done":
                    self.outputs = event[1]
                    self._run_busy = False
                    self.stop_button.configure(state="disabled")
                    self.project_step = "Results analysis"
                    self._update_status()
                    self._refresh_results()
                    messagebox.showinfo("Completed", "Pipeline evaluation completed.")
                elif event[0] == "error":
                    kind = event[3] if len(event) > 3 else "run"
                    if kind == "run":
                        self._run_busy = False
                        self.stop_button.configure(state="disabled")
                    elif kind == "export":
                        self._export_busy = False
                    elif kind == "ai":
                        self._ai_busy = False
                    print(event[2])
                    title = {"run": "Pipeline run failed", "export": "Candidate export failed", "ai": "AI report failed"}.get(kind, "Processing failed")
                    messagebox.showerror(title, str(event[1]))
                elif event[0] == "export_done":
                    self._export_busy = False
                    exported = event[1]
                    text = f"Pipeline ID: {exported.pipeline_id}\nModel: {exported.bundle_path}\nManifest: {exported.manifest_path}\nInternal holdout: {exported.validation_status}"
                    if exported.validation_error:
                        text += "\n" + exported.validation_error
                    print(text)
                    messagebox.showinfo("Candidate exported", text)
                elif event[0] == "ai_done":
                    self._ai_busy = False
                    _, report, paths = event
                    self.ai_text.delete("1.0", "end")
                    self.ai_text.insert("end", report)
                    print("AI reports:\n" + "\n".join(map(str, paths)))
                    self.project_step = "AI interpretation"
                    self._update_status()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker_queue)

    def _stop(self) -> None:
        self.stop_requested = True
        print("Stop requested; the current atomic pipeline step will finish before shutdown.")

    def _refresh_results(self) -> None:
        if not self.outputs:
            return
        frame = self.outputs.results
        columns = [
            "Selection_Rank",
            "Pipeline_ID",
            "Preprocessing",
            "Feature_Selection",
            "Model",
            "Selection_Score",
            "Cal_MCC" if self.task is TaskType.CLASSIFICATION else "Cal_R2",
            "CV_MCC" if self.task is TaskType.CLASSIFICATION else "CV_R2",
            "Val_MCC" if self.task is TaskType.CLASSIFICATION else "Val_R2",
            "Val_Evaluated",
            "Val_Status",
            "Status",
        ]
        available = [column for column in columns if column in frame.columns]
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "Default calibration-CV display order (first 15 rows):\n\n" + frame.head(15)[available].to_string(index=False))
        eligible = frame[frame["Status"].eq("SUCCESS") & frame["Eligible_For_Ranking"].eq(True) & frame["Selection_Rank"].notna()]
        ids = eligible["Pipeline_ID"].astype(str).tolist()
        self.export_id_combo.configure(values=ids)
        if self.export_id_var.get() not in ids:
            self.export_id_var.set(ids[0] if ids else "")
        self.export_metric_combo.configure(values=["Selection_Score"] + [name for name in frame if name.startswith("CV_")])
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", frame.to_string(index=False))
        self.training_text.delete("1.0", "end")
        self.training_text.insert("end", f"Completed: {len(frame)}\nSuccess: {frame['Status'].eq('SUCCESS').sum()}\nFailed: {(~frame['Status'].eq('SUCCESS')).sum()}")

    def _require_outputs(self) -> bool:
        if self.outputs is None:
            messagebox.showwarning("Missing results", "Train models first.")
            return False
        return True

    def _performance_report(self) -> None:
        if not self._require_outputs():
            return
        destination = Path(self.base_dir.get()) / f"performance_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.outputs.results.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Performance report: {destination}")

    def _create_plots(self) -> None:
        if not self._require_outputs():
            return
        paths = create_visualization_suite(self.outputs.results, Path(self.base_dir.get()) / "plots", self.task)
        print("Visualization plots:\n" + "\n".join(map(str, paths)))

    def _compare_models(self) -> None:
        if not self._require_outputs():
            return
        path = compare_models_plot(self.outputs.results, Path(self.base_dir.get()) / "plots" / "model_comparison.png")
        print(f"Model comparison plot: {path}")

    def _export_results(self) -> None:
        if not self._require_outputs():
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.outputs.results.to_csv(path, index=False, encoding="utf-8-sig")

    def _candidate_export_context(self):
        """Reconstruct the evaluation from its manifest, never edited GUI controls."""
        if self.outputs is None or self.last_run_data_file is None:
            raise ValueError("A completed run and its original input file are required.")
        manifest = json.loads(self.outputs.manifest.read_text(encoding="utf-8"))
        config_data = dict(manifest["config"])
        if TaskType(config_data["task"]) is not self.task:
            raise ValueError("Run task does not match this application.")
        config_data.update(output_dir=self.outputs.manifest.parent, n_jobs=1, resume=False, strict=False)
        config = RunConfig(**config_data)
        specs = []
        for item in manifest["pipeline_specs"]:
            spec = PipelineSpec(
                TaskType(item["task"]), item["preprocessing"], item["selector"], item["model"],
                hyperparameters=item.get("hyperparameters", {}), seed=int(item.get("seed", 42)),
            )
            if item.get("pipeline_id") != spec.pipeline_id:
                raise ValueError("Stored pipeline definition does not match its ID.")
            specs.append(spec)
        dataset = load_dataset(self.last_run_data_file, self.task)
        if dataset.fingerprint != manifest["dataset_fingerprint"]:
            raise ValueError("The original dataset has changed since evaluation. Run it again before export.")
        split = split_dataset(dataset, config)
        for key, values in (("calibration_samples", split.sample_cal), ("validation_samples", split.sample_validation)):
            if key in manifest and list(map(str, manifest[key])) != values.astype(str).tolist():
                raise ValueError("Recreated calibration/internal holdout partition differs from the stored run.")
        engine = EvaluationEngine(split, config, logger=lambda message: self.worker_queue.put(("log", message)))
        return engine, specs, manifest

    def _export_candidate(self) -> None:
        if not self._require_outputs():
            return
        if self._run_busy or self._export_busy:
            messagebox.showwarning("Processing in progress", "Wait for the current run or export to finish.")
            return
        mode = self.export_mode_var.get()
        pipeline_id = self.export_id_var.get().strip() if mode == "id" else None
        metric = self.export_metric_var.get().strip() if mode == "metric" else None
        if mode not in {"id", "metric"} or not (pipeline_id or metric):
            messagebox.showwarning("Selection required", "Choose a pipeline ID or calibration CV metric.")
            return
        direction = self.export_direction_var.get()
        higher = {"auto": None, "higher": True, "lower": False}.get(direction)
        reason = self.export_reason_var.get().strip() or None
        ranked = self.outputs.results.copy(deep=True)
        self._export_busy = True

        def work():
            try:
                engine, specs, manifest = self._candidate_export_context()
                exported = engine.export_candidate(
                    ranked, specs, pipeline_id=pipeline_id, metric=metric,
                    higher_is_better=higher if metric else None, selection_reason=reason,
                    source_manifest=manifest,
                )
                self.worker_queue.put(("export_done", exported))
            except Exception as exc:
                self.worker_queue.put(("error", exc, traceback.format_exc(), "export"))
        threading.Thread(target=work, daemon=True).start()

    def _open_leaderboard(self) -> None:
        if self._require_outputs():
            webbrowser.open(self.outputs.leaderboard_html.resolve().as_uri())

    def _select_all(self) -> None:
        for mapping in (self.preprocessing_vars, self.selector_vars, self.model_vars):
            for variable in mapping.values():
                variable.set(True)
        print("All 15 preprocessing, 6 selector, and 11 model options selected (990 pipelines).")

    def _predict_new(self) -> None:
        model_path = filedialog.askopenfilename(filetypes=[("GB pipeline", "*.joblib")])
        if not model_path:
            return
        data_path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xlsx *.xls")])
        if not data_path:
            return
        try:
            bundle = load_bundle(model_path)
            table = pd.read_csv(data_path) if Path(data_path).suffix.lower() == ".csv" else pd.read_excel(data_path)
            sample = table["Sample name"].astype(str) if "Sample name" in table else pd.Series([f"Sample_{i + 1}" for i in range(len(table))])
            prediction = bundle.predict_frame(table.loc[:, bundle.feature_names])
            destination = Path(data_path).with_name(Path(data_path).stem + "_gb_predictions.csv")
            pd.DataFrame({"Sample name": sample, "Prediction": prediction}).to_csv(destination, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Prediction completed", str(destination))
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))

    def _report_summary(self):
        manifest = json.loads(self.outputs.manifest.read_text(encoding="utf-8"))
        context = {name: manifest.get(name) for name in (
            "run_id", "evaluation_revision", "dataset_fingerprint", "source_provenance",
            "calibration_observations", "internal_holdout_observations",
            "calibration_groups", "internal_holdout_groups",
        )}
        config = manifest.get("config", {})
        context["evaluation_config"] = {name: config.get(name) for name in (
            "task", "mode", "outer_folds", "inner_folds", "random_state",
        )}
        return structured_summary(self.outputs.results, self.task, context=context)

    def _analyze(self) -> None:
        if not self._require_outputs():
            return
        try:
            prompt = build_deepseek_prompt(self._report_summary())
        except Exception as exc:
            messagebox.showerror("Report input failed", str(exc))
            return
        self.ai_text.delete("1.0", "end")
        self.ai_text.insert("end", prompt)
        path = self.outputs.manifest.parent / "ai_reports" / f"deepseek_prompt_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt, encoding="utf-8")

    def _generate_ai_report(self) -> None:
        if self._ai_busy:
            messagebox.showwarning("Report in progress", "Wait for the current AI report to finish.")
            return
        if not self._require_outputs():
            return
        try:
            summary = self._report_summary()
        except Exception as exc:
            messagebox.showerror("Report input failed", str(exc))
            return
        # Capture the effective key before starting the worker, including the
        # environment fallback, so queued exceptions can redact the same key.
        api_key = self.api_key_var.get().strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
        model_name = self.deepseek_model_var.get()
        if model_name not in DEEPSEEK_MODELS:
            messagebox.showerror("Unsupported model", "Select deepseek-flash or deepseek-v4-pro.")
            return
        report_dir = self.outputs.manifest.parent / "ai_reports"
        self._ai_busy = True

        def work():
            try:
                report = call_deepseek(summary, api_key, model_name, record_dir=report_dir / "call_records")
                paths = write_report_files(report, report_dir, f"gb_{self.task.value}_report")
                self.worker_queue.put(("ai_done", report, paths))
            except Exception as exc:
                error_text, trace = str(exc), traceback.format_exc()
                if api_key:
                    error_text, trace = error_text.replace(api_key, "[REDACTED]"), trace.replace(api_key, "[REDACTED]")
                self.worker_queue.put(("error", RuntimeError(error_text), trace, "ai"))

        threading.Thread(target=work, daemon=True).start()

    def _save_state(self) -> None:
        path = Path(self.base_dir.get()) / "project_state_v2.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": self.task.value,
            "data_file": self.data_file.get(),
            "base_dir": self.base_dir.get(),
            "validation_ratio": self.validation_ratio.get(),
            "mode": self.mode_var.get(),
            "outer_folds": self.outer_folds.get(),
            "inner_folds": self.inner_folds.get(),
            "seed": self.seed_var.get(),
            "preprocessing": {k: v.get() for k, v in self.preprocessing_vars.items()},
            "selectors": {k: v.get() for k, v in self.selector_vars.items()},
            "models": {k: v.get() for k, v in self.model_vars.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.last_saved = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._update_status()

    def _load_state(self) -> None:
        path = Path(self.base_dir.get()) / "project_state_v2.json"
        if not path.exists():
            messagebox.showinfo("State", "No saved v2 project state was found.")
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        for variable, key in [(self.data_file, "data_file"), (self.validation_ratio, "validation_ratio"), (self.mode_var, "mode"), (self.outer_folds, "outer_folds"), (self.inner_folds, "inner_folds"), (self.seed_var, "seed")]:
            legacy_key = "holdout_ratio" if key == "validation_ratio" else key
            variable.set(state.get(key, state.get(legacy_key, variable.get())))
        for key, mapping in [("preprocessing", self.preprocessing_vars), ("selectors", self.selector_vars), ("models", self.model_vars)]:
            for name, value in state.get(key, {}).items():
                if name in mapping:
                    mapping[name].set(bool(value))
        self.last_saved = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        self._update_status()

    def _reset(self) -> None:
        if self._run_busy or self._export_busy or self._ai_busy:
            messagebox.showwarning("Processing in progress", "Wait for processing to finish before resetting.")
            return
        if messagebox.askyesno("Reset", "Clear loaded data and in-memory results?"):
            self.dataset_split = None
            self.outputs = None
            self.last_specs = []
            self.last_run_data_file = None
            self.project_step = "Data loading"
            self.progress_var.set(0)
            for widget in (self.preprocessing_text, self.selector_text, self.training_text, self.summary_text, self.detail_text, self.ai_text):
                widget.delete("1.0", "end")
            self._update_status()

    def _update_status(self) -> None:
        next_steps = {"Data loading": "Preprocessing", "Preprocessing": "Variable selection", "Variable selection": "Model training", "Model training": "Results analysis", "Results analysis": "AI report generation"}
        self.progress_status_label.configure(text=f"Current progress: {self.project_step}")
        self.next_step_label.configure(text=f"Next step: {next_steps.get(self.project_step, 'Data loading')}")
        self.last_saved_label.configure(text=f"Last saved: {self.last_saved}")

    def close(self) -> None:
        self.stop_requested = True
        sys.stdout = self.original_stdout
        self.root.destroy()


def _show_splash(root: tk.Tk, task: TaskType) -> None:
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    width, height = 500, 400
    x = (splash.winfo_screenwidth() - width) // 2
    y = (splash.winfo_screenheight() - height) // 2
    splash.geometry(f"{width}x{height}+{x}+{y}")
    frame = tk.Frame(splash, bg="white", relief="ridge", bd=2)
    frame.pack(fill="both", expand=True)
    try:
        from PIL import Image, ImageTk

        image = Image.open(get_resource_path("123.png")).resize((300, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(frame, image=photo, bg="white")
        label.image = photo
        label.pack(pady=20)
    except Exception:
        tk.Label(frame, text=f"Grand Banquet {TaskType(task).value.title()}", font=("Arial", 24, "bold"), bg="white", fg="#2E86AB").pack(pady=50)
    tk.Label(frame, text="Grand Banquet  v0.0", font=("Arial", 12), bg="white", fg="#666666").pack(pady=5)
    progress = ttk.Progressbar(frame, mode="indeterminate", length=300)
    progress.pack(pady=10)
    progress.start(10)
    splash.after(1600, splash.destroy)
    root.wait_window(splash)


def launch_gui(task: TaskType) -> None:
    root = tk.Tk()
    root.withdraw()
    _show_splash(root, task)
    root.deiconify()
    app = GBApplication(root, task)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
