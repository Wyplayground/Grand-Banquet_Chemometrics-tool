#!/usr/bin/env python3
"""Grand Banquet vNext regression entry point."""

from __future__ import annotations

import sys
import traceback
import multiprocessing
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gb_core.cli import main_for_task
from gb_core.contracts import TaskType


def main() -> int:
    return main_for_task(TaskType.REGRESSION)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        Path("error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Grand Banquet Regression", f"{type(exc).__name__}: {exc}")
            root.destroy()
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)
