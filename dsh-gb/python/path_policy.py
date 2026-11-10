"""Canonical path boundaries, including Windows extended-length drive/UNC aliases."""
import os
from pathlib import Path, PureWindowsPath


def windows_path(value):
    text = str(value).replace('/', '\\')
    if text.startswith('\\\\?\\UNC\\'):
        text = '\\\\' + text[8:]
    elif text.startswith('\\\\?\\'):
        text = text[4:]
    return PureWindowsPath(text)


def inside(root, candidate, windows=None):
    windows = os.name == 'nt' if windows is None else windows
    if windows:
        # Compare path components, never a string prefix (workspace2 != workspace).
        return windows_path(candidate).is_relative_to(windows_path(root))
    return Path(candidate).is_relative_to(Path(root))


def contained(root, value):
    root = Path(root).resolve(strict=True)
    path = Path(value).resolve()
    if not inside(root, path):
        raise ValueError(f'Path must stay inside the configured workspace. Resolved path: {path}; boundary: {root}')
    return path
