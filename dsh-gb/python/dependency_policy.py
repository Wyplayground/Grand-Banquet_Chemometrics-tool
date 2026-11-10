"""Runtime verification for exact, platform-aware Python dependency pins."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def lock_status():
    from packaging.markers import Marker
    manifest = json.loads((ROOT / 'python' / 'dependencies.lock.json').read_text())
    # Canonical LF is independent of Git's Windows checkout newline policy.
    digest = hashlib.sha256((ROOT / 'requirements.lock').read_text(encoding='utf-8').encode()).hexdigest()
    pins_digest = hashlib.sha256((ROOT / 'requirements.versions.json').read_text(encoding='utf-8').encode()).hexdigest()
    if digest != manifest['requirements_sha256'] or pins_digest != manifest['pins_sha256']:
        raise ValueError('Dependency lock changed; regenerate and review the lock manifest.')
    pins = json.loads((ROOT / 'requirements.versions.json').read_text())
    actual, mismatches = {}, {}
    for name, expected in pins['packages'].items():
        marker = pins['markers'].get(name)
        if marker and not Marker(marker).evaluate():
            continue
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        if actual[name] != expected:
            mismatches[name] = dict(expected=expected, installed=actual[name])
    supported = f'{sys.version_info.major}.{sys.version_info.minor}' in pins['python']
    return dict(matches=not mismatches and supported, supported_python=supported, lock_sha256=digest,
                packages=actual, mismatches=mismatches)


def require_lock():
    status = lock_status()
    if not status['matches']:
        raise ValueError('Python dependencies differ from the pinned environment. Create/select the managed GB environment. Mismatches: ' + ', '.join(status['mismatches']))
    return status


if __name__ == '__main__':
    status = lock_status()
    print(json.dumps(status, ensure_ascii=False))
    raise SystemExit(0 if status['matches'] else 2)
