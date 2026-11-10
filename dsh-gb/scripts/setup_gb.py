"""Fetch the pinned public GB source; never install packages implicitly."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / 'gb.lock.json').read_text())
target = root / '.runtime' / 'gb'
if not target.exists():
    subprocess.run(['git', 'clone', '--no-checkout', lock['repository'], str(target)], check=True)
current = subprocess.run(['git', '-C', str(target), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
if current and (target / 'gb_core').exists():
    raise SystemExit('GB checkout has local changes; refusing to replace them.')
subprocess.run(['git', '-C', str(target), 'checkout', '--detach', lock['commit']], check=True)
print('GB ready:', target)
print('Install dependencies explicitly with your chosen Python: -m pip install -r', target / 'requirements.txt')
