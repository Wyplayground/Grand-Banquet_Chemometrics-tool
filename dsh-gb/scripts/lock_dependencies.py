"""Explicit maintainer operation: fetch PyPI wheel hashes for reviewed exact versions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]


def main():
    pins = json.loads((ROOT / 'requirements.versions.json').read_text())
    def fetch(item):
        name, version = item
        request = urllib.request.Request(f'https://pypi.org/pypi/{name}/{version}/json', headers={'User-Agent': 'dsh-gb-lock-builder/1'})
        with urllib.request.urlopen(request, timeout=45) as reply:
            data = json.load(reply)
        requires = data['info'].get('requires_python') or ''
        if any(p not in SpecifierSet(requires) for p in pins['python']):
            raise ValueError(f'{name}=={version} does not support the declared Python versions: {requires}')
        hashes = sorted({f['digests']['sha256'] for f in data['urls'] if f['filename'].endswith('.whl') and not f.get('yanked')})
        if not hashes:
            raise ValueError(f'No non-yanked wheel for {name}=={version}')
        return name, version, hashes
    with ThreadPoolExecutor(max_workers=4) as pool:
        packages = sorted(pool.map(fetch, pins['packages'].items()))
    lines = ['# Exact direct/transitive pins and PyPI wheel hashes. Regenerate explicitly with scripts/lock_dependencies.py.',
             '# Install with --require-hashes --only-binary=:all:. Supported Python: ' + ', '.join(pins['python']), '']
    for name, version, hashes in packages:
        marker = pins['markers'].get(name)
        lines.append(f'{name}=={version}' + (f'; {marker}' if marker else '') + ' \\')
        lines.extend('    --hash=sha256:' + digest + (' \\' if i < len(hashes)-1 else '') for i, digest in enumerate(hashes))
    text = '\n'.join(lines) + '\n'
    (ROOT / 'requirements.lock').write_text(text, encoding='utf-8')
    manifest = dict(schema=1, python=pins['python'], packages=len(packages),
                    requirements_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    pins_sha256=hashlib.sha256((ROOT / 'requirements.versions.json').read_bytes()).hexdigest())
    (ROOT / 'python' / 'dependencies.lock.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
