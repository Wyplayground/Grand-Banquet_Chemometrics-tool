"""Explicit, restartable private-environment setup with exact dependency hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'python'))
from path_policy import contained, inside
from process_lock import run_lock


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--install-deps', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--json-progress', action='store_true')
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        parser.error('Use Python 3.10 through 3.13.')
    if not args.check and not args.install_deps:
        parser.error('Choose --install-deps or the read-only --check.')
    workspace = args.workspace.expanduser().resolve()
    runtime = workspace / '.dsh-gb' / 'runtime'
    gb, environment = runtime / 'gb', runtime / 'venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    source = json.loads((PACKAGE / 'gb.lock.json').read_text())
    dependency_lock = json.loads((PACKAGE / 'python' / 'dependencies.lock.json').read_text())
    requirements = PACKAGE / 'requirements.lock'
    if hashlib.sha256(requirements.read_text(encoding='utf-8').encode()).hexdigest() != dependency_lock['requirements_sha256']:
        raise ValueError('Dependency lock differs from its manifest.')

    def progress(stage, status='running', **extra):
        value = dict(status=status, stage=stage, updated_at=time.time(), **extra)
        if args.json_progress:
            print('GB_SETUP:' + json.dumps(value, ensure_ascii=False), flush=True)
        else:
            print(stage, flush=True)
        if not args.check:
            destination = contained(runtime, runtime / 'setup-state.json')
            temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
            temporary.replace(destination)

    def verify():
        if not python.is_file() or not (gb / 'gb_core').is_dir():
            raise ValueError('Environment is incomplete; run installation first.')
        progress('Verifying exact Python dependencies')
        run([python, PACKAGE / 'python' / 'dependency_policy.py'])
        run([python, '-m', 'pip', 'check'])
        prefix = subprocess.check_output([python, '-c', 'import sys;print(sys.prefix)'], text=True).strip()
        if Path(prefix).resolve() != environment.resolve():
            raise ValueError('Interpreter does not activate the private environment.')
        run([python, '-c', 'import sys;sys.path.insert(0,sys.argv[1]);from gb_core.engine import EvaluationEngine', gb])
        return dict(pythonPath=str(python), gbRoot=str(gb), workspaceRoot=str(workspace),
                    dependency_lock_sha256=dependency_lock['requirements_sha256'], gb_commit=source['commit'])

    if args.check:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    workspace.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    runtime = contained(workspace, runtime)
    for child in (gb, environment):
        if child.is_symlink() or not inside(runtime, child.resolve()):
            raise ValueError('Managed engine/environment cannot redirect outside the private runtime.')
    with run_lock(runtime / 'install.lock'):
        try:
            progress('Fetching pinned GB engine')
            if not gb.exists():
                staging = contained(runtime, runtime / ('gb-staging-' + uuid.uuid4().hex[:12]))
                run(['git', 'clone', '--no-checkout', source['repository'], staging])
                run(['git', '-C', staging, 'checkout', '--detach', source['commit']])
                staging.rename(gb)
            else:
                origin = subprocess.check_output(['git', '-C', str(gb), 'remote', 'get-url', 'origin'], text=True).strip()
                if origin != source['repository']:
                    raise ValueError('GB directory has a different origin; refusing to replace it.')
                if subprocess.check_output(['git', '-C', str(gb), 'status', '--porcelain'], text=True).strip():
                    raise ValueError('GB checkout has local changes; refusing to replace them.')
                run(['git', '-C', gb, 'checkout', '--detach', source['commit']])
            progress('Creating private Python environment')
            if not python.is_file() or not (environment / 'pyvenv.cfg').is_file():
                run([sys.executable, '-m', 'venv', environment])
            progress('Installing exact versions with wheel hash verification')
            run([python, '-m', 'pip', 'install', '--require-hashes', '--only-binary=:all:', '-r', requirements])
            config = verify()
            record = contained(runtime, runtime / 'setup.json')
            temporary = record.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(record)
            progress('Fixed-version environment ready', status='completed')
            print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
        except Exception as error:
            progress('Installation failed; retry is supported', status='failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
