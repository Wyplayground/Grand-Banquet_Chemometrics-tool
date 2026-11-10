"""Local JSON/NDJSON bridge. GB alone owns scientific computation."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import tempfile
from path_policy import contained, inside
PROTOCOL_OUT = sys.stdout


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), file=PROTOCOL_OUT, flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def write_report(path, text):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


@contextlib.contextmanager
def run_lock(path):
    """OS lock is released even if the worker is forcibly terminated."""
    with path.open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0'); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def dispatch(req):
    root = Path(req['workspaceRoot']).resolve(strict=True)
    gb = Path(req['gbRoot']).resolve(strict=True)
    sys.path.insert(0, str(gb))
    from dependency_policy import lock_status, require_lock
    environment_lock = lock_status()
    if req['op'] in ('prepare', 'plan', 'run', 'export', 'predict'):
        require_lock()
    from gb_core.contracts import (TaskType, RunConfig, EvaluationMode, enumerate_specs,
        PREPROCESSING_METHODS, SELECTOR_METHODS, CLASSIFICATION_MODELS, REGRESSION_MODELS)
    from gb_core.data import load_dataset, split_dataset
    from gb_core.artifacts import source_provenance, dependency_versions
    from input_ops import discover, columns, prepare
    import pandas as pd

    scope = req.get('scope', 'local')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', scope):
        raise ValueError('Invalid session scope.')
    storage = contained(root, root / '.dsh-gb' / scope)
    storage.mkdir(parents=True, exist_ok=True)
    op = req['op']
    args = req.get('args', {})
    if op == 'artifact':
        import base64
        if args.get('model_id'):
            from model_ops import model_record
            record = model_record(storage, args['model_id'], contained)
            if args.get('kind') not in ('bundle', 'manifest'):
                raise ValueError('Choose bundle or manifest.')
            artifact = contained(storage, record[args['kind'] + '_path'])
        elif args.get('prediction_id'):
            if not re.fullmatch(r'[0-9a-f]{24}', args['prediction_id']):
                raise ValueError('Invalid prediction identifier.')
            artifact = contained(storage, storage / 'predictions' / args['prediction_id'] / 'predictions.csv')
        else:
            if not re.fullmatch(r'[0-9a-f]{24}', args.get('plan_id', '')):
                raise ValueError('Invalid plan identifier.')
            folder = contained(storage, storage / 'runs' / args['plan_id'])
            artifact = contained(folder, folder / args['name'])
            if artifact.suffix.lower() not in ('.csv', '.md', '.png', '.json') or artifact.name in ('state.json',):
                raise ValueError('Only result tables, reports, figures and manifests may be read.')
        if artifact.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('Browser downloads support up to 8 MiB; open the recorded local artifact path.')
        return dict(name=artifact.name, base64=base64.b64encode(artifact.read_bytes()).decode('ascii'))
    if op == 'status':
        with tempfile.TemporaryFile(dir=storage) as probe:
            probe.write(b'ready')
        return dict(ready=environment_lock['matches'], python=sys.executable, python_version=sys.version.split()[0],
                    gb_root=str(gb), workspace_root=str(root), dependencies=dependency_versions(),
                    source=source_provenance(gb), output_writable=True, dependency_lock=environment_lock)
    if op == 'files':
        return discover(root, args.get('directory', '.'), contained)
    if op == 'columns':
        return columns(contained(root, root / args['path']), args.get('sheet'))
    if op == 'prepare':
        return prepare(root, storage, args, contained, write_json)
    if op == 'predict':
        from model_ops import predict
        return predict(root, storage, gb, args, contained, write_json)
    if op == 'models':
        from model_ops import model_record
        records = []
        for file in sorted((storage / 'models').glob('*.json'))[:100]:
            records.append(model_record(storage, file.stem, contained))
        return dict(models=records)
    if op == 'history':
        records = []
        for file in sorted((storage / 'plans').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            saved = json.loads(contained(storage, file).read_text(encoding='utf-8'))
            state = storage / 'runs' / file.stem / 'state.json'
            records.append(dict(plan_id=file.stem, task=saved['settings']['task'], path=saved['data']['path'],
                                candidates=saved['candidate_count'], state=json.loads(contained(storage, state).read_text(encoding='utf-8')) if state.exists() else dict(status='not_started')))
        return dict(plans=records)
    if op in ('inspect', 'plan'):
        task = TaskType(args['task'])
        path = contained(root, root / args['path'])
        dataset = load_dataset(path, task)
        info = {'task': task.value, 'path': str(path), 'fingerprint': dataset.fingerprint,
                'samples': len(dataset.target), 'features': len(dataset.features.columns),
                'missing_feature_values': int(dataset.features.isna().sum().sum()),
                'groups': None if dataset.groups is None else int(dataset.groups.nunique()),
                'explicit_split': dataset.split_labels is not None,
                'missing_targets_policy': 'GB excludes rows with missing/non-numeric targets; samples is the retained count',
                'available_methods': {'preprocessing': list(PREPROCESSING_METHODS),
                    'selectors': list(SELECTOR_METHODS),
                    'models': list(CLASSIFICATION_MODELS if task is TaskType.CLASSIFICATION else REGRESSION_MODELS)}}
        # Readiness is checked using ordinary file I/O, never ACL changes or a shell.
        with tempfile.TemporaryFile(dir=storage) as probe:
            probe.write(b'gb-readiness-check')
            probe.flush()
        info['environment'] = dict(workspace_root=str(root), gb_root=str(gb),
                                   python_executable=sys.executable, python_version=sys.version.split()[0],
                                   dependency_versions=dependency_versions(), output_writable=True,
                                   dependency_lock=environment_lock)
        info['file'] = dict(exists=True, readable=True, size_bytes=path.stat().st_size)
        info['target_column'] = 'Class' if task is TaskType.CLASSIFICATION else 'Target'
        info['feature_names_preview'] = [str(c) for c in dataset.features.columns[:16]]
        info['feature_names_truncated'] = len(dataset.features.columns) > 16
        imported = path.parent / 'import.json'
        if inside(storage / 'inputs', path) and imported.is_file():
            from input_ops import digest
            info['import'] = json.loads(contained(storage, imported).read_text(encoding='utf-8'))
            if info['import']['normalized_sha256'] != digest(path):
                raise ValueError('Mapped dataset changed; import again.')
        if op == 'inspect':
            return info
        mode = EvaluationMode(args.get('mode', 'exploration'))
        library = args.get('library', 'custom')
        if library not in ('custom', 'full'):
            raise ValueError('Choose custom or full candidate library.')
        defaults = ['PLS-DA'] if task is TaskType.CLASSIFICATION else ['PLS Regression']
        methods = {'preprocessing': args.get('preprocessing', ['Raw']),
                   'selectors': args.get('selectors', ['None']), 'models': args.get('models', defaults)}
        if library == 'full':
            if any(key in args for key in methods):
                raise ValueError('Full library selects all methods; do not combine it with custom method lists.')
            methods = {key: list(info['available_methods'][key]) for key in methods}
        for key, values in methods.items():
            if not values or len(values) != len(set(values)) or any(v not in info['available_methods'][key] for v in values):
                raise ValueError('Invalid/duplicate/empty method selection: ' + key)
        n = len(methods['preprocessing']) * len(methods['selectors']) * len(methods['models'])
        if n > (990 if library == 'full' else 32):
            raise ValueError('Custom plans support at most 32 candidates; choose full for the 990-candidate library.')
        settings = {'task': task.value, 'mode': mode.value, 'outer_folds': args.get('outer_folds', 3),
                    'inner_folds': args.get('inner_folds', 2), 'random_state': args.get('seed', 42),
                    'n_jobs': args.get('jobs', 1), 'validation_ratio': args.get('validation_ratio', 0.2)}
        for key in ('outer_folds', 'inner_folds', 'random_state', 'n_jobs'):
            if type(settings[key]) is not int:
                raise ValueError(key + ' must be an integer')
        if not 1 <= settings['n_jobs'] <= 4 or not 2 <= settings['outer_folds'] <= 10 or not 2 <= settings['inner_folds'] <= 10:
            raise ValueError('jobs must be 1..4; folds must be 2..10')
        config = RunConfig(**settings, output_dir=storage / 'preflight')
        split = split_dataset(dataset, config)
        # Preflight the actual outer/inner folds rather than trusting sample counts.
        from gb_core.engine import EvaluationEngine
        import numpy as np
        engine = EvaluationEngine(split, config)
        splitter = engine._outer_splitter()
        groups = None if split.groups_cal is None else np.asarray(split.groups_cal)
        outer = splitter if isinstance(splitter, list) else splitter.split(engine.X_cal, engine.y_cal, groups)
        for train, test in outer:
            if not len(train) or not len(test):
                raise ValueError('Empty calibration fold')
            engine._inner_splits(enumerate_specs(task, **methods)[0], engine.X_cal[train], engine.y_cal[train],
                                 None if groups is None else groups[train])
        provenance = source_provenance(gb)
        plan = {'schema': 2, 'library': library, 'data': info, 'settings': settings, 'methods': methods,
                'candidate_count': n, 'source_sha256': provenance['combined_sha256'],
                'gb_commit': provenance.get('git_commit'),
                'interpretation': 'CV-based candidate ordering; Val_* is internal holdout, never used for ranking.'}
        plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:24]
        plan['plan_id'] = plan_id
        write_json(storage / 'plans' / (plan_id + '.json'), plan)
        return plan

    plan_id = args['plan_id']
    if not re.fullmatch(r'[0-9a-f]{24}', plan_id):
        raise ValueError('Invalid plan_id')
    plan = json.loads((storage / 'plans' / (plan_id + '.json')).read_text(encoding='utf-8'))
    signed = {key: value for key, value in plan.items() if key != 'plan_id'}
    expected = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()[:24]
    if expected != plan_id or plan.get('plan_id') != plan_id:
        raise ValueError('Saved plan changed. Create a new plan.')
    run_dir = contained(root, storage / 'runs' / plan_id)
    state_path = run_dir / 'state.json'
    if op == 'results':
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'status': 'not_started'}
        result = {'plan_id': plan_id, 'task': plan['settings']['task'], 'state': state}
        if state.get('status') in ('completed', 'completed_with_errors'):
            from gb_core.reporting import structured_summary
            from result_view import build_result_view
            result['summary'] = structured_summary(pd.read_csv(contained(run_dir, run_dir / 'leaderboard_v3.csv')), TaskType(plan['settings']['task']), top_n=5)
            result.update(build_result_view(run_dir, plan, state, result['summary'], write_report))
            from model_ops import candidate_table
            offset, limit = args.get('candidate_offset', 0), args.get('candidate_limit', 20)
            if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 990:
                raise ValueError('Candidate offset must be nonnegative; limit must be 1..990.')
            all_rows = candidate_table(run_dir)
            result['candidates'] = all_rows[offset:offset+limit]
            result['candidates_total'] = len(all_rows)
            result['candidate_page'] = dict(offset=offset, limit=limit, returned=len(result['candidates']))
        return result
    if op == 'export':
        from model_ops import export_model
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
        if state.get('status') not in ('completed', 'completed_with_errors'):
            raise ValueError('Export requires a completed saved run.')
        with run_lock(run_dir / 'worker.lock'):
            return export_model(root, storage, gb, run_dir, plan, args, contained, write_json)
    if op != 'run':
        raise ValueError('Unsupported operation')
    dataset = load_dataset(contained(root, plan['data']['path']), TaskType(plan['settings']['task']))
    saved_lock = plan['data'].get('environment', {}).get('dependency_lock')
    if not saved_lock or saved_lock['lock_sha256'] != environment_lock['lock_sha256'] or saved_lock['packages'] != environment_lock['packages']:
        raise ValueError('Python environment changed since planning. Create a new plan under the pinned runtime.')
    if dataset.fingerprint != plan['data']['fingerprint']:
        raise ValueError('Dataset changed after planning. Create a new plan.')
    if source_provenance(gb)['combined_sha256'] != plan['source_sha256']:
        raise ValueError('GB source changed after planning. Create a new plan.')
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir / 'worker.lock'):
        config = RunConfig(**plan['settings'], output_dir=run_dir)
        split = split_dataset(dataset, config)
        specs = enumerate_specs(config.task, **plan['methods'], seed=config.random_state)
        from gb_core.engine import EvaluationEngine
        from gb_core.plotting import create_visualization_suite
        state = {'status': 'running', 'completed': 0, 'total': len(specs), 'started_at': time.time()}
        write_json(state_path, state)
        def progress(done, total, spec):
            state.update(completed=done, total=total)
            write_json(state_path, state)
            emit({'event': 'progress', 'completed': done, 'total': total})
        try:
            # All library logging goes to stderr, keeping stdout a strict NDJSON channel.
            with contextlib.redirect_stdout(sys.stderr):
                outputs = EvaluationEngine(split, config, logger=lambda s: print(s, file=sys.stderr)).run(specs, progress=progress)
                create_visualization_suite(outputs.results, run_dir / 'plots', config.task)
            frame = outputs.results
            failed = int(frame['Status'].ne('SUCCESS').sum())
            manifest = json.loads(outputs.manifest.read_text(encoding='utf-8'))
            issues = failed or manifest.get('internal_holdout_failures', 0) or manifest.get('default_export_failed', False)
            state.update(status='completed_with_errors' if issues else 'completed', failed=failed, finished_at=time.time())
            write_json(state_path, state)
            return {'plan_id': plan_id, 'state': state, 'output_dir': str(run_dir)}
        except BaseException as exc:
            state.update(status='failed', error=str(exc), finished_at=time.time())
            write_json(state_path, state)
            raise


def main():
    try:
        req = json.load(sys.stdin)
        result = dispatch(req)
        emit({'event': 'result', 'value': result})
    except Exception as exc:
        emit({'event': 'error', 'message': str(exc), 'error_type': type(exc).__name__})
        return 2
    return 0


if __name__ == '__main__':
    # Windows multiprocessing must import this module without starting a second request.
    raise SystemExit(main())
