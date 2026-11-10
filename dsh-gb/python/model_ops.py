"""Export verified GB candidates and predict with this session's registered bundles only."""
import contextlib
import json
from pathlib import Path
import re
import secrets
import sys
from input_ops import digest, table


def candidate_table(run_dir):
    import pandas as pd
    frame = pd.read_csv(run_dir / 'leaderboard_v3.csv', dtype={'Pipeline_ID': str})
    # JSON uses null for NaN/Inf, never non-standard JSON numbers.
    frame = frame.replace([float('inf'), float('-inf')], float('nan'))
    return json.loads(frame.to_json(orient='records'))


def export_model(root, storage, gb, run_dir, plan, args, contained, write_json):
    from gb_core.cli import load_stored_run
    from gb_core.contracts import TaskType
    from gb_core.artifacts import source_provenance
    from dependency_policy import require_lock
    environment = require_lock()
    saved_lock = plan['data'].get('environment', {}).get('dependency_lock')
    if not saved_lock or saved_lock['lock_sha256'] != environment['lock_sha256'] or saved_lock['packages'] != environment['packages']:
        raise ValueError('Stored CV run uses a different Python environment; create a new plan before exporting.')
    if source_provenance(gb)['combined_sha256'] != plan['source_sha256']:
        raise ValueError('GB source changed since planning; cannot refit this stored run.')
    source = contained(root, plan['data']['path'])
    pipeline_id = args.get('pipeline_id')
    if not pipeline_id or not isinstance(pipeline_id, str):
        raise ValueError('Choose a pipeline_id from the saved result table.')
    with contextlib.redirect_stdout(sys.stderr):
        engine, specs, ranked, manifest = load_stored_run(run_dir, source, TaskType(plan['settings']['task']), logger=lambda s: print(s, file=sys.stderr))
        exported = engine.export_candidate(ranked, specs, pipeline_id=pipeline_id,
                                          output_dir=contained(storage, run_dir / 'exports'),
                                          selection_reason=args.get('reason') or 'User-selected candidate; holdout may already have been viewed.',
                                          source_manifest=manifest)
    model_id = secrets.token_hex(12)
    bundle = contained(storage, exported.bundle_path)
    export_manifest = contained(storage, exported.manifest_path)
    record = dict(model_id=model_id, plan_id=plan['plan_id'], pipeline_id=pipeline_id,
                  bundle_path=str(bundle), bundle_sha256=digest(bundle), manifest_path=str(export_manifest),
                  manifest_sha256=digest(export_manifest), feature_names=list(engine.feature_names),
                  task=plan['settings']['task'], gb_source_sha256=plan['source_sha256'],
                  dependency_lock_sha256=environment['lock_sha256'], dependency_versions=environment['packages'],
                  validation_status=exported.validation_status, validation_error=exported.validation_error,
                  interpretation='Refitted on calibration data only. Viewing holdout before selecting compromises confirmatory use.')
    write_json(contained(storage, storage / 'models' / (model_id + '.json')), record)
    return dict(**record, presentation=dict(default_files=[dict(path=str(export_manifest), role='manifest', title='模型导出记录')]))


def model_record(storage, model_id, contained):
    if not isinstance(model_id, str) or not re.fullmatch(r'[0-9a-f]{24}', model_id):
        raise ValueError('Invalid model_id; use an export made in this session.')
    path = contained(storage, storage / 'models' / (model_id + '.json'))
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('model_id') != model_id:
        raise ValueError('Invalid registered model.')
    for key in ('bundle', 'manifest'):
        p = contained(storage, record[key + '_path'])
        if digest(p) != record[key + '_sha256']:
            raise ValueError('Registered model artifact changed; export it again.')
    return record


def predict(root, storage, gb, args, contained, write_json):
    import pandas as pd
    from gb_core.artifacts import load_bundle, source_provenance
    record = model_record(storage, args['model_id'], contained)
    from dependency_policy import require_lock
    environment = require_lock()
    if record.get('dependency_lock_sha256') != environment['lock_sha256'] or record.get('dependency_versions') != environment['packages']:
        raise ValueError('Model runtime differs from its export environment; re-export under the pinned environment.')
    if source_provenance(gb)['combined_sha256'] != record['gb_source_sha256']:
        raise ValueError('GB source differs from the exported model runtime.')
    source = contained(root, root / args['path'])
    original_hash = digest(source)
    if args.get('sha256') and original_hash != args['sha256']:
        raise ValueError('Prediction input changed since inspection.')
    data = table(source, args.get('sheet'))
    features = record['feature_names']
    mapping = args.get('feature_mapping') or {c: c for c in features}
    if set(mapping) != set(features) or len(set(mapping.values())) != len(features) or any(c not in data for c in mapping.values()):
        raise ValueError('Prediction feature mapping must match every training feature exactly once.')
    frame = pd.DataFrame({c: data[mapping[c]] for c in features})
    for c in frame:
        numeric = pd.to_numeric(frame[c], errors='coerce')
        if (frame[c].notna() & numeric.isna()).any() or not numeric.notna().any():
            raise ValueError(f'Prediction feature is not numeric: {c}')
        frame[c] = numeric
    sample_column = args.get('sample_column')
    if sample_column and (sample_column not in data or sample_column in mapping.values()):
        raise ValueError('Choose a sample column separate from predictors.')
    sample_column = sample_column or ('Sample name' if 'Sample name' in data and 'Sample name' not in mapping.values() else None)
    # Unpickle only a bundle registered by this plugin, after confinement and digest checks.
    bundle = load_bundle(contained(storage, record['bundle_path']))
    predictions = bundle.predict_frame(frame)
    if digest(source) != original_hash:
        raise ValueError('Prediction input changed during reading.')
    prediction_id = secrets.token_hex(12)
    folder = contained(storage, storage / 'predictions' / prediction_id)
    folder.mkdir(parents=True, exist_ok=False)
    destination = folder / 'predictions.csv'
    label = 'Predicted_Class' if record['task'] == 'classification' else 'Predicted_Target'
    pd.DataFrame({'Sample name': data[sample_column].astype(str) if sample_column else [f'Sample_{i+1}' for i in range(len(data))], label: predictions}).to_csv(destination, index=False, encoding='utf-8-sig')
    result = dict(model_id=record['model_id'], prediction_id=prediction_id, rows=len(data), path=str(destination),
                  input_path=str(source), input_sha256=original_hash, feature_mapping=mapping,
                  ignored_columns=[c for c in data if c not in mapping.values() and c != sample_column],
                  interpretation='Predictions only; no accuracy estimate without independent observed targets.')
    write_json(folder / 'prediction_manifest.json', result)
    result['presentation'] = dict(default_files=[dict(path=str(destination), role='table', title='新样本预测')])
    return result
