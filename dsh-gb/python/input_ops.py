"""Bounded file discovery and explicit, provenance-bound copies for column mapping."""
import hashlib
import json
from pathlib import Path
import csv

FORMATS = {'.csv', '.xlsx', '.xls'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(path, sheet=None):
    import pandas as pd
    path = Path(path)
    if path.suffix.lower() not in FORMATS:
        raise ValueError('Choose CSV, XLSX or XLS.')
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('Interactive import supports files up to 128 MiB.')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            raise ValueError('Duplicate column names are ambiguous; rename them first.')
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, header=None)
        if frame.empty:
            raise ValueError('The sheet is empty.')
        header = frame.iloc[0].tolist()
        if any(pd.isna(c) for c in header) or len(set(map(str, header))) != len(header):
            raise ValueError('Missing or duplicate column names; rename them first.')
        frame = frame.iloc[1:].reset_index(drop=True)
        frame.columns = header
    frame.columns = frame.columns.map(str)
    if frame.empty or len(frame.columns) > 20000:
        raise ValueError('The table is empty or exceeds 20,000 columns.')
    return frame


def discover(root, directory, contained):
    folder = contained(root, root / directory)
    if not folder.is_dir():
        raise ValueError('Choose a workspace directory.')
    entries = []
    truncated = False
    for p in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if p.name.startswith('.') or p.is_symlink():
            continue
        resolved = contained(root, p)
        if resolved.is_dir() or resolved.suffix.lower() in FORMATS:
            if len(entries) == 200:
                truncated = True
                break
            entries.append(dict(path=str(resolved.relative_to(root)), name=p.name,
                                kind='directory' if p.is_dir() else 'data'))
    return dict(directory=str(folder.relative_to(root)), entries=entries, truncated=truncated)


def columns(path, sheet=None):
    import pandas as pd
    frame = table(path, sheet)
    sheets = []
    if path.suffix.lower() != '.csv':
        with pd.ExcelFile(path) as book:
            sheets = book.sheet_names
    return dict(path=str(path), sha256=digest(path), rows=len(frame), sheet=sheet,
                sheets=sheets, columns=[dict(name=c, numeric_values=int(pd.to_numeric(frame[c], errors='coerce').notna().sum()),
                                            missing=int(frame[c].isna().sum())) for c in frame],
                instructions='Select a target and explicit feature columns. Group/Split are preserved only when mapped; they must never be used as predictors.')


def prepare(root, storage, args, contained, write_json):
    import pandas as pd
    path = contained(root, root / args['path'])
    original_hash = digest(path)
    if args.get('sha256') != original_hash:
        raise ValueError('Source changed since column inspection; inspect it again.')
    frame = table(path, args.get('sheet'))
    mapping = args['mapping']
    if not isinstance(mapping, dict) or set(mapping) - {'target', 'sample', 'group', 'split', 'features'}:
        raise ValueError('Unknown column mapping field.')
    features = mapping.get('features')
    roles = [mapping[k] for k in ('target', 'sample', 'group', 'split') if mapping.get(k)]
    if not mapping.get('target') or not isinstance(features, list) or not features:
        raise ValueError('Choose one target and at least one feature.')
    chosen = roles + features
    if any(not isinstance(c, str) or c not in frame for c in chosen) or len(chosen) != len(set(chosen)):
        raise ValueError('Columns must exist and each may have only one role.')
    if args['task'] not in ('classification', 'regression'):
        raise ValueError('Unknown analysis task.')
    target = 'Class' if args['task'] == 'classification' else 'Target'
    reserved = {'Sample name', 'Class', 'Target', 'Group', 'Split'}
    if any(c in reserved for c in features):
        raise ValueError('Reserved GB metadata columns cannot be predictors.')
    normalized = pd.DataFrame({'Sample name': frame[mapping['sample']].astype(str) if mapping.get('sample') else [f'Sample_{i+1}' for i in range(len(frame))], target: frame[mapping['target']]})
    for field, name in [('group', 'Group'), ('split', 'Split')]:
        if mapping.get(field):
            if frame[mapping[field]].isna().any():
                raise ValueError(f'{field} column contains missing values.')
            normalized[name] = frame[mapping[field]]
    for c in features:
        numeric = pd.to_numeric(frame[c], errors='coerce')
        # Reject text silently becoming missing values; existing missing values remain explicit.
        if (frame[c].notna() & numeric.isna()).any() or not numeric.notna().any():
            raise ValueError(f'Feature is not numeric: {c}')
        normalized[c] = numeric
    if 'Split' in normalized and not normalized['Split'].astype(str).str.lower().isin({'train', 'training', 'cal', 'calibration', 'val', 'validation', 'holdout', 'test'}).all():
        raise ValueError('Split labels must be train/calibration or validation/holdout/test.')
    if digest(path) != original_hash:
        raise ValueError('Source changed while importing; inspect it again.')
    record = dict(original_path=str(path), original_sha256=original_hash, sheet=args.get('sheet'),
                  task=args['task'], mapping=mapping, original_rows=len(frame), generated_sample_names=not mapping.get('sample'))
    input_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:24]
    folder = contained(root, storage / 'inputs' / input_id)
    folder.mkdir(parents=True, exist_ok=True)
    destination = contained(storage, folder / 'data.csv')
    normalized.to_csv(destination, index=False, encoding='utf-8-sig')
    from gb_core.data import load_dataset
    from gb_core.contracts import TaskType
    dataset = load_dataset(destination, TaskType(args['task']))
    record.update(input_id=input_id, path=str(destination), normalized_sha256=digest(destination),
                  retained_rows=len(dataset.target), excluded_target_rows=len(frame)-len(dataset.target))
    write_json(contained(storage, folder / 'import.json'), record)
    return record
