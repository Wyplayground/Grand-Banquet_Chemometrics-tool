import base64
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / '.runtime' / 'gb'))
from worker import dispatch
from gb_core.contracts import TaskType
from gb_core.data import create_demo_data
from gb_core.artifacts import load_bundle
import pandas as pd


class WorkbenchFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / '.test-output' / 'workbench'
        cls.folder.mkdir(parents=True, exist_ok=True)

    def call(self, op, args=None, scope='workbench-tests'):
        return dispatch(dict(op=op, args=args or {}, scope=scope, workspaceRoot=str(ROOT), gbRoot=str(ROOT / '.runtime' / 'gb')))

    def test_mapping_export_prediction_roundtrip_for_both_tasks(self):
        for task in TaskType:
            with self.subTest(task=task.value):
                source = self.folder / (task.value + '.csv')
                create_demo_data(source, task, n_samples=36, n_features=8)
                target = 'Class' if task is TaskType.CLASSIFICATION else 'Target'
                frame = pd.read_csv(source).rename(columns={'Sample name': '编号', target: '测量结果'})
                frame['实验备注'] = 'do not train on this text'
                frame.to_csv(source, index=False)
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                metadata = self.call('columns', dict(path=str(source)))
                features = [c for c in frame if c not in ('编号', '测量结果', '实验备注')]
                imported = self.call('prepare', dict(task=task.value, path=str(source), sha256=metadata['sha256'], mapping=dict(sample='编号', target='测量结果', features=features)))
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
                self.assertEqual(imported['retained_rows'], 36)
                plan = self.call('plan', dict(task=task.value, path=imported['path'], preprocessing=['Raw', 'SNV']))
                self.assertEqual(plan['data']['import']['original_sha256'], source_hash)
                self.call('run', dict(plan_id=plan['plan_id']))
                results = self.call('results', dict(plan_id=plan['plan_id']))
                self.assertEqual(len(results['candidates']), 2)
                row = next(r for r in results['candidates'] if r['Status'] == 'SUCCESS')
                exported = self.call('export', dict(plan_id=plan['plan_id'], pipeline_id=row['Pipeline_ID'], reason='Test explicit candidate'))
                self.assertEqual(exported['pipeline_id'], row['Pipeline_ID'])
                self.assertEqual(exported['feature_names'], features)
                new = self.folder / (task.value + '-predict.csv')
                # Reordered and renamed predictors plus irrelevant metadata.
                sample = frame.iloc[:5].copy().rename(columns={c: 'new_' + c for c in features})
                sample.loc[:, list(reversed(sample.columns))].to_csv(new, index=False)
                mapping = {c: 'new_' + c for c in features}
                predicted = self.call('predict', dict(model_id=exported['model_id'], path=str(new), sample_column='编号', feature_mapping=mapping))
                self.assertEqual(predicted['rows'], 5)
                self.assertIn('实验备注', predicted['ignored_columns'])
                actual = pd.read_csv(predicted['path']).iloc[:, 1]
                expected = load_bundle(exported['bundle_path']).predict_frame(frame.loc[:4, features])
                if task is TaskType.REGRESSION:
                    import numpy as np
                    np.testing.assert_allclose(actual, expected, rtol=1e-12)
                else:
                    self.assertEqual(actual.astype(str).tolist(), list(map(str, expected)))
                artifact = self.call('artifact', dict(model_id=exported['model_id'], kind='bundle'))
                self.assertEqual(base64.b64decode(artifact['base64']), Path(exported['bundle_path']).read_bytes())
                with self.assertRaises((FileNotFoundError, ValueError)):
                    self.call('predict', dict(model_id=exported['model_id'], path=str(new)), scope='foreign-workbench')
                with self.assertRaisesRegex(ValueError, 'match every training feature'):
                    self.call('predict', dict(model_id=exported['model_id'], path=str(new), feature_mapping={features[0]: 'new_' + features[0]}))
                with self.assertRaisesRegex(ValueError, 'Unknown or ineligible'):
                    self.call('export', dict(plan_id=plan['plan_id'], pipeline_id='not-a-candidate'))
                with self.assertRaisesRegex(ValueError, 'inside'):
                    self.call('artifact', dict(plan_id=plan['plan_id'], name='../../plans/private.json'))
                bundle = Path(exported['bundle_path'])
                bundle.write_bytes(bundle.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    self.call('predict', dict(model_id=exported['model_id'], path=str(new), feature_mapping=mapping))

    def test_mapping_rejections_excel_and_freshness(self):
        source = self.folder / 'arbitrary.xlsx'
        frame = pd.DataFrame({'id': list('abcdefghi'), 'y': range(9), 'x': range(9), 'batch': ['a']*9})
        with pd.ExcelWriter(source) as writer:
            frame.to_excel(writer, sheet_name='数据', index=False)
            frame.rename(columns={'y': 'answer'}).to_excel(writer, sheet_name='other', index=False)
        cols = self.call('columns', dict(path=str(source), sheet='other'))
        self.assertEqual(cols['sheets'], ['数据', 'other'])
        self.assertIn('answer', [c['name'] for c in cols['columns']])
        args = dict(task='regression', path=str(source), sheet='other', sha256=cols['sha256'], mapping=dict(target='answer', sample='id', features=['x']))
        mapped = self.call('prepare', args)
        self.assertEqual(mapped['retained_rows'], 9)
        with self.assertRaisesRegex(ValueError, 'one role'):
            self.call('prepare', {**args, 'mapping': dict(target='answer', features=['answer'])})
        with self.assertRaisesRegex(ValueError, 'not numeric'):
            self.call('prepare', {**args, 'mapping': dict(target='answer', features=['batch'])})
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.call('prepare', {**args, 'sha256': 'old-hash'})
        csv = self.folder / 'duplicate.csv'
        csv.write_text('y,x,x\n1,2,3\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.call('columns', dict(path=str(csv)))
        discovered = self.call('files', dict(directory=str(self.folder)))
        self.assertIn('arbitrary.xlsx', [f['name'] for f in discovered['entries']])


if __name__ == '__main__':
    unittest.main()
