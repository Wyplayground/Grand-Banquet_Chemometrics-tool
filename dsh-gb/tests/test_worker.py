import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GB = ROOT / '.runtime' / 'gb'
sys.path.insert(0, str(GB))


class WorkerIntegration(unittest.TestCase):
    def call(self, op, args, ok=True, scope='test'):
        request = dict(op=op, args=args, scope=scope, gbRoot=str(GB), workspaceRoot=str(ROOT))
        p = subprocess.run([sys.executable, str(ROOT / 'python' / 'worker.py')], input=json.dumps(request),
                           capture_output=True, text=True, encoding='utf-8', cwd=ROOT, timeout=180)
        events = [json.loads(line) for line in p.stdout.splitlines()]
        if ok:
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr[-3000:])
            self.assertEqual(events[-1]['event'], 'result')
            return events[-1]['value']
        self.assertNotEqual(p.returncode, 0)
        return events[-1]

    def test_classification_and_regression_match_direct_engine(self):
        from gb_core.contracts import TaskType, RunConfig, enumerate_specs
        from gb_core.data import create_demo_data, load_dataset, split_dataset
        from gb_core.engine import EvaluationEngine
        import pandas as pd
        for task in TaskType:
            with self.subTest(task=task.value):
                path = ROOT / '.test-output' / (task.value + '.csv')
                path.parent.mkdir(exist_ok=True)
                create_demo_data(path, task, n_samples=48, n_features=16)
                info = self.call('inspect', dict(task=task.value, path=str(path)))
                self.assertEqual(info['samples'], 48)
                self.assertNotIn('sample_names', info)
                self.assertTrue(info['environment']['output_writable'])
                self.assertTrue(info['file']['readable'])
                self.assertEqual(len(info['feature_names_preview']), 16)
                plan = self.call('plan', dict(task=task.value, path=str(path)))
                run = self.call('run', dict(plan_id=plan['plan_id']))
                self.assertEqual(run['state']['status'], 'completed')
                result = self.call('results', dict(plan_id=plan['plan_id']))
                self.assertEqual(result['summary']['successful_pipelines'], 1)
                presented = result['presentation']['default_files']
                self.assertEqual([p['role'] for p in presented], ['summary', 'table', 'plot'])
                self.assertTrue(all(Path(p['path']).is_file() for p in presented))
                self.assertTrue(any(k.startswith('plots/') for k in result['artifacts']))
                self.assertFalse(result['artifact_index']['truncated'])
                report = Path(result['artifacts']['gb_summary.md']).read_text(encoding='utf-8')
                self.assertIn('CV_* orders', report)
                self.assertIn('Val_* is same-source internal holdout', report)
                self.assertIn('](plots/', report)
                for row in result['summary']['top_pipelines']:
                    from importlib import import_module
                    sys.path.insert(0, str(ROOT / 'python'))
                    format_cell = import_module('result_view').cell
                    metric = 'CV_Accuracy' if task is TaskType.CLASSIFICATION else 'CV_RMSE'
                    self.assertIn(format_cell(row[metric]), report)
                config = RunConfig(**plan['settings'], output_dir=ROOT / '.test-output' / ('direct-' + task.value))
                data = load_dataset(path, task)
                direct = EvaluationEngine(split_dataset(data, config), config).run(enumerate_specs(task, **plan['methods'], seed=config.random_state))
                actual = pd.read_csv(result['artifacts']['leaderboard_v3.csv'])
                for column in direct.results:
                    if column.startswith(('CV_', 'Val_', 'Cal_')) and pd.api.types.is_numeric_dtype(direct.results[column]):
                        pd.testing.assert_series_equal(actual[column], direct.results[column].reset_index(drop=True), check_dtype=False, atol=1e-10, rtol=1e-10)
                resumed = self.call('run', dict(plan_id=plan['plan_id']))
                self.assertEqual(resumed['state']['completed'], 1)

    def test_bad_input_and_path_escape(self):
        error = self.call('inspect', dict(task='regression', path='../outside.csv'), ok=False)
        self.assertIn('inside', error['message'])
        error = self.call('results', dict(plan_id='../escape'), ok=False)
        self.assertIn('Invalid', error['message'])

    def test_plan_rejects_changed_data_and_other_session(self):
        from gb_core.contracts import TaskType
        from gb_core.data import create_demo_data
        path = ROOT / '.test-output' / 'changing.csv'
        create_demo_data(path, TaskType.REGRESSION, n_samples=24, n_features=8)
        plan = self.call('plan', dict(task='regression', path=str(path)))
        import pandas as pd
        changed = pd.read_csv(path)
        changed.loc[0, 'Target'] += 100
        changed.to_csv(path, index=False)
        self.assertIn('changed', self.call('run', dict(plan_id=plan['plan_id']), ok=False)['message'])
        self.call('results', dict(plan_id=plan['plan_id']), ok=False, scope='different-session')


if __name__ == '__main__':
    unittest.main()
