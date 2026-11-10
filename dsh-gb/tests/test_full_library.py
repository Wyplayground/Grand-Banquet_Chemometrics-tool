import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / '.runtime' / 'gb'))
from worker import dispatch
from gb_core.data import create_demo_data
from gb_core.contracts import TaskType, enumerate_specs


class FullLibrary(unittest.TestCase):
    def call(self, op, args):
        return dispatch(dict(op=op,args=args,scope='full-library-tests',gbRoot=str(ROOT / '.runtime' / 'gb'),workspaceRoot=str(ROOT)))

    def test_full_library_contains_990_unique_specs_for_both_tasks(self):
        for task in TaskType:
            path = ROOT / '.test-output' / ('library-' + task.value + '.csv')
            path.parent.mkdir(exist_ok=True)
            create_demo_data(path, task, n_samples=32, n_features=8)
            args = dict(task=task.value,path=str(path),library='full')
            plan = self.call('plan', args)
            self.assertEqual(plan['candidate_count'], 990)
            self.assertEqual(len(set(s.pipeline_id for s in enumerate_specs(task, **plan['methods']))), 990)
            with self.assertRaisesRegex(ValueError, 'do not combine'):
                self.call('plan', {**args, 'models': ['PLS-DA' if task is TaskType.CLASSIFICATION else 'PLS Regression']})

    def test_failed_candidate_is_explicit_and_pages_do_not_hide_it(self):
        path = ROOT / '.test-output' / 'candidate-failure.csv'
        create_demo_data(path, TaskType.REGRESSION, n_samples=12, n_features=8)
        plan = self.call('plan', dict(task='regression',path=str(path),outer_folds=2,inner_folds=2,models=['PLS Regression','KNN Regression']))
        result = self.call('run', dict(plan_id=plan['plan_id']))
        self.assertEqual(result['state']['status'], 'completed_with_errors')
        first = self.call('results', dict(plan_id=plan['plan_id'],candidate_limit=1))
        second = self.call('results', dict(plan_id=plan['plan_id'],candidate_limit=1,candidate_offset=1))
        self.assertEqual(first['candidates_total'], 2)
        self.assertNotEqual(first['candidates'][0]['Pipeline_ID'], second['candidates'][0]['Pipeline_ID'])
        self.assertEqual(first['summary']['failed_pipelines'], 1)
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.call('results', dict(plan_id=plan['plan_id'],candidate_limit=0))
