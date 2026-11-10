from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from result_view import build_result_view, cell


class ResultViewTests(unittest.TestCase):
    def test_no_eligible_rows_or_plots_does_not_invent_success(self):
        workspace = Path(__file__).resolve().parents[1] / '.test-output'
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            root = Path(directory)
            self.assertTrue(root.resolve().is_relative_to(workspace.resolve()))
            (root / 'leaderboard_v3.csv').write_text('Model,Status\nPLS,FAILED_CV\n')
            (root / 'leaderboard_v3.html').write_text('<h1>Result</h1>')
            view = build_result_view(root, dict(settings=dict(task='regression'), data=dict(samples=8, features=2)),
                dict(status='completed_with_errors',completed=1,total=1),
                dict(top_pipelines=[],eligible_pipeline_rows=0,successful_pipelines=0,failed_pipelines=1,holdout_failed_pipelines=0),
                lambda path, text: path.write_text(text, encoding='utf-8'))
            self.assertEqual([f['role'] for f in view['presentation']['default_files']], ['summary', 'table'])
            text = (root / 'gb_summary.md').read_text(encoding='utf-8')
            self.assertIn('No eligible candidate', text)
            self.assertIn('Successful: 0 · Failed: 1', text)
            self.assertEqual(view['presentation']['supplementary_files'][0]['role'], 'html')

    def test_cells_are_safe_and_nan_is_not_a_metric(self):
        self.assertEqual(cell(float('nan')), '—')
        self.assertEqual(cell(float('inf')), '—')
        self.assertIn('&lt;script&gt;', cell('<script>'))
        self.assertIn('\\[', cell('[click](https://example.com)'))
