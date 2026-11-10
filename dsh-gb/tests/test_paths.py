from pathlib import Path
import os
import subprocess
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from path_policy import contained, inside


class PathPolicy(unittest.TestCase):
    def test_windows_aliases_and_sibling_rejection(self):
        for root, child in [(r'C:\Work Space', r'\\?\C:\work space\数据\input.csv'),
                            (r'\\?\C:\Work Space', r'c:\work space\new\result.csv'),
                            (r'\\server\share\workspace', r'\\?\UNC\server\share\workspace\结果.csv')]:
            self.assertTrue(inside(root, child, windows=True))
        for bad in [r'C:\workspace2\data.csv', r'D:\workspace\data.csv', r'C:\elsewhere\data.csv']:
            self.assertFalse(inside(r'C:\workspace', bad, windows=True))

    def test_real_existing_and_missing_children(self):
        folder = ROOT / '.test-output' / '路径边界测试'
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(100):
            self.assertEqual(contained(folder, folder / 'new' / str(i) / '结果.csv'), folder.resolve() / 'new' / str(i) / '结果.csv')
        with self.assertRaisesRegex(ValueError, 'inside'):
            contained(folder, folder / '..' / 'outside.csv')

    @unittest.skipUnless(os.name == 'nt', 'Windows junction behavior')
    def test_junction_inside_and_outside_boundary(self):
        folder = ROOT / '.test-output' / 'junction-policy'
        target = folder / 'target'; target.mkdir(parents=True, exist_ok=True)
        alias = folder / 'alias'
        outside = folder / 'outside-alias'
        # New test-only junctions; never delete or replace any existing path.
        for path, destination in [(alias, target), (outside, ROOT.parent)]:
            if not path.exists():
                command = 'New-Item -ItemType Junction -Path $env:GB_TEST_LINK -Value $env:GB_TEST_TARGET | Out-Null'
                result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                                        env={**os.environ, 'GB_TEST_LINK': str(path), 'GB_TEST_TARGET': str(destination)}, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertEqual(contained(folder, alias / 'future.csv'), target.resolve() / 'future.csv')
        with self.assertRaisesRegex(ValueError, 'inside'):
            contained(folder, outside / 'future.csv')
