import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import dependency_policy


class DependencyLock(unittest.TestCase):
    def test_current_environment_exactly_matches_declared_pins(self):
        status = dependency_policy.require_lock()
        self.assertTrue(status['matches'])
        self.assertGreaterEqual(len(status['packages']), 43)

    def test_wrong_version_is_reported_and_blocks_computation(self):
        original = dependency_policy.importlib.metadata.version
        with patch('dependency_policy.importlib.metadata.version', side_effect=lambda name: '0.0.0' if name == 'numpy' else original(name)):
            self.assertFalse(dependency_policy.lock_status()['matches'])
            with self.assertRaisesRegex(ValueError, 'numpy'):
                dependency_policy.require_lock()

    def test_changed_lock_file_is_rejected(self):
        folder = ROOT / '.test-output'; folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            copied = Path(directory).resolve()
            self.assertTrue(copied.is_relative_to(folder.resolve()))
            (copied / 'python').mkdir()
            for filename in ('requirements.lock', 'requirements.versions.json', 'python/dependencies.lock.json'):
                shutil.copyfile(ROOT / filename, copied / filename)
            with (copied / 'requirements.lock').open('a') as stream:
                stream.write('# changed\n')
            with patch('dependency_policy.ROOT', copied), self.assertRaisesRegex(ValueError, 'lock changed'):
                dependency_policy.lock_status()
