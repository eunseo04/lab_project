"""Publication checks use synthetic tokens and never call model APIs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import yaml
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

check=module('check_publication');prepare=module('prepare_checkout')


class PublicationTests(unittest.TestCase):
    def test_private_home_files_are_rejected(self):
        self.assertIn('private_file_or_directory',check.blob_issues('.codex/state.json','100644',b'{}'))

    def test_model_weights_are_rejected(self):
        self.assertIn('model_or_binary_artifact',check.blob_issues('weights/policy.safetensors','100644',b'weights'))

    def test_unrelated_files_are_rejected(self):
        self.assertIn('outside_research_scope',check.blob_issues('personal/notes.txt','100644',b'notes'))

    def test_relative_internal_asset_link_is_safe(self):
        self.assertEqual(check.blob_issues('artifacts/local-env','120000',b'../.venv'),[])

    def test_absolute_link_is_rejected(self):
        self.assertIn('external_symlink',check.blob_issues('artifacts/local-env','120000',b'/private/environment'))

    def test_secret_values_are_detected_without_printing_them(self):
        synthetic=('ghp_'+'x'*36).encode()
        self.assertIn('github_token',check.blob_issues('config.txt','100644',synthetic))

    def test_raw_trajectory_is_rejected(self):
        self.assertIn('raw_or_transient_output',check.blob_issues('research_outputs/episode.jsonl','100644',b'{}'))

    def test_portable_config_preserves_model_and_admission_criteria(self):
        source=ROOT/'research_configs/rollout-B4.yaml'
        original=source.read_text()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            result=prepare.prepare(ROOT,output)
            copied=(output/'rollout-B4.yaml').read_text()
            expected=yaml.safe_load(original)
            for field in ('weights','output','pool'):
                if expected.get(field):expected[field]=str(ROOT/expected[field])
            self.assertEqual(yaml.safe_load(copied),expected)
            self.assertIn('Qwen/Qwen3-8B',copied)
            self.assertIn('minimum_win_rate: 0.6',copied)
            self.assertEqual(source.read_text(),original)
            self.assertEqual(len(result),6)
            self.assertEqual(yaml.safe_load((output/'rollout-B3-no-support.yaml').read_text())['comparison_id'],'B3-no-support')

    def test_shared_initial_pool_preserves_the_fixed_skill_snapshot(self):
        pool=json.loads((ROOT/'artifacts/initial_pool.json').read_text())
        reference=json.loads((ROOT/'tests/fixtures/reference_integrity.json').read_text())
        self.assertTrue(pool['initialization_only'])
        self.assertEqual(pool['episodes_completed'],0)
        self.assertEqual(hashlib.sha256(json.dumps(pool['pool'],sort_keys=True).encode()).hexdigest(),
                         reference['initial_pool_snapshot_sha256'])
        for label in ['B1','B2','B3','B4']:
            cfg=yaml.safe_load((ROOT/f'research_configs/rollout-{label}.yaml').read_text())
            self.assertEqual(cfg['pool'],'artifacts/initial_pool.json')


if __name__=='__main__':unittest.main()
