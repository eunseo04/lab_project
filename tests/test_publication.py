"""Publication checks use synthetic tokens and never call model APIs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

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

    def test_relative_revision_asset_link_is_safe(self):
        self.assertEqual(check.blob_issues('revisions/frozen/.venv','120000',b'../../.venv'),[])

    def test_absolute_link_is_rejected(self):
        self.assertIn('external_symlink',check.blob_issues('revisions/frozen/.venv','120000',b'/private/environment'))

    def test_secret_values_are_detected_without_printing_them(self):
        synthetic=('ghp_'+'x'*36).encode()
        self.assertIn('github_token',check.blob_issues('config.txt','100644',synthetic))

    def test_raw_trajectory_is_rejected(self):
        self.assertIn('raw_or_transient_output',check.blob_issues('research_outputs/episode.jsonl','100644',b'{}'))

    def test_portable_config_preserves_model_and_admission_criteria(self):
        source=ROOT/'revisions/rollout-v5-reusable/research_configs/rollout-B4.yaml'
        original=source.read_text()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            result=prepare.prepare(ROOT,output)
            copied=(output/'rollout-B4.yaml').read_text()
            self.assertEqual(copied,original.replace(prepare.RECORDED_ROOT,str(ROOT)))
            self.assertIn('Qwen/Qwen3-8B',copied)
            self.assertIn('minimum_win_rate: 0.6',copied)
            self.assertEqual(source.read_text(),original)
            self.assertEqual(len(result),5)


if __name__=='__main__':unittest.main()
