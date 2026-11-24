import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recompute_public_metrics',ROOT/'scripts/recompute_public_metrics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def episode(label,reward,version='v6',split='test',seed=1):
    return {'label':label,'joint_return':reward,'execution_time_s':1.,'version':version,
        'split':split,'seed':seed,'red':'finite','status':'completed','kind':'real_cage4'}


class PublicMetricsTests(unittest.TestCase):
    def test_versions_validation_and_test_are_separate(self):
        values=[episode('B0',-10),episode('B4',-5),episode('B4',-100,split='validation'),episode('B4',-1000,version='v5')]
        result=module.recompute(values)
        self.assertEqual(len(result['groups']),4)
        self.assertEqual(result['paired_vs_B0'][0]['mean_difference'],5)

    def test_duplicate_seed_and_broken_reward_totals_reject(self):
        with self.assertRaises(ValueError):module.recompute([episode('B0',-1),episode('B0',-2)])
        broken=episode('B0',-2);broken['scored_step_rewards']=[-1]
        with self.assertRaises(ValueError):module.recompute([broken])

    def test_mock_and_incomplete_are_not_performance_samples(self):
        mock=episode('B0',999);mock['kind']='mock_test'
        partial=episode('B4',998);partial['status']='partial'
        self.assertEqual(module.recompute([mock,partial])['groups'],[])


if __name__=='__main__':unittest.main()
