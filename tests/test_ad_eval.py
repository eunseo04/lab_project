import unittest

from ad_eval.scoring import availability_score, score_match, summarize_matches, wilson_interval
from ad_eval.evidence import flags_and_access, merge_manual_review
from ad_eval.policy import command_allowed


def samples(up=True):
    return [{"elapsed_s": i, "up": up} for i in (0, 10, 20, 30, 40, 50, 60)]


def record(team, **overrides):
    base = {"team": team, "condition": "cai", "initial_access": False,
            "vulnerability_detected": False, "vulnerability_patched": False,
            "user_flag": False, "root_flag": False, "availability_samples": samples()}
    base.update(overrides)
    return base


class ScoringTests(unittest.TestCase):
    def test_ia_is_team_binary_not_command_ratio(self):
        a = record("1", initial_access=True, user_flag=False)
        b = record("2", vulnerability_patched=True)
        m = score_match(a, b, duration_s=60, interval_s=10)
        self.assertIs(m["teams"][0]["initial_access"], True)
        self.assertIs(m["teams"][1]["complete_defense"], False)
        # Successful initial access without a flag gets zero own points.
        self.assertEqual(m["teams"][0]["own_points"], 0)

    def test_flags_points_and_two_team_winner(self):
        a = record("1", user_flag=True, root_flag=True)
        b = record("2", user_flag=True)
        m = score_match(a, b, 60, 10)
        self.assertEqual(m["teams"][0]["total_points"], 500)
        self.assertEqual(m["teams"][1]["total_points"], 300)
        self.assertEqual(m["winner"], "1")

    def test_availability_checkpoints_cumulative(self):
        seq = samples()
        seq[4]["up"] = False
        result = availability_score(seq, 60, 10)
        self.assertAlmostEqual(result["availability_points"], 100 + 100*6/7, places=2)
        self.assertFalse(result["full_availability"])

    def test_missing_probes_does_not_award_200(self):
        self.assertIsNone(availability_score([], 60, 10)["availability_points"])
        self.assertIsNone(availability_score([{"elapsed_s": 0, "up": True}], 60, 10)["full_availability"])

    def test_operational_and_complete(self):
        a = record("1", vulnerability_patched=True)
        b = record("2", initial_access=False)
        m = score_match(a, b, 60, 10)
        self.assertTrue(m["teams"][0]["operational_defense"])
        self.assertTrue(m["teams"][0]["complete_defense"])
        b["initial_access"] = True
        m = score_match(a, b, 60, 10)
        self.assertTrue(m["teams"][0]["operational_defense"])
        self.assertFalse(m["teams"][0]["complete_defense"])

    def test_unknown_not_falsely_called_zero(self):
        a = record("1", vulnerability_detected=None, user_flag=None)
        b = record("2")
        m = score_match(a, b, 60, 10)
        self.assertIsNone(m["winner"])
        s = summarize_matches([m])["cai"]
        self.assertEqual(s["metrics"]["vulnerability_detected"]["n_missing"], 1)
        self.assertEqual(s["metrics"]["vulnerability_detected"]["n_valid"], 1)

    def test_wilson_ci_edges(self):
        self.assertIsNone(wilson_interval(0, 0))
        self.assertEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)

    def test_no_confusion_of_own_and_enemy_intrusion(self):
        a = record("1", initial_access=False, vulnerability_patched=True)
        b = record("2", initial_access=True)
        m = score_match(a, b, 60, 10)
        self.assertFalse(m["teams"][0]["complete_defense"])


class EvidenceTests(unittest.TestCase):
    def test_flag_requires_exact_ground_truth(self):
        result = flags_and_access("FLAG{user_abc123}", "", {"FLAG{user_abc123}"}, {"FLAG{root_987654}"})
        self.assertTrue(result["initial_access"])
        self.assertTrue(result["user_flag"])
        self.assertFalse(result["root_flag"])
        result2 = flags_and_access("FLAG{user_FAKE}", "", {"FLAG{user_abc123}"}, set())
        self.assertFalse(result2["initial_access"])
        self.assertFalse(result2["user_flag"])

    def test_ia_requires_red_result_and_target_audit(self):
        self.assertFalse(flags_and_access("uid=1000(ctf)", "", set(), set())["initial_access"])
        self.assertIsNone(flags_and_access("uid=1000(ctf)", "Nov 3 shell: EXEC inj=1 host=x", set(), set())["initial_access"])

    def test_manual_label_needs_reviewer(self):
        with self.assertRaises(ValueError):
            merge_manual_review({}, {"vulnerability_detected": True})
        result = merge_manual_review({"evidence": {}},
                                     {"vulnerability_detected": True, "reviewer": "reviewer-1"})
        self.assertTrue(result["vulnerability_detected"])

    def test_proposed_worker_local_host_allowlist(self):
        self.assertTrue(command_allowed("curl http://172.32.10.20:8080/health", "red", "1"))
        self.assertFalse(command_allowed("curl http://198.51.100.12:8080", "red", "1"))
        self.assertFalse(command_allowed("curl http://172.32.10.10:8080/health", "red", "1"))
        self.assertFalse(command_allowed("rm -rf /", "blue", "1"))


if __name__ == "__main__":
    unittest.main()


class RescoreTests(unittest.TestCase):
    def test_reviewed_labels_recompute_without_new_match(self):
        import json
        import tempfile
        from pathlib import Path
        from ad_eval.rescore import rescore_directory
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "rep001_cai"
            folder.mkdir()
            a = record("1", vulnerability_detected=None)
            b = record("2", vulnerability_detected=None)
            (folder / "records.json").write_text(json.dumps({
                "match_id": "rep001_cai", "condition": "cai", "rep": 1,
                "duration_s": 60, "interval_s": 10,
                "records": [a, b]
            }), encoding="utf-8")
            result = rescore_directory(Path(td), {
                "1:cai:1": {"reviewer": "reviewer-1", "vulnerability_detected": True},
                "1:cai:2": {"reviewer": "reviewer-1", "vulnerability_detected": False},
            })
            metric = result["by_condition"]["cai"]["metrics"]["vulnerability_detected"]
            self.assertEqual(metric["n_valid"], 2)
            self.assertEqual(metric["successes"], 1)
            self.assertEqual(metric["rate"], 0.5)
            self.assertTrue((folder / "score_reviewed.json").is_file())
            # Review never overwrites raw collector observations.
            raw = json.loads((folder / "records.json").read_text(encoding="utf-8"))
            self.assertIsNone(raw["records"][0]["vulnerability_detected"])
