"""Post-registration branch checks; synthetic arithmetic is never performance."""
from copy import deepcopy
from research.rollout import verify_transaction,pool_digest
from research.skills import SecuritySkillPool
from research.upstream import Skill
from test_research import offline
from test_rollout_verification import episodes,rule


def prepared():
    pool=SecuritySkillPool(7)
    pool.skills[0].frequency=2;pool.skills[0].avg_gain=-1.;pool.skills[0].total_gain=-2.
    return pool,Skill('checked_child','When an observed lead exists',['Inspect the actual outcome'],'When reviewed')


def test_capacity_pruning_on_rejected_copy_does_not_remove_live_parent(offline):
    pool,candidate=prepared();before=deepcopy(pool.snapshot());removed=pool.skills[0].name;seen={}
    def evaluate(snapshot,role):
        seen[role]=snapshot
        return episodes(0 if role=='previous' else -10)
    result=verify_transaction(pool,candidate,evaluate,rule())
    assert len(seen['candidate']['skills'])==7
    assert removed not in {s['name'] for s in seen['candidate']['skills']}
    assert not result['accepted'] and pool.snapshot()==before
    assert result['maintenance_before_validation']


def test_admitted_snapshot_includes_the_exact_prevalidated_capacity_pruning(offline):
    pool,candidate=prepared();before=deepcopy(pool.snapshot());removed=pool.skills[0].name
    result=verify_transaction(pool,candidate,lambda snapshot,role:episodes(0 if role=='previous' else 10),rule())
    assert result['accepted'] and pool.snapshot()==before
    pool.restore(result['verified_pool'])
    assert len(pool.skills)==7 and removed not in {s.name for s in pool.skills}
    assert candidate.name in {s.name for s in pool.skills}
    assert pool_digest(pool.snapshot())==result['candidate_pool_sha256']
