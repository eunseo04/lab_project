"""Paper score equations, separate from the published code's LCB maintenance."""
from .gate import returns_to_go


def update_running_baselines(results, baselines, task, config):
    """EMA of complete joint time series; never duplicate by Blue or skill."""
    rewards = [r for result in results for r in result['scored_step_rewards']]
    rtgs = [g for result in results for g in returns_to_go(result['scored_step_rewards'], config['gamma'])]
    alpha = config['ema_alpha']
    for key, values in ((task, rtgs), (task+':step', rewards)):
        measured = sum(values)/len(values)
        baselines[key] = measured if key not in baselines else (1-alpha)*baselines[key]+alpha*measured


def update_online_scores(pool, experiences, step_baseline):
    by_name={s.name:s for s in pool.skills}
    for exp in experiences:
        skill=by_name.get(exp.skill)
        active=[t for t in exp.transitions if t.get('score_included',True)]
        if skill is None or not active: continue
        calls=sum(1 for t in exp.transitions for event in (t.get('skill_controller') or {}).get('events',[])
                  if event.get('activated')==skill.name)
        gain=sum(float(t['joint_reward'])-step_baseline for t in active)/len(active)
        skill.total_gain+=gain
        skill.frequency+=calls
        skill.avg_gain=skill.total_gain/max(1,skill.frequency)


def maintain_paper(pool):
    """Eq.7 online gain / invocation count, dedup and ascending score capacity.

    A new candidate with no invocation has no measured online score. It is kept
    provisionally, not assigned invented gains, until observed or capacity-trimmed.
    This cold-start rule is explicit because the paper does not specify it.
    """
    logs=[]
    for skill in list(pool.skills):
        if skill.frequency>0 and skill.avg_gain<=0:
            pool.skills.remove(skill)
            logs.append({'name':skill.name,'stage':'paper_nonpositive_observed_score','score':skill.avg_gain})
    def remove(skill,stage):
        logs.append({'name':skill.name,'stage':stage,'score':skill.avg_gain})
    pool._semantic_dedup_inplace(pool.skills,score_fn=lambda s:s.avg_gain,
                                log_remove=remove,stage='paper_semantic_duplicate',thr=.95)
    while len(pool.skills)>pool.max_size:
        victim=min(pool.skills,key=lambda s:(s.avg_gain,s.maturity))
        pool.skills.remove(victim);remove(victim,'paper_capacity')
    return logs
