"""CAGE4 observation checks for proposed SGs; these do not prove causation."""
COMPONENTS = {'initiation', 'execution', 'termination'}
FINDINGS = {'completed_success', 'completed_failure', 'pending', 'observed_pid',
            'observed_admin_indicator', 'observed_service_state'}


def operation(row):
    completed = (row.get('behavior_outcome') or {}).get('completed_action')
    if completed:
        return completed
    pending = (row.get('behavior_context') or {}).get('pending_action')
    if pending and not row.get('policy_invoked'):
        return {**pending, 'status': row.get('action_status', 'UNKNOWN')}
    from .behavior import chosen_action
    action, target = chosen_action(row)
    status = row.get('action_status', 'UNKNOWN')
    return {'name': action, 'target': target, 'started_at': row['step'],
            'status': status,
            'completed_at': row['step'] + 1 if status in ('TRUE', 'FALSE') and row.get('request_applied') else None}


def restore_boundary(rows, agent, host, at):
    boundaries = []
    for row in rows:
        if row.get('agent') != agent:
            continue
        context = row.get('behavior_context') or {}
        ops = list(context.get('recent_completed_actions', []))
        completed = (row.get('behavior_outcome') or {}).get('completed_action')
        if completed:
            ops.append(completed)
        for item in context.get('open_cases', []):
            if item.get('host') == host and 0 <= item.get('last_successful_restore_at', -1) <= at:
                boundaries.append(item['last_successful_restore_at'])
        for item in ops:
            end = item.get('completed_at')
            if item.get('name') == 'Restore' and item.get('target') == host and item.get('status') == 'TRUE' and isinstance(end, int) and end <= at:
                boundaries.append(end)
    return max(boundaries, default=-1)


def validate_diagnosis(data, rows, provenance):
    """Require typed, falsifiable observation claims for each changed component.

    Free-form prose is not a formal proof. Checked claims establish consistency
    with a partial observation, never hidden compromise or individual reward credit.
    """
    errors, valid = [], []
    claims = data.get('claims')
    if data.get('diagnosis_status') != 'supported' or not isinstance(claims, list) or not claims:
        return {'passed': False, 'status': 'insufficient_evidence', 'errors': ['missing_supported_observation_claims'], 'valid_claims': []}
    catalog = {f'E{i}': row for i, row in enumerate(rows)}
    allowed = {(r.get('component'), r.get('evidence_id')) for r in provenance.get('valid_references', [])}
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append('invalid_claim'); continue
        component = 'execution' if claim.get('component') == 'policy' else claim.get('component')
        row = catalog.get(claim.get('evidence_id'))
        finding = claim.get('finding')
        if row is None or component not in COMPONENTS or (component, claim.get('evidence_id')) not in allowed:
            errors.append('claim_without_matching_citation'); continue
        if finding not in FINDINGS or claim.get('causal_attribution') != 'unknown':
            errors.append('unproven_causal_or_unknown_claim'); continue
        op = operation(row)
        if claim.get('action') != op.get('name'):
            errors.append('action_mismatch'); continue
        host = claim.get('target')
        if not host or host != op.get('target'):
            errors.append('target_mismatch'); continue
        phase = claim.get('phase', 'after')
        if phase not in ('before', 'after'):
            errors.append('invalid_observation_phase'); continue
        recorded_at = row['step'] + int(phase == 'after')
        as_of = claim.get('as_of_step')
        if type(as_of) is not int or as_of != recorded_at:
            errors.append('observation_time_mismatch'); continue
        end, start = op.get('completed_at'), op.get('started_at')
        if finding.startswith('completed_'):
            expected = 'TRUE' if finding == 'completed_success' else 'FALSE'
            if op.get('status') != expected or end is None or phase != 'after' or end > as_of or start is None or end < start:
                errors.append('not_a_completed_action'); continue
        elif finding == 'pending':
            if op.get('status') != 'IN_PROGRESS' or end is not None:
                errors.append('not_pending'); continue
        else:
            body = (row.get('observation_' + phase) or {}).get(host)
            if not isinstance(body, dict):
                errors.append('target_observation_missing'); continue
            if finding == 'observed_pid':
                pids = {str(p['PID']) for p in body.get('Processes', []) if isinstance(p, dict) and 'PID' in p and 'username' not in p}
                if str(claim.get('pid')) not in pids:
                    errors.append('pid_not_observed_at_claimed_time'); continue
            if finding == 'observed_admin_indicator':
                names = {f.get('File Name') for f in body.get('Files', []) if isinstance(f, dict)}
                if claim.get('indicator') not in names & {'escalate.sh', 'escalate.exe'}:
                    errors.append('admin_indicator_not_observed'); continue
            if finding == 'observed_service_state':
                services = body.get('Services', {})
                if not isinstance(services, dict) or claim.get('service') not in services or services[claim['service']] != claim.get('value'):
                    errors.append('service_state_not_observed'); continue
        boundary = restore_boundary(rows, row['agent'], host, as_of)
        if 'restore_epoch' not in claim or claim['restore_epoch'] != boundary:
            errors.append('restore_epoch_mismatch'); continue
        # A newly observed PID is not evidence that an older Remove failed.
        if finding == 'completed_failure' and claim.get('pid') is not None:
            errors.append('call_status_is_not_pid_cleanup_evidence'); continue
        valid.append({**claim, 'component': component, 'status': 'observation_consistent'})
    supported = {c['component'] for c in valid}
    for component in provenance.get('changed_components', []):
        if component not in supported:
            errors.append('missing_diagnosis_for_' + component)
    # Forbid explicit direct joint-reward blame, including structured assertions.
    if data.get('reward_attribution', 'unknown') != 'unknown':
        errors.append('individual_joint_reward_attribution_unidentified')
    return {'passed': bool(valid) and not errors, 'status': 'supported_observation' if valid and not errors else 'insufficient_evidence',
            'errors': errors, 'valid_claims': valid,
            'limitation': 'Typed claims check observed facts and timing. They do not prove causal diagnosis or validate all natural-language advice.'}


PROMPT = '''
### CAGE4 diagnosis consistency schema (Proposed v6, separate from Skill-Pro)
Keep the original initiation/policy/termination semantic gradients. In addition,
return diagnosis_status="supported|unknown|insufficient_evidence", reward_attribution="unknown",
and claims=[{"component":"policy", "evidence_id":"E0", "action":"Remove",
"target":"actual host from that record", "finding":"completed_failure", "phase":"after",
"as_of_step":1, "restore_epoch":-1, "causal_attribution":"unknown"}].
Allowed findings: completed_success, completed_failure, pending, observed_pid (include pid),
observed_admin_indicator (include indicator), observed_service_state (include service and value).
Use one matching evidence citation and observation-consistent claim for EVERY changed component.
Action and target must describe the cited own operation, not a suggested future operation.
as_of_step equals record step for before, step+1 for after. restore_epoch is the latest
successful same-host Restore completion visible at that time, or -1 if none is recorded.
IN_PROGRESS never supports completed_failure/success. FALSE is a call status, not proof
of its cause. Observed PIDs/files/services must exist in that exact host observation.
Retained old context is not a fresh observation. New PIDs or a post-Restore evidence
epoch never establish an earlier Remove failure. Joint reward cannot identify a Skill's
causal contribution. If evidence is insufficient, use empty gradients and is_related=false.
Do not claim that these checks prove causation. Never insert evidence IDs into Skills.
'''
