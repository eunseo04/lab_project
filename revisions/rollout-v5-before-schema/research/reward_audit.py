"""Read-only official reward event instrumentation, excluded from policy inputs."""
from collections import Counter
from copy import deepcopy
import json
from CybORG.Simulator.Actions.AbstractActions.Impact import Impact
from CybORG.Simulator.Actions.GreenActions import GreenAccessService,GreenLocalWork


class RewardAudit:
    def __init__(self,env,path):
        self.controller=env.env.environment_controller
        self.path=path;self.events=[];self.totals=Counter();self.ticks=[]
        calc=self.controller.team_reward_calculators['Blue']['BlueRewardMachine']
        native=calc.calculate_reward
        def calculate(current,actions,observations,done,state):
            reward=native(current,actions,observations,done,state)
            weights=calc.get_phase_rewards(state.mission_phase);events=[]
            for agent,choices in actions.items():
                if not choices:continue
                action=choices[0];kind=None
                if not any(s.active for s in state.sessions[agent].values()):continue
                success=observations[agent].observations[0].data['success']
                if isinstance(action,Impact) and success:
                    kind='RIA';host=action.hostname
                elif isinstance(action,(GreenLocalWork,GreenAccessService)) and success==False:
                    kind='LWF' if isinstance(action,GreenLocalWork) else 'ASF';host=state.ip_addresses[action.ip_address]
                if kind:
                    zone=state.hostname_subnet_map[host].value
                    events.append({'kind':kind,'host':host,'zone':zone,'agent':agent,'phase':state.mission_phase,'reward':weights[zone][kind]})
            if sum(e['reward'] for e in events)!=reward:raise RuntimeError('Official reward event decomposition mismatch')
            self.events=events
            return reward
        calc.calculate_reward=calculate

    def capture(self,step,joint,included):
        components=Counter({'action_cost':self.controller.reward['Blue']['action_cost']})
        for event in self.events:components[event['kind']]+=event['reward']
        if sum(components.values())!=joint:raise RuntimeError('Official shared reward decomposition mismatch')
        if included:self.totals.update(components)
        self.ticks.append({'step':step,'score_included':included,'joint_reward':joint,
            'components':dict(components),'events':deepcopy(self.events)})

    def finish(self,total):
        if sum(self.totals.values())!=total:raise RuntimeError('Reward audit total differs from original ledger')
        with self.path.open('w') as stream:
            for tick in self.ticks:stream.write(json.dumps(tick)+'\n')
        return {'reward_components':dict(self.totals),'reward_events':str(self.path),'reward_audit_passed':True}
