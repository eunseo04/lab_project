from .searchlight.gameplay.agents import PromptGenerator
from searchlightimprove.prompts.prompts import *
from baseline import CTFState

class CTFSPromptGenerator(PromptGenerator):
    '''
    Generates prompts for CTF action self improvement 
    '''
    def __init__(self, function_signature: str, sys_prompt: str):
        super().__init__(
            environment_rules= CTF_RULES,
            function_signature=function_signature,
            sys_prompt=sys_prompt
        )
    
    @staticmethod
    def gen_state_description(state: CTFState):
        state_description = f'''
        The current CTF environment state is as follows:

        - Attacker OS: {state.attacker_os}
        - Attacker IP: {state.attacker_ip}
        - Target IP: {state.target_ip}
        - Target OS: {state.target_os}

        - Discovered services: {state.discovered_services}
        - Discovered vulnerabilities: {state.discovered_vulnerabilities}

        - Previous observations: {state.previous_observations}

        - Completed objectives: {state.completed_objectives}
        - Current objective: {state.current_objective}

        - Current attack phase: {state.attack_phase}
        '''

        return state_description
    
class CTFValuePromptGenerator(CTFSPromptGenerator):

    def __init__(self):
        super().__init__(function_signature=CTF_FUNCTION_SIGNATURE, sys_prompt=SYS_PROMPT)

class CTFPolicyPromptGenerator(CTFSPromptGenerator):

    def __init__(self):
        super().__init__(function_signature=CTF_FUNCTION_SIGNATURE, sys_prompt=SYS_PROMPT_POLICY)