from .prompts import *
from strategist.searchlight.utils import AbstractLogged
import re

#아이디어 프롬프트 생성
class PromptGenerator(AbstractLogged):
    #초기화
    def __init__(self, environment_state, function_signature, attacker_prompt = ATTACKER_PROMPT, seed_heuristic_thought_prompt = HEURISTICS_SEED_THOUGHT_PROMPT):
        self.environment_state = environment_state #가상 환경 정보
        self.function_signature = function_signature
        self.attacker_prompt = attacker_prompt
        self.seed_heuristic_thought_prompt = seed_heuristic_thought_prompt
        super().__init__()
        
    #초기 전략/아이디어를 생각하도록 하는 프롬프트 생성
    def gen_seed_thought_prompt(self):
        return self.attacker_prompt + self.environment_state + self.seed_heuristic_thought_prompt
    
    #생성된 평가 함수를 1~10점으로 평가하도록 하는 프롬프트 생성
    def gen_critic_score_prompt(self, func: str):
        return self.attacker_prompt + self.environment_state + "\n Previously you generated the following function: \n" + func + """\n \n On a scale of 1 to 10, how well do you think this function will perform in evaluating the value of a state in the game? Please provide a brief explanation for your score. 
        You should use the following format for your score and explanation:
        Explanation: <your explanation here>
        Score: <float, your score here>"""
    
    #LLM 응답에서 설명과 점수 추출
    @staticmethod
    def parse_critic_score(string: str) -> tuple[str, float]:
        explanation_pattern = re.compile(r"Explanation: (.+)")
        score_pattern = re.compile(r"Score: (\d+\.\d+)")
        
        explanation = explanation_pattern.findall(string)[0]
        score = float(score_pattern.findall(string)[0])
        
        return explanation, score
    
    #생각해둔 아이디어를 바탕으로 평가 함수 코드 생성 프롬프트
    def gen_seed_function_prompt_with_thought(self, thought):
        return self.attacker_prompt + self.environment_state + "You previously came up with the following thoughts and pseudocode: \n" + thought + "\n \n Implement the function you just described into python code. \n" + self.function_signature 
    
    #아이디어 없이 평가 함수 코드 생성 프롬프트
    def gen_seed_function_prompt_without_thought(self):
        return self.attacker_prompt + self.environment_state + "Given the rules of the game, come up with a function that can be used to evaluate the value of a state in the game." + self.function_signature
    
    #실행 결과/피드백을 보고 문제점 분석
    def gen_feedback_analyze_prompt(self, function, feedback):
        return self.attacker_prompt + self.environment_state + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_INTRO + feedback + CONCLUSION_FROM_FEEDBACK 
    
    #기존 생각을 바탕으로 개선 방향을 다시 생각
    def gen_improvement_thought_prompt(self, previous, prev_thoughts):
        return previous + MODIFY_ABSTRACT + prev_thoughts 
    
    #구체적으로 어떤 부분을 개선할지 아이디어 생성
    def gen_specific_improvement_prompt(self, previous_prompt, conclusions, num_ideas = 1):
        return previous_prompt + conclusions + f'''\n \n Based on the function, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the function that you think will have the most impact? Be as specific and concrete as possible, mentioning specific code pieces or helper functions that you can add. If the game has different phases, you can also mention what specific phase the idea applies to. Write them out in the following format:
        
        Thoughts: <your thoughts here>

        Idea 1: <your idea here>

        Idea 2: <your idea here>

        ...

        Here's an example of what this might look like for 3 improvement ideas:

        Thoughts: The feedback suggests that the main problem is that the attack prompt does not sufficiently account for the available vulnerabilities and the current state of the target environment, which can lead to ineffective attack strategies.

        Idea 1: I can make the prompt explicitly identify the available attack surface and prioritize vulnerabilities that are most relevant to the current target state.

        Idea 2: I can add instructions to analyze the available information and evidence before selecting an attack approach, so that the generated strategy is better aligned with the specific challenge environment.

        Idea 3: I can make the prompt consider the current stage of the attack and the information obtained from previous attempts, allowing the strategy to adapt when an attempted approach does not produce the expected result.

        '''
    
    #LLM 응답에서 idea (n):의 내용만 추출
    @staticmethod
    def parse_improvement_ideas(string: str, num_ideas: int) -> list[str]:
        idea_pattern = re.compile(r"Idea \d+: (.+)")
        matches = idea_pattern.findall(string) #패턴에 해당하는 것을 찾아서 리스트로 반환
        return matches[:num_ideas]
    
    #피드백으로부터 개선에 필요한 결론 도출
    def gen_draw_conclusions_from_feedback_prompt(self, function, feedback):
        return self.attacker_prompt + self.environment_state + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_INTRO + feedback + CONCLUSION_FROM_FEEDBACK 
    
    #개선 아이디어를 실제 함수 코드에 반영하도록 요청
    def gen_implement_function_from_improvement_prompt(self, function, idea):
        return self.attacker_prompt + self.environment_state + PREVIOUS_FUNCTION_INTRO + function + "\n \n Here is a possible way to improve this function: \n" + idea + "\n \n Implement this improvement into the function as best as you can. Make sure not to change the function signature, which we reproduce below: \n" + self.function_signature
    
    #기존 함수를 개선된 버전으로 다시 생성
    def gen_improved_function_prompt(self, previous):
        return previous +  '''\n \n Now modify the function you generated previously to improve it based on the feedback given and your thoughts, and output the new function.\n''' + self.function_signature 
    
    #코드 실행 오류를 LLM에게 전달해서 수정하도록 요청
    def gen_execution_error_feedback(self, previous, function, execution_error):
        return previous + f"You previously generated the following improved function based on your thoughts and analysis of the feedback: \n {function} \n However, it ran into the following error when running: {execution_error} \n Please fix the error and output your function again." 
    
    #실행 오류 메시지만 전달해 수정하도록 요청
    def gen_execution_error_feedback_2(self, error_message):
        string = f'''There was an execution error when running the function you generated on test states. 
        The error message was:
        {error_message}
        Please fix the error and try again.
        '''
        return string
    
    #특정 게임 상태에서 예측값 vs 실제 결과를 보여주는 피드백 생성
    def gen_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        string = f'''---
        Example {i}:
        The state you were trying to estimate a value for is:
        {state_description}

        The function you generated returned the following values:
        {estimated_score}
        for the expected end of game scores of the players. 

        Some intermediate values that you used to calculate the scores were:
        {intermediate_values}

        The estimated end of game scores of the players using lookahead search with your function was:
        {search_score}

        The actual scores of the players at the end of the game in the simulation were:
        {actual_score}
        ---
        '''
        return string
    
    #특정 상태의 예측 결과와 실제 결과를 비교하는 피드백 생성
    def gen_outcome_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        string = f'''---
        Example {i}:
        The state you were trying to estimate a value for is:
        {state_description}

        The function you generated returned the following values:
        {estimated_score}
        for the expected end of game scores of the players. 

        Some intermediate values that you used to calculate the scores were:
        {intermediate_values}

        The actual scores of the players at the end of the game in the simulation were:
        {actual_score}
        ---
        '''
        return string
    
    
#전략 생성
class StrategyPromptGenerator(PromptGenerator):
    PREVIOUS_GUIDE_INTRO = '''You previously generated the following section of the strategy guide: \n'''
    
    #초기화
    def __init__(self, environment_rules):
        function_signature = StrategyPromptGenerator.strategy_guide_signature()
        attacker_prompt = StrategyPromptGenerator.gen_attacker_prompt()
        super().__init__(environment_rules, function_signature, attacker_prompt)

    def gen_critic_score_prompt(self, func: str):
        return self.attacker_prompt + self.environment_state + "\n Previously you generated the following attack prompt: \n" + func + """\n \n On a scale of 1 to 10, how effectively do you think this attack prompt will perform in achieving its intended objective? Please provide a brief explanation for your score.
        You should consider how well the prompt guides the player to make effective decisions, exploit opportunities, and achieve the desired outcome during the game.
        You should use the following format for your score and explanation:
        Explanation: <your explanation here>
        Score: <float, your score here>
        """
    
    @staticmethod
    def parse_critic_score(string: str) -> tuple[str, float]:
        explanation_pattern = re.compile(r"Explanation: (.+)")
        score_pattern = re.compile(r"Score: (\d+\.\d+)")
        
        explanation = explanation_pattern.findall(string)[0]
        score = float(score_pattern.findall(string)[0])
        
        return explanation, score

    @staticmethod
    #전략 가이드 생성 전 질문
    def strategy_guide_signature():
        out = f'''Your guide should be in the form of a worksheet that the strategist can use
        to evaluate the current cyber attack state and construct a high-level strategy.

        The worksheet should contain no more than six questions.
        The questions should be ordered from understanding the current state
        to identifying the strategic objective, prioritizing intermediate goals,
        evaluating risks, and estimating long-term reward.

        The questions must remain at the STRATEGIC level.
        They must NOT ask the reader to generate specific commands,
        exact attack steps, exploit instructions, or a predetermined action sequence.

        The title of your section should be:
        "Six questions to fill out before planning".

        Below is an example of how your worksheet should look like:

        Six questions to fill out before planning

        Q1: What is the current security state of the environment,
        and which characteristics of the state are most favorable or unfavorable
        for achieving the attack objective?

        Q2: What is the primary strategic objective,
        and what intermediate objectives should be prioritized to make progress toward it?

        Q3: Which general attack direction is most promising given the current state,
        and what evidence from the attack dataset or previous observations supports this direction?

        Q4: What characteristics should a future state have for the strategy
        to be considered successful, and how would progress toward that state
        be reflected in the expected reward?

        Q5: What risks, uncertainties, or potential dead ends could make the strategy ineffective,
        and what alternative strategic direction should be considered if those conditions arise?

        Q6: Based on the observed state, previous outcomes, and expected long-term reward,
        should the current strategy be maintained, modified, or abandoned, and why?
        '''
        return out

    @staticmethod
    def gen_attacker_prompt():
        goal = ""
        out = ATTACKER_PROMPT + goal
        return out

    def gen_seed_thought_prompt(self):
        # combine attacker_prompt, GOPS_RULES, HEURISTICS_SEED_PROMPT, and GOPS_FUNCTION_SIGNATURE
        return self.attacker_prompt + self.game_rules + '''Given the rules of the game, write down your thoughts on how to write the strategy guide. Here's an example:
        
        Thought:
        I should first ask the player to consider which other player is most suspicious of them and why. This will help the player understand the dynamics of the game and how they are perceived by others.
        Next I should ask the player to consider which other player is most supportive of them and why. This will help the player understand who they can trust and rely on in the game.
        '''

    def gen_seed_function_prompt(self, seed_thought_prompt):
        # combine seed_thought_prompt and function
        return seed_thought_prompt + "\n Write out the guide based on your thoughts.\n" + self.function_signature
    
    def gen_seed_function_prompt_with_thought(self, thought):
        # combine seed_thought_prompt and function
        return self.attacker_prompt + self.game_rules + "\n You previously came up with the following thoughts: \n" + thought + "\n \n Write out the guide based on your thoughts.\n" + self.function_signature 
    
    def gen_seed_function_prompt_without_thought(self):
        # combine seed_thought_prompt and function
        return self.attacker_prompt + self.game_rules + "\n Given the rules of the game, write out the guide.\n" + self.function_signature
    
    def gen_feedback_analyze_prompt(self, function, feedback):
        '''
        tells the LLM to analyze the feedback
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        out = self.attacker_prompt + self.game_rules + self.PREVIOUS_GUIDE_INTRO + function + '\n Below is some feedback on how your guide performed when a student used it to play the game: \n' + feedback + '''\n \n Based on the feedback given and the guide section you generated previously, what are some conclusions you can draw from the feedback? Make sure to cite the specific examples in the feedback to justify your analysis.\n '''

        # self.logger.info(out)
        return out

    def gen_improvement_thought_prompt(self, previous, prev_thoughts):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        out = previous + '''\n \n Given the conclusions you drew from the feedback, modify your previous thoughts accordingly. We reproduce your previous thoughts below for your reference. \n''' + prev_thoughts 
        # self.logger.info(out)
        return out
    
    def gen_specific_improvement_prompt(self, previous_prompt, conclusions, num_ideas = 1):
        '''
        tells the LLM to improve upon previous thoughts
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return previous_prompt + conclusions + f'''\n \n Based on the worksheet, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the worksheet that you think will have the most impact? Be as specific and concrete as possible, including what questions to add, edit, or remove, and write them out in the following format:
        
        
        Thoughts: <your thoughts here>

        Idea 1: <your idea here>

        Idea 2: <your idea here>

        ...

        Here's an example of what this might look like for 3 improvement ideas:

        Thoughts: I should tell the reader to address each player individually in the guide.

        Idea 1: Add a question asking the reader who they think is most suspicious of them and produce a statement addressing their suspicions.

        Idea 2: Add a question asking the reader to consider which other player is most supportive of them and produce a statement addressing their support.

        Idea 3: Add a question asking the reader produce a statement addressed to the quest leader asking them to consider the reader for the quest team. 

        Recall that the goal of Merlin is to convince the Good players that they are Good without revealing that they are Merlin to the Evil players.
        '''
        
    def gen_draw_conclusions_from_feedback_prompt(self, function, feedback):
        '''
        tells the LLM to draw conclusions from feedback
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        return self.gen_feedback_analyze_prompt(function, feedback)

    def gen_implement_function_from_improvement_prompt(self, function, idea):
        '''
        tells the LLM to implement the function from the improvement
        '''
        # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
        out =  self.attacker_prompt + self.game_rules + self.PREVIOUS_GUIDE_INTRO + function + "\n \n Here is a possible way to improve your guide: \n" + idea + "\n\n Implement this improvement into the guide section as best as you can, but do not change the original guide too much. Make sure to stay within the scope of the guide section, which we reiterate below: \n" + self.function_signature
        # self.logger.info(out)
        return out

    def gen_improved_function_prompt(self, previous):
        '''
        tells the LLM to generate improved function
        '''
        # combine new_improvements_prompt and new_improvements
        out = previous +  '''\n Now modify the guide section you generated previously to improve it based on the feedback given and your thoughts, and output the new section. Do not change the original guide too much. Make sure to stay within the scope of the guide section, which we reiterate below: \n''' + self.function_signature 
        # self.logger.info(out)
        return out

    def gen_execution_error_feedback(self, previous, function, execution_error):
        raise NotImplementedError

    def gen_execution_error_feedback_2(self, error_message):
        raise NotImplementedError

    def gen_single_state_example_feedback(self, i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
        raise NotImplementedError
    
    def gen_strategy_feedback(self, i, dialogue, feedback, eval_player, thought): 
        feedback_str = self.gen_dialogue_feedback_description(feedback)
        string = f'''---
        Example {i}:
        The student using your guide was playing as player {eval_player} in this example filled out your guide as follows:\n 
        {thought}

        Then they said the following during the discussion phase:

        {dialogue}

        Below is some feedback from the other players on how player {eval_player} performed during the discussion phase. Recall that Good players are trying to access which other players are likely to be Good, while Evil players are trying to access which Good player is likely to be Merlin.

        {feedback_str}
        ---
        '''
        return string
    
    def gen_outcome_strategy_feedback(self, i, dialogue, feedback, eval_player, thought, score): 
        # feedback_str = self.gen_dialogue_feedback_description(feedback)
        string = f'''---
        Example {i}:
        The student using your guide was playing as player {eval_player} in this example filled out your guide as follows:\n 
        {thought}

        Then they said the following during the discussion phase:

        {dialogue}

        After being evaluated by the other players, player {eval_player} received a score of {score} for their performance during the discussion phase.
        ---
        '''
        return string
    
    def gen_dialogue_feedback_description(self, feedback: dict[int, tuple[str, str]]):
        feedback_strs = []
        for player, (response, role) in feedback.items():
            feedback_strs.append(f"Player {player} with role {role} had the following thoughts on this discussion round: \n {response}")
        return "\n".join(feedback_strs)