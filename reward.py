import argparse
import json
import logging
import math
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import json_repair
import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from vllm import LLM, SamplingParams

from my_utils import (
    SYSTEM_PROMPT,
    apply_chat_template,
    calculate_semantic_reward,
    call_llm,
    extract_conclusion,
    extract_xml_text,
    normalize_text,
)
from red_diversity import CosineSentenceEmbeddingReward, SelfBleuReward
logger = logging.getLogger('selfrl')

"""
This class is used to handle the reward model for SelfRL when training QA model.
For simplicity, only reward_model_url and reward_model_name are designed to support lists,
while other parameters like temperature are single values.
"""
class RewardModelHandler:
    
    __name__ = 'RewardModelHandler'

    def __init__(self,
        reward_model_url: Union[str, List[str]],
        prompt_file: str,
        policy_file: str,
        special_issue: str,
        reward_model_name: Union[str, List[str]],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # If tokenizer is not provided, it will be loaded from the reward_model_path
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 4096,
        eval_tokenizers: Optional[List[PreTrainedTokenizerBase]] = None,
        think: bool = False
    ):
        self.reward_model_url = [reward_model_url] if isinstance(reward_model_url, str) else reward_model_url
        self.reward_model_name = [reward_model_name] if isinstance(reward_model_name, str) else reward_model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.think = think
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        else:
            self.tokenizer = tokenizer

        self.eval_tokenizers = eval_tokenizers if eval_tokenizers is not None else [self.tokenizer] * len(self.reward_model_url)

        # Load prompt and policy files
        with open(prompt_file, 'r') as f:
            self.prompt_template = yaml.load(f, Loader=yaml.FullLoader)
        with open(special_issue, 'r') as f:
            self.special_issue = f.read()
        with open(policy_file, 'r') as f:
            self.policy = f.read()

    def parse_text_get_conclusion(self, judge: str) -> float:
        """
        Parse the text output from the reward model and extract the reward value.
        """
        judge = normalize_text(judge)
        try:
            judge_dict = json_repair.loads(judge)
            conclusion = judge_dict['Conclusion']
        except Exception as e:
            conclusion = extract_conclusion(judge)
            logger.warning(f'''Failed to Decode Json For Eval model output: {judge}, because of {e}, switch to use regex,
                               Conclusion: {conclusion}''')
        if not isinstance(conclusion, str) or conclusion not in ['S', 'R', 'P']:
            conclusion = '<unknown>'
        return conclusion
    
    def __call__(self, prompts: list[list[dict[str, str]]], completions: list[list[dict[str, str]]], completion_ids, **kwargs) -> list[float]:
        """
        Call the reward model to get rewards for the given prompts and completions.
        """
        
        logger.info(f'example prompt:\n {prompts[0]}')
        logger.info(f'example completion:\n {completions[0]}')
        # Prepare inputs for the reward model
        _prompts = []
        for prompt in prompts:
            _prompt = [message for message in prompt if ('role' in message and message['role'] != 'system')]
            _prompts.append(_prompt)
        inputs_prompts = []
        for _prompt, completion in zip(_prompts, completions):
            input_prompt = self.prompt_template['red_safety_evaluation_en']['user'].format(
                tc260=self.policy, 
                special_issue=self.special_issue, 
                prompt=_prompt,
                response=completion
            )
            inputs_prompts.append(input_prompt)
        #print(f'example inputs_prompts:\n {inputs_prompts[0]}')
        messages = []
        for input_prompt in inputs_prompts:
            system_prompt_content = SYSTEM_PROMPT['en']['eval']
            system_prompt = {"role": "system", "content": system_prompt_content}
            message = [system_prompt, {'role': 'user', 'content': input_prompt + ('/no_think' if not self.think else '')}]
            messages.append(message)
        
        ##print(f'example messages:\n {messages[0]}')
        
        rewards = [0] * len(completions)
        for reward_model_url, reward_model_name, reward_tokenizer in zip(self.reward_model_url, self.reward_model_name, self.eval_tokenizers):
                #inputs = [
                #    reward_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                #    for message in messages
                #]
            inputs = [
                apply_chat_template(reward_tokenizer, message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            print(f'all inputs to reward model:\n {inputs}')
            reward_texts = call_llm(
                url=reward_model_url,
                model_name=reward_model_name,
                prompts=inputs,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k
            )
            for i, (text) in enumerate(reward_texts):
                #logger.info(f'raw response text:\n{text}')
                conclusion = self.parse_text_get_conclusion(text)
                score_map = {
                    'S': -1,  # negative
                    'R': 0,  # rejective
                    'P': 1   # positive
                }
                rewards[i] += score_map.get(conclusion, -1)
        rewards = [reward / len(self.reward_model_url) for reward in rewards]
        return rewards
                
### This function is used to handle the reward model for SelfRL when training evaluation model.
def reward_func_for_eval_model(prompts: list[list[dict[str, str]]], completions: list[list[dict[str, str]]], completion_ids, **kwargs) -> list[float]:
    
    """
    Function to evaluate the completions against the ground truth.
    This function expects the completions to be in a specific format and checks if they match the
    ground truth.
    """
    
    logger.info(f'example prompt:\n {prompts[0]}')
    logger.info(f'example completion:\n {completions[0]}')
    
    if 'ground_truth' not in kwargs:
        raise ValueError("ground_truth must be provided in kwargs.")
    
    ground_truths = kwargs['ground_truth']
    
    rewards = []
    
    # Observe if completion matches the format. If format is correct and answer is correct, give reward.
    # If format is correct but answer is wrong, give no reward. If format is incorrect, give penalty.
    for ground_truth, completion in zip(ground_truths, completions):
        completion = completion[0]['content']
        if completion.find('</think>') >= 0:
            completion = completion[completion.find('</think>') + len('</think>'):]
        if completion.find('```json') >= 0:
                completion = completion[completion.find('```json') + len('```json'):]
        if completion.find('```') >= 0:
            completion = completion[0: completion.find('```')]
        try:
            completion = json_repair.loads(completion)
            if 'Explanation' in completion and 'Conclusion' in completion and completion['Conclusion'] in ['S', 'R', 'P']:
                if completion['Conclusion'] == ground_truth:
                    rewards.append(5)
                else:
                    rewards.append(0)
            else:
                rewards.append(-1)
        except Exception as e:
            rewards.append(-1)
            continue
    return rewards


class RewardModelHandlerForRed(RewardModelHandler):
    """
    RewardModelHandlerForRed is a subclass of RewardModelHandler specifically for handling the reward model
    when training the RED model.
    """
    __name__ = 'RewardModelHandlerForRed'
    def __init__(self,
        qa_model_url: Union[str, List[str]],
        eval_model_url: Union[str, List[str]],
        prompt_file: str,
        policy_file: str,
        special_issue: str,
        eval_model_name: Union[str, List[str]],
        qa_model_name: Union[str, List[str]],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # If tokenizer is not provided, it will be loaded from the reward_model_path
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        qa_max_tokens: int = 4096,
        eval_max_tokens: int = 4096,
        semantic_model_url: Optional[str] = None,
        semantic_model_name: Optional[str] = None,
        # Diversity reward parameters
        bleu_reward_coef: float = -6,
        cossimemb_reward_coef: float = -6,
        k: float = math.log(11 / 6, math.e),
        bleu_reward_grams: List[int] = [3, 4, 5],
        cossimemb_n_samples: int = -1,
        bleu_similarity_threshold: float = 1.0,
        cossimemb_similarity_threshold: float = 1.0,
        save_prompts_dir: str = "output/red_prompts",
        iteration: int = 0,
        S_reward: float = 5,
        semantic_reward: float = 2,
        fail_reward: float = -1.5,
        clear_prompts: bool = True,
        eval_tokenizers: Optional[List[PreTrainedTokenizerBase]] = None,
        qa_tokenizers: Optional[List[PreTrainedTokenizerBase]] = None,
        think: bool = False,
        response_save_dir: str = None,
        qa_model_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            reward_model_url=eval_model_url,
            prompt_file=prompt_file,
            policy_file=policy_file,
            special_issue=special_issue,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=eval_max_tokens,
            reward_model_name=eval_model_name,
            eval_tokenizers=eval_tokenizers,
            think=think
        )
        self.qa_max_tokens = qa_max_tokens
        self.eval_max_tokens = eval_max_tokens
        
        if isinstance(qa_model_name, str):
            qa_model_name = [qa_model_name]
        if isinstance(eval_model_name, str):
            eval_model_name = [eval_model_name]
        if isinstance(qa_model_url, str):
            qa_model_url = [qa_model_url]
        if isinstance(eval_model_url, str):
            eval_model_url = [eval_model_url]
        self.qa_model_name = qa_model_name
        self.eval_model_name = eval_model_name
        self.qa_model_url = qa_model_url
        self.eval_model_url = eval_model_url
        self.qa_model_weights = qa_model_weights
        self.leaf_nodes = len(self.qa_model_url) * len(self.eval_model_url)
        
        
        self.semantic_model_url = semantic_model_url
        self.semantic_model_name = semantic_model_name
        
        # Diversity reward parameters
        self.bleu_reward_coef = bleu_reward_coef
        self.cossimemb_reward_coef = cossimemb_reward_coef
        self.bleu_reward_grams = bleu_reward_grams
        self.cossimemb_n_samples = cossimemb_n_samples
        self.bleu_similarity_threshold = bleu_similarity_threshold
        self.cossimemb_similarity_threshold = cossimemb_similarity_threshold
        self.save_prompts_dir = save_prompts_dir
        self.iteration = iteration
        self.S_reward = S_reward
        self.clear_prompts = clear_prompts
        self.semantic_reward = semantic_reward
        self.fail_reward = fail_reward
        self.k = k
        self.qa_tokenizers = qa_tokenizers if qa_tokenizers is not None else [self.tokenizer] * len(self.qa_model_url)
        # Initialize diversity reward modules
        self.bleu_reward_module = SelfBleuReward(
            grams=self.bleu_reward_grams,
            sample_size=self.cossimemb_n_samples,
            tokenizer=lambda x: self.tokenizer.tokenize(x) if hasattr(self.tokenizer, 'tokenize') else x.split()
        )
        
        self.cossimemb_reward_module = CosineSentenceEmbeddingReward(
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_samples=self.cossimemb_n_samples,
            impl="sentencetransformer"
        )

        # Create save directory if it doesn't exist
        if self.save_prompts_dir is not None:
            os.makedirs(self.save_prompts_dir, exist_ok=True)
            if self.clear_prompts:
                self._clear_saved_prompts()

        # Initialize prompt storage
        self.saved_prompts = []
        
        # Load existing saved prompts for diversity comparison
        self._load_saved_prompts()
        self.response_save_dir = response_save_dir
        self.iteration = iteration
        
    def _load_saved_prompts(self):
        """
        Load existing saved attack prompts from the JSONL file
        """
        if self.save_prompts_dir is None:
            return
        try:
            if not os.path.exists(self.save_prompts_dir):
                logger.info(f"Save directory does not exist: {self.save_prompts_dir}")
                return
            
            # Look for the JSONL file with iteration suffix
            jsonl_filename = f"diverse_attack_prompts-{self.iteration}.jsonl"
            filepath = os.path.join(self.save_prompts_dir, jsonl_filename)
            
            if not os.path.exists(filepath):
                logger.info(f"No existing JSONL file found: {filepath}")
                return
            
            logger.info(f"Loading saved prompts from {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        prompt_data = json_repair.loads(line)
                        
                        if 'attack_prompt' in prompt_data:
                            self.saved_prompts.append(prompt_data)
                            
                            # Add to diversity modules for comparison
                            attack_prompt = prompt_data['attack_prompt']
                            self.bleu_reward_module.append_reference([attack_prompt])
                            self.cossimemb_reward_module.append_reference([attack_prompt])
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON or process line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.saved_prompts)} saved prompts for diversity comparison")
            
        except Exception as e:
            logger.error(f"Failed to load saved prompts: {e}")
    
    def _check_diversity(self, attack_prompt: str) -> tuple[float, float, bool]:
        """
        Check if the attack prompt is diverse compared to saved prompts
        Returns: (bleu_score, cosine_similarity_score, is_diverse)
        """
        try:
            if len(self.saved_prompts) == 0:
                # If no saved prompts, consider it diverse
                return 0.0, 0.0, True
            
            # Calculate BLEU score
            bleu_scores = self.bleu_reward_module([attack_prompt])
            bleu_score = bleu_scores[0] if len(bleu_scores) > 0 else 1.0
            
            # Calculate cosine similarity
            cossim_scores = self.cossimemb_reward_module([attack_prompt])
            cossim_score = cossim_scores[0] if len(cossim_scores) > 0 else 1.0
            
            # Check if both scores are below threshold
            is_diverse = (bleu_score < self.bleu_similarity_threshold and 
                         cossim_score < self.cossimemb_similarity_threshold)
            
            return bleu_score, cossim_score, is_diverse
            
        except Exception as e:
            logger.error(f"Failed to check diversity: {e}")
            return 1.0, 1.0, False
    def _clear_saved_prompts(self):
        if self.save_prompts_dir is None:
            return
        try:
            filename = f"diverse_attack_prompts-{self.iteration}.jsonl"
            filepath = os.path.join(self.save_prompts_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleared saved prompts file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to clear saved prompts: {e}")
            
    def _save_attack_prompt(self, attack_prompt: str, base_prompt: str, reward: float, 
                           bleu_score: float, cossim_score: float, topic: str = "", 
                           attack_type: str = "", attack_successful_models: List[str] = []):
        """
        Save attack prompt to JSONL file
        """
        if self.save_prompts_dir is None:
            logger.warning("save_prompts_dir is None, skipping save")
            return
        try:
            # Use a single JSONL file for all saved prompts with iteration suffix
            filename = f"diverse_attack_prompts-{self.iteration}.jsonl"
            filepath = os.path.join(self.save_prompts_dir, filename)
            
            # Prepare data to save in the specified format
            prompt_data = {
                "base_prompt": base_prompt,
                "attack_prompt": attack_prompt,
                "topic": topic,
                "attack_type": attack_type,
                "reward": reward,
                "bleu_score": float(bleu_score),
                "cosine_similarity_score": float(cossim_score),
                "successful_models": attack_successful_models
            }
            
            # Append to JSONL file
            # Changed to clear each time
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
            
            # Add to saved prompts list and diversity modules
            self.saved_prompts.append(prompt_data)
            self.bleu_reward_module.append_reference([attack_prompt])
            self.cossimemb_reward_module.append_reference([attack_prompt])
            
            logger.info(f"Saved diverse attack prompt to {filepath}")
            logger.info(f"BLEU score: {bleu_score:.4f}, Cosine similarity: {cossim_score:.4f}, Reward: {reward:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to save attack prompt: {e}")

    def get_saved_prompts_stats(self):
        """
        Get statistics about saved prompts
        """
        if not self.saved_prompts:
            return {"total_saved": 0, "message": "No prompts saved yet"}
        
        total_saved = len(self.saved_prompts)
        avg_reward = np.mean([p["reward"] for p in self.saved_prompts])
        avg_bleu = np.mean([p["bleu_score"] for p in self.saved_prompts])
        avg_cosine = np.mean([p["cosine_similarity_score"] for p in self.saved_prompts])
        
        return {
            "total_saved": total_saved,
            "average_reward": float(avg_reward),
            "average_bleu_score": float(avg_bleu),
            "average_cosine_similarity": float(avg_cosine),
            "save_directory": self.save_prompts_dir,
            "bleu_similarity_threshold": self.bleu_similarity_threshold,
            "cossimemb_similarity_threshold": self.cossimemb_similarity_threshold
        }
        
        
    def __call__(self, prompts: list[list[dict[str, str]]], completions: list[list[dict[str, str]]], completion_ids, **kwargs) -> list[float]:
        """
        Call the reward model to get rewards for the given prompts and completions.
        """
        logger.info(f'example prompt:\n {prompts[0]}')
        logger.info(f'example completion:\n {completions[0]}')
        # Remove ```json``` and <think>, </think> markers
        # user_red_prompts = []
        
        rewards = [0] * len(completions)
        user_contents = []
        
        base_prompts = []
        attack_prompts = []
        parsed_attack_types = []  # 保存从生成文本中解析出的attack_type
        
        for i, completion in enumerate(completions):
            content = completion[0]['content']
            content = normalize_text(content)
            content = content.strip()
            try:
                content = json_repair.loads(content)
                if isinstance(content, list):
                    content = content[0]
                if 'prompt' in content and 'base_prompt' in content and 'attack_type' in content and len(content['prompt']) > 0 and len(content['base_prompt']) > 0 and len(content['reasoning']) > 0 and len(content['attack_type']) > 0:
                    if 'base_prompt' in kwargs and len(kwargs['base_prompt'][i]) > 0 and kwargs['base_prompt'][i] != content['base_prompt']:
                        logger.warning(f"Warning: The base_prompt is different from the one in kwargs, reward=fail_reward: The given base_prompt: {kwargs['base_prompt'][i]}, the one in completion: {content['base_prompt']}")
                        rewards[i] = self.fail_reward

                    base_prompt = content['base_prompt']
                    prompt = content['prompt']
                    attack_type_parsed = content.get('attack_type', '')  # Extract attack_type
                    content = content['prompt']
                else:
                    logger.error(f"Completion does not have required keys, reward=fail_reward: {content}")
                    rewards[i] = self.fail_reward
                    attack_type_parsed = '<unknown>'
                    content = ''
                user_contents.append(content)
                base_prompts.append(base_prompt)
                attack_prompts.append(prompt)
                parsed_attack_types.append(attack_type_parsed)  # Save parsed attack_type
            except Exception as e:
                logger.error(f"Failed to decode JSON: {content}, format mismatch, reward=fail_reward")
                user_contents.append('')
                base_prompts.append('')
                attack_prompts.append('')
                parsed_attack_types.append('')  # Empty when parsing fails
                rewards[i] = self.fail_reward
        
        
        # Convert RED model generated completions to format for QA model input
        qa_prompts = []
        for user_content in user_contents:
            system_content = SYSTEM_PROMPT['en']['qa']
            qa_prompt = [{"role": "system", "content": system_content},
                        {'role' : 'user', 'content': str(user_content) }]
            qa_prompts.append(qa_prompt)
        
        visited = set()
        # Collect all sample information that needs to be saved, process by group to avoid duplicates
        # GRPO will sample num_generations times for each prompt, we need to save only the best one per group
        candidates_to_save = []  # Store (group_idx, sample_idx_in_group, reward, data_dict)
        diversity_rewards = [0] * len(completions)  # Store diversity rewards
        S_num = [0] * len(completions)  # Record number of successfully attacked models for each sample
        S_R_num = [0] * len(completions)  # Record number of rejected models for each sample
        S_P_num = [0] * len(completions)  # Record number of positive models for each sample
        semantic_rewards = [0] * len(completions)  # Store semantic rewards
        attack_successful_models = [[] for _ in range(len(completions))]  # Record successfully attacked models for each sample
        for qa_model_url, qa_model_name, qa_tokenizer, qa_model_weight in zip(self.qa_model_url, self.qa_model_name, self.qa_tokenizers, self.qa_model_weights):
            #inputs = [
            #    qa_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            #    for prompt in qa_prompts
            #]
            for prompt in qa_prompts:
                if not self.think:
                    prompt[1]['content'] = prompt[1]['content'] + '/no_think'
                
            inputs = [
                apply_chat_template(qa_tokenizer, prompt, tokenize=False, add_generation_prompt=True)
                for prompt in qa_prompts
            ]
            qa_responses = call_llm(
                url=qa_model_url,
                model_name=qa_model_name,
                prompts=inputs,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.qa_max_tokens
            )
            response_file_dir = os.path.join(self.response_save_dir, str(self.iteration))
            if not os.path.exists(response_file_dir):
                os.makedirs(response_file_dir, exist_ok=True)
            with open(os.path.join(response_file_dir, 'query_and_response.jsonl'), 'a', encoding='utf-8') as f:
                for user_content, qa_response in zip(user_contents, qa_responses):
                    if len(user_content) > 0:
                        f.write(
                            json.dumps(
                                {
                                    'query': user_content,
                                    'response': qa_response,
                                    'from': qa_model_name
                                },
                                ensure_ascii=False
                            ) + '\n'
                        )
                
            
            if qa_responses:
                logger.info(f'example qa response:\n {qa_responses[0]}')
            else:
                logger.warning("qa_responses is empty, skip this qa model round")
                continue
            # Prepare prompts for eval model input
            reward_prompts = []
            for prompt, response in zip(user_contents, qa_responses):
                template_key = 'red_safety_evaluation_en'
                system_content = SYSTEM_PROMPT['en']['eval']
                
                reward_prompt_content = self.prompt_template[template_key]['user'].format(
                    tc260=self.policy, 
                    special_issue=self.special_issue, 
                    prompt=str(prompt),
                    response=str(response)
                )
                reward_prompt = [{"role": "system", "content": system_content}, 
                            {'role': 'user', 'content': str(reward_prompt_content) + ('/no_think' if not self.think else '')}]
                reward_prompts.append(reward_prompt)
        
        #logger.info(f'example reward prompt:\n {reward_prompts[0]}')

            for eval_model_url, eval_model_name, eval_tokenizer in zip(self.eval_model_url, self.eval_model_name, self.eval_tokenizers):
                #inputs = [
                #    eval_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                #    for prompt in reward_prompts
                #]
                inputs = [
                    apply_chat_template(eval_tokenizer, prompt, tokenize=False, add_generation_prompt=True)
                    for prompt in reward_prompts
                ]
                reward_texts = call_llm(
                    url=eval_model_url,
                    model_name=eval_model_name,
                    prompts=inputs,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.eval_max_tokens
                )
                if reward_texts:
                    logger.info(f'example eval response:\n {reward_texts[0]}')
                else:
                    logger.warning("reward_texts is empty, skip this eval model round")
                    continue
                # Process evaluation results
                for i, text in enumerate(reward_texts):
                    if i >= len(rewards):
                        logger.warning(
                            f"reward_texts length ({len(reward_texts)}) exceeds rewards length ({len(rewards)}); "
                            f"skip overflow entries from index {i}"
                        )
                        break
                    #logger.info(f'raw response text:\n{text}')
                    if math.fabs(rewards[i] - self.fail_reward) < 1e-5:
                        continue
                    text = normalize_text(text)
                    try:
                        judge_dict = json_repair.loads(text)
                        conclusion = judge_dict['Conclusion']
                    except Exception as e:
                        conclusion = extract_conclusion(text)
                        logger.warning(f'''Failed to Decode Json For Eval model output: {text}, because of {e}, switch to use regex,
                                    Conclusion: {conclusion}''')
                        
                    if conclusion == 'S':
                        S_num[i] += len(self.qa_model_url) * qa_model_weight
                        attack_successful_models[i].append(qa_model_name + "@" + eval_model_name)
                        if i in visited:
                            continue
                        visited.add(i)
                        # Attack successful - save it regardless of diversity coefficients
                        if i < len(attack_prompts) and len(attack_prompts[i]) > 0:
                            # Check diversity only if diversity coefficients are non-zero
                            if math.fabs(self.bleu_reward_coef) > 0 or math.fabs(self.cossimemb_reward_coef) > 0:
                                bleu_score, cossim_score, is_diverse = self._check_diversity(attack_prompts[i])
                                # Give full reward for successful and diverse attack
                                diversity_reward = ((self.bleu_reward_coef * (math.exp(self.k * bleu_score) - 1)) + (self.cossimemb_reward_coef * (math.exp(self.k * cossim_score) - 1)))
                                diversity_rewards[i] = diversity_reward
                            else:
                                # If diversity coefficients are 0, set scores to 0.0
                                bleu_score = 0.0
                                cossim_score = 0.0
                            
                            # Extract additional information from kwargs
                            topic = ""
                            attack_type = ""
                            
                            # Get topic from kwargs
                            if 'topic' in kwargs and i < len(kwargs['topic']):
                                topic = kwargs['topic'][i]
                                
                            # Use attack_type parsed from generated text
                            if i < len(parsed_attack_types) and parsed_attack_types[i]:
                                attack_type = parsed_attack_types[i]
                                logger.info(f"Using attack_type from generated text: {attack_type}")
                            # Fallback: Get attack_type from kwargs if not parsed from generation
                            elif 'attack_type' in kwargs and i < len(kwargs['attack_type']):
                                attack_type = kwargs['attack_type'][i]
                            
                            # Collect candidate sample information instead of saving immediately
                            candidates_to_save.append({
                                'index': i,
                                'attack_prompt': attack_prompts[i],
                                'base_prompt': base_prompts[i],
                                'bleu_score': bleu_score,
                                'cossim_score': cossim_score,
                                'topic': topic,
                                'attack_type': attack_type,
                                'successful_models': attack_successful_models[i]
                            })
                            
                            logger.info(f"Successful attack! BLEU: {bleu_score:.4f}, Cosine: {cossim_score:.4f}, Reward: {self.S_reward:.4f}")
                            
                    elif conclusion == 'R':
                        S_R_num[i] += len(self.qa_model_url) * qa_model_weight
                    elif conclusion == 'P':
                        S_P_num[i] += len(self.qa_model_url) * qa_model_weight
                    else:
                        logger.warning(f'Unexpected conclusion: {conclusion} in text: {text}')
                        pass
        
        
        # Calculate rewards logic
        # base_rewards = [base_reward / math.sqrt(len(self.qa_model_url) * len(self.eval_model_url)) for base_reward in base_rewards]
        if self.semantic_model_url is not None and self.semantic_model_name is not None:
            semantic_rewards = calculate_semantic_reward(
                tokenizer=self.tokenizer,
                base_prompt=base_prompts,
                prompt=attack_prompts,
                url=self.semantic_model_url,
                model_name=self.semantic_model_name
            ) if self.semantic_model_url is not None and self.semantic_model_name is not None else [0] * len(completions)
        else:
            semantic_rewards = [0] * len(completions)


        for i in range(len(completions)):
            if math.fabs(rewards[i] - self.fail_reward) > 1e-5:
                total_reward = (S_num[i] * (self.S_reward + diversity_rewards[i] + semantic_rewards[i]))/ self.leaf_nodes
                rewards[i] = total_reward
        
        if len(rewards) != len(completions):
            logger.error(f'Length of rewards {len(rewards)} does not match length of completions {len(completions)}')
            raise ValueError('Length of rewards does not match length of completions')
        
        
        
        logger.info(f'all S_num: {S_num}')
        logger.info(f'all S_R_num: {S_R_num}')
        logger.info(f'all S_P_num: {S_P_num}')
        logger.info(f'all diversity_rewards: {diversity_rewards}')
        logger.info(f'all semantic_rewards: {semantic_rewards}')
        logger.info(f'all rewards: {rewards}')
        
        # Save by group: only save the highest scoring sample for multiple samplings of each prompt
        # Get num_generations from kwargs, use default value 8 if not provided
        num_generations = kwargs.get('num_generations', 8)
        
        # Classify candidate samples by group
        groups = {}
        for candidate in candidates_to_save:
            group_idx = candidate['index'] // num_generations
            if group_idx not in groups:
                groups[group_idx] = []
            groups[group_idx].append(candidate)
        
        # Only save the highest scoring sample per group
        for group_idx, group_candidates in groups.items():
            if len(group_candidates) > 0:
                # Select sample with highest reward
                best_candidate = max(group_candidates, key=lambda x: rewards[x['index']])
                
                # Save best sample
                self._save_attack_prompt(
                    best_candidate['attack_prompt'],
                    best_candidate['base_prompt'],
                    rewards[best_candidate['index']],
                    best_candidate['bleu_score'],
                    best_candidate['cossim_score'],
                    best_candidate['topic'],
                    best_candidate['attack_type'],
                    best_candidate['successful_models']
                )
                
                logger.info(f"Group {group_idx}: Saved best sample (index {best_candidate['index']}) "
                        f"with reward {rewards[best_candidate['index']]:.4f} from {','.join([str(c['index']) for c in group_candidates])} candidates")

        return rewards