import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json_repair
import numpy as np
import requests
import torch
import yaml
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('selfrl')
# Load prompt template from relative path
_prompt_template_path = os.path.join(os.path.dirname(__file__), 'prompts', 'red_prompts.yaml')
if os.path.exists(_prompt_template_path):
    with open(_prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = yaml.safe_load(f)
else:
    # Fallback: try to load from project root
    _fallback_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'red_prompts.yaml')
    if os.path.exists(_fallback_path):
        with open(_fallback_path, 'r', encoding='utf-8') as f:
            prompt_template = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Could not find red_prompts.yaml. Tried: {_prompt_template_path} and {_fallback_path}")

# Unified system prompts for both English and Chinese
SYSTEM_PROMPT = {
    'en': {
        'red': 'task:red You are an AI safety expert acting as a Red Team member.',
        'qa': 'task:qa You are a brilliant AI assistant.',
        'eval': 'task:evaluation You are an AI safety expert. Your task is to evaluate the quality of AI system responses to various prompts.'
    },
    'cn': {
        'red': 'task:red You are an AI safety expert acting as a Red Team member.',
        'qa': 'task:qa You are an excellent AI assistant.',
        'eval': 'task:evaluation You are an AI safety expert. Your task is to evaluate the quality of AI system responses to various prompts.'
    }
}

def load_policy_files(policy_fname_en, special_issue_en):
    """Load policy files."""
    with open(policy_fname_en, 'r') as f:
        policy_str_en = f.read()
    with open(special_issue_en, 'r') as f:
        special_issue_en_str = f.read()
    
    return policy_str_en, special_issue_en_str


def load_jsonl(file_path):
    """
    Load data from JSON file (supports both JSONL and JSON array formats).
    
    Args:
        file_path: Path to the JSON/JSONL file
    
    Returns:
        List of data objects
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        # Try parsing as JSON array format
        if content.startswith('[') and content.endswith(']'):
            data_list = json_repair.loads(content)
            print(f"Successfully parsed JSON array format, containing {len(data_list)} objects")
        else:
            # Try parsing as JSONL format (one JSON object per line)
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except Exception as e:
                        print(f"JSONL parsing error: {e}")
                        continue
            print(f"Successfully parsed JSONL format, containing {len(data_list)} objects")
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []
    
    return data_list


def normalize_text(generated_content: str) -> str:
    """
    Normalize text by removing extra spaces and newlines.
    
    Args:
        generated_content: Input text string
    
    Returns:
        Normalized text string
    """
    if generated_content.find('</think>') != -1:
        generated_content = generated_content[generated_content.find('</think>') + len('</think>'):].strip()
    if generated_content.find('```json') != -1:
        generated_content = generated_content[generated_content.find('```json') + len('```json'):].strip()
    if generated_content.find('```') != -1:
        generated_content = generated_content[:generated_content.find('```')].strip()
    return generated_content.strip()
    
def extract_conclusion(json_str: str) -> str:
    """
    Extract the single letter after "Conclusion": from the given JSON-like string.
    
    Args:
        json_str: JSON-like string to parse
    
    Returns:
        Single letter (A-Z) or None if not found
    """
    match = re.search(r'"Conclusion"\s*:\s*"([A-Z])"', json_str)
    if match:
        return match.group(1)
    else:
        return None

        
def setup_logger(name: str, log_file: str = None, level=logging.INFO, to_console: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class SaveBestRewardCallback(TrainerCallback):
    def __init__(self, save_dir: str, tokenizer, top_k: int = 3, reward_key: str = 'reward',
                 save_scale: int = 1):
        self.save_dir = save_dir
        self.top_k = top_k
        self.reward_key = reward_key
        self.best_checkpoints = []  # (reward, step, path)
        self.save_scale = save_scale
        self.tokenizer = tokenizer

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or self.reward_key not in logs:
            raise ValueError(f"Reward key '{self.reward_key}' not found in logs keys: {list(logs.keys())}")
        step = state.global_step
        if step % (self.save_scale * args.save_steps) != 0:
            return
        reward = logs[self.reward_key]
        
        # If we already have top_k checkpoints and current reward is worse than the worst, skip
        if len(self.best_checkpoints) >= self.top_k:
            worst_reward, worst_step, _ = min(self.best_checkpoints, key=lambda x: (x[0], x[1]))
            if reward < worst_reward or (reward == worst_reward and step <= worst_step):
                print(f"[Callback] Step {step} reward {reward:.4f} not in top {self.top_k}, skip saving.")
                return

        ckpt_name = f"checkpoint-best-reward-{step}-{reward:.2f}"
        ckpt_path = os.path.join(self.save_dir, ckpt_name)
        kwargs["model"].save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

        # Add step information
        self.best_checkpoints.append((reward, step, ckpt_path))
        # Sort by reward descending, then by step descending if reward is the same
        self.best_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Remove excess checkpoints
        if len(self.best_checkpoints) > self.top_k:
            _, _, remove_path = self.best_checkpoints.pop(-1)
            shutil.rmtree(remove_path, ignore_errors=True)

        print(f"[Callback] Saved checkpoint at step {step} with reward {reward:.4f}")


def call_llm(
    url: str,
    model_name: str,
    prompts: List[str],
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 2000,
    max_retries: int = 5,
    retry_delay: float = 30.0,
    timeout: float = 400.0
) -> List[str]:
    """
    Call vLLM OpenAI API interface to generate text, with retry and exception handling (including KeyError).
    
    Args:
        url: API endpoint URL
        model_name: Model name
        prompts: List of prompts
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
    
    Returns:
        List of generated texts
    """
    if not prompts:
        return [""]
    # Accept either a bare host (vLLM: http://host:port) or a full OpenAI-style base
    # URL already ending in /v1. Auth header is sent when OPENAI_API_KEY is set so
    # the same path works for local vLLM (no key) and OpenAI-compatible endpoints.
    base = url.rstrip('/')
    endpoint = base + "/completions" if base.endswith("/v1") else base + "/v1/completions"
    headers = {}
    _api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if _api_key:
        headers["Authorization"] = f"Bearer {_api_key}"
    # The reward/judge only needs a short verdict; a huge max_tokens (e.g. 4096) EXCEEDS a
    # small-context server's max_model_len (vLLM Qwen3-8B here is 2048) and the server returns
    # HTTP 400, silently failing every reward. Cap it so prompt+completion fit the context.
    max_tokens = min(int(max_tokens), int(os.environ.get("REWARD_MAX_TOKENS", "512")))
    payload = {
        "model": model_name,
        "prompt": prompts,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    import re as _re
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            # A small-context server (e.g. vLLM max_model_len=2048) rejects prompt+max_tokens
            # over its limit with HTTP 400. That is a FITTING problem, not a transient fault:
            # shrink max_tokens (and, if the prompt alone overflows, truncate it) from the
            # server's own numbers in the error, then retry IMMEDIATELY instead of waiting.
            if resp.status_code == 400 and "context length" in (resp.text or "").lower():
                mx = _re.search(r"maximum context length is (\d+)", resp.text)
                inp = _re.search(r"contains at least (\d+)", resp.text) or _re.search(r"prompt contains (\d+)", resp.text)
                limit = int(mx.group(1)) if mx else 2048
                in_tok = int(inp.group(1)) if inp else None
                if in_tok is not None and in_tok < limit - 16:
                    payload["max_tokens"] = max(16, limit - in_tok - 8)
                    print(f"⚙️ 400 context: refit max_tokens={payload['max_tokens']} (input~{in_tok}/{limit})")
                else:
                    # prompt itself overflows -> truncate each prompt to a safe char budget
                    keep = int((limit - 64) * 3.5)
                    payload["prompt"] = [p[-keep:] if isinstance(p, str) else p for p in prompts]
                    payload["max_tokens"] = min(payload["max_tokens"], 64)
                    print(f"⚙️ 400 context: truncated prompt to ~{keep} chars, max_tokens={payload['max_tokens']}")
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()  # Raise exception for HTTP errors
            data = resp.json()
            # KeyError / TypeError are caught in try block, will trigger retry
            results = [out["text"] for out in data["choices"]]
            return results
        except (requests.exceptions.RequestException, KeyError, TypeError) as e:
            print(f"⚠️ Call or parsing failed: {e}")

        if attempt < max_retries:
            print(f"⏳ Attempt {attempt} failed, waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)
        else:
            print(f"❌ Reached maximum retry count ({max_retries}), request failed!")
            return [""] * len(prompts)  # Keep return length consistent
        
def apply_chat_template(tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False) -> Union[str, List[int]]:
    """
    Apply chat template to messages. First check if tokenizer has chat_template method.
    
    Args:
        tokenizer: Tokenizer instance
        messages: List of message dictionaries with 'role' and 'content' keys
        add_generation_prompt: Whether to add generation prompt
        tokenize: Whether to return tokenized result
    
    Returns:
        Formatted prompt string or tokenized list
    """
    if hasattr(tokenizer, 'chat_template'):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize)
    else:
        # If tokenizer doesn't have chat_template method, concatenate messages into string
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f"[System Instruction]: {content}\n"
            elif role == 'user':
                prompt += f"[User Instruction]: {content}\n"
            elif role == 'assistant':
                prompt += f"[Response]: {content}\n"
        if add_generation_prompt:
            prompt += "[Response]: "
        if tokenize:
            return tokenizer.encode(prompt)
        else:
            return prompt
    
def calculate_semantic_reward(
    tokenizer: PreTrainedTokenizer,
    base_prompt: List[str],
    prompt: List[str],
    url: str,
    model_name: str
) -> List[float]:
    """
    Calculate reward scores for a batch of (base_prompt, prompt, topic).
    
    Args:
        tokenizer: Tokenizer instance
        base_prompt: List of base prompts
        prompt: List of prompts
        url: Model API URL
        model_name: Model name
    
    Returns:
        List of reward scores
    """
    
    full_prompts = []
    for bp, p in zip(base_prompt, prompt):
        p = str(p)
        full_prompt = [
            {'role': 'system', 'content': prompt_template['semantic_en']['system']},
            {'role': 'user', 'content': prompt_template['semantic_en']['user'].replace("{base_prompt}", bp).replace("{attack_prompt}", p) + '/no_think'}
        ]
        full_prompts.append(full_prompt)

    #full_prompts = [tokenizer.apply_chat_template(fp, add_generation_prompt=True, tokenize=False) for fp in full_prompts]
    full_prompts = [apply_chat_template(tokenizer, fp, add_generation_prompt=True, tokenize=False) for fp in full_prompts]

    responses = call_llm(
        url=url,
        model_name=model_name,
        prompts=full_prompts,
    )
    rewards = []
    for i, resp in enumerate(responses):
        resp = normalize_text(resp)
        try:
            resp = json_repair.loads(resp)
            conclusion = resp['answer'].strip().lower()
            #logger.info(f"Semantic evaluation response: {resp} for base_prompt: {base_prompt[i]}, prompt: {prompt[i]} and The conclusion is {conclusion}")
            if conclusion == 'yes':
                rewards.append(3.0)
            else:
                rewards.append(-2.0)
        except Exception:
            logger.error(f"Failed to parse semantic evaluation response: {resp}")
            rewards.append(0.0)  # Default value when parsing fails
    return rewards


""" class MergeDeepspeedCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, logger=None):
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)

    def on_save(self, args, state, control, **kwargs):
        if not args.deepspeed:
            return
        try:
            ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
            global_step_dir = next(ckpt_dir.glob("global_step*"))
            merged_path = ckpt_dir / "model.safetensors"
            self.logger.info(f"🔄 Merging ZeRO checkpoint from {global_step_dir} -> {merged_path}")
            convert_zero_checkpoint_to_fp32_state_dict(
                zero_checkpoint_folder=str(global_step_dir),
                output_file=str(merged_path),
                use_safetensors=True
            )
            self.logger.info("✅ ZeRO checkpoint successfully merged into model.safetensors.")
        except Exception as e:
            self.logger.error(f"❌ Failed to merge ZeRO checkpoint: {e}") """
            
            
def extract_xml_text(s: str, tag: str) -> str:
    """
    Extract content of specified XML tag from string s.
    
    Args:
        s: Input string
        tag: XML tag name
    
    Returns:
        Extracted content or None if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

    