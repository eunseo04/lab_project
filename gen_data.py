import argparse
import gc
import json
import logging
import os
import re
import time
from functools import partial
from typing import Any, Dict, List, Optional, Union

import json_repair
import torch
import yaml
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM

from my_utils import (
    SYSTEM_PROMPT,
    call_llm,
    load_jsonl,
    normalize_text
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training data using language models.")
    parser.add_argument("--mode", type=str, choices=["eval", "red", "qa-positive"], required=True, help="Training mode: qa, eval, or red")
    parser.add_argument("--traindataset_path", type=str, default=None, help="Path to the training dataset (JSONL format)")
    parser.add_argument("--use_existing_dataset_for_training", type=str, default="", help="Path to an existing dataset to use for training/evaluation")
    parser.add_argument("--red_model_name", type=str, default="../model/qwen/Qwen3-8B", help="Name or path of the red model")
    parser.add_argument("--qa_model_name", type=str, default="../model/qwen/Qwen3-8B", help="Name or path of the QA model")
    parser.add_argument("--red_model_url", type=str, help="url to the red model ")
    parser.add_argument("--qa_model_url", type=str, help="url to the QA model ")
    parser.add_argument("--tokenizer_path", type=str, default="../model/qwen/Qwen3-8B", help="Path to the tokenizer model")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    parser.add_argument("--save_dir", type=str, default="./output", help="Path to save the final dataset")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for data preprocessing")
    parser.add_argument("--prompt_fname", type=str, default="prompts/prompt_red_20250626.yaml", help="Path to the prompt file")
    parser.add_argument("--policy_fname", type=str, default="prompts/safety_spec.txt", help="Path to the policy file")
    parser.add_argument("--special_issue", type=str, default="prompts/special_issue.txt", help="Path to the special issue file")
    parser.add_argument('--think', action='store_true', help='Whether to include think process in the output')
    parser.add_argument('--chunks', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--skip_red_when_generating_eval', action='store_true', help='Whether to skip red generation when generating eval data')
    return parser.parse_args()

def generate_risky_prompts(
    tokenizer: PreTrainedTokenizer,
    args: Optional[argparse.Namespace] = None,
    prompts: Optional[List[Dict[str, str]]] = None,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Generate risky prompt dataset using the model.
    
    Args:
        tokenizer: Corresponding tokenizer
        args: Command line arguments
        prompts: Optional list of prompts to use instead of loading from file
        batch_size: Batch size for processing
    
    Returns:
        List of generated risky prompt datasets
        Example: {'prompt':[{'role':'system', 'content':...}, {'role':'user', 'content':}], 'ground_truth':...}
    """
    import random
    from vllm import SamplingParams
    
    # Load dataset
    if args is not None:
        dataset_path = args.traindataset_path
        prompts_data = load_jsonl(dataset_path)
        split_size = len(prompts_data) // args.chunks
        start = ((args.iteration - 1) % args.chunks) * split_size
        end = start + split_size if (args.iteration - 1) % args.chunks < args.chunks - 1 else len(prompts_data)
        prompts_data = prompts_data[start:end]
    elif prompts is not None:
        prompts_data = prompts
    else:
        raise ValueError("No valid input data provided")
    
    generated_data = []
    
    # Get number of generations per prompt
    # num_generations_per_prompt = getattr(args, 'num_generations_per_prompt', 1)
    
    for i in tqdm(range(0, len(prompts_data), batch_size), desc="Generating risky prompts"):
        batch_prompts = prompts_data[i:i+batch_size]
        
        # Build input prompts
        inputs = []
        for prompt_data in batch_prompts:
            # Support both:
            # 1) {"prompt": [{"role":"system"...}, {"role":"user","content":"..."}], ...}
            # 2) {"query": "...", "response": "...", ...}  (e.g. eval intermediate file)
            user_content = None
            prompt_list = prompt_data.get("prompt")
            if isinstance(prompt_list, list):
                for item in prompt_list:
                    if isinstance(item, dict) and item.get("role") == "user":
                        user_content = item.get("content")
                        if user_content:
                            break
            if not user_content:
                user_content = prompt_data.get("query")
            if not user_content:
                # Skip malformed sample instead of crashing the full pipeline.
                continue

            system_prompt_dict = SYSTEM_PROMPT["en"]
            input_prompt = [
                {"role": "system", "content": system_prompt_dict["red"]},
                {"role": "user", "content": user_content + ("" if args.think else "/no_think")},
            ]
            inputs.append(input_prompt)

        if not inputs:
            continue
        
        inputs = [
            tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True) 
            for input in inputs
        ]
        
        # Generate using LLM
        outputs = call_llm(
            model_name=args.red_model_name,
            url=args.red_model_url,
            prompts=inputs
        )
        
        for j, output in enumerate(outputs):
            # Extract generated text
            generated_contents = output.strip()
            generated_contents = normalize_text(generated_contents)
            
            try:
                generated_contents = json_repair.loads(generated_contents)
                if isinstance(generated_contents, dict) and 'prompt' in generated_contents:
                    generated_contents = [generated_contents['prompt']]
                elif isinstance(generated_contents, list):
                    generated_contents = [item['prompt'] for item in generated_contents if 'prompt' in item]
                else:
                    generated_contents = [generated_contents]
            except Exception as e:
                generated_contents = []
            
            # Build data items
            for generated_content in generated_contents:
                system_prompt_dict = SYSTEM_PROMPT["en"]

                if generated_content is not None and generated_content != "":
                    data_item = {
                        'prompt': [
                            {'role': 'system', 'content': system_prompt_dict['qa']},
                            {'role': 'user', 'content': generated_content}
                        ]
                    }
                    generated_data.append(data_item)
    
    return generated_data

def generate_answers_from_risky_prompts(
    tokenizer: PreTrainedTokenizer,
    args: argparse.Namespace,
    prompts: List[Dict[str, Any]],
    batch_size: int = 32,
) -> List[Dict[str, str]]:
    """
    Generate answers from risky prompts using the model.
    
    Note: The prompts here use the positive response template.
    
    Args:
        tokenizer: Corresponding tokenizer
        args: Command line arguments
        prompts: List of risky prompts, format: [{'prompt':[{'role':'system', 'content':...}, {'role':'user', 'content':}], 'ground_truth':...}]
        batch_size: Batch size for processing
    
    Returns:
        List of generated answers
        Example: {'prompt': 'original prompt user content', 'response': 'generated answer', 'ground_truth': 'correct answer'}
    """
    from vllm import SamplingParams
    
    generated_data = []
    mode = args.mode
    
    # Get number of answer generations per prompt
    # num_generations_per_prompt = getattr(args, 'num_generations_per_prompt_for_answers', 1)
    
    # Load prompt template and policies
    with open(args.prompt_fname, 'r') as f:
        prompt_template = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_fname, 'r', encoding='utf-8') as f:
        policy = f.read()
    with open(args.special_issue, 'r', encoding='utf-8') as f:
        special_issue_content = f.read()
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating answers from risky prompts"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Build input prompts
        inputs = []
        user_contents = []
        
        for prompt_data in batch_prompts:
            prompt_list = prompt_data.get('prompt')
            ground_truth = prompt_data.get('ground_truth')
            
            if not prompt_list:
                continue

            system_prompt_dict = SYSTEM_PROMPT["en"]

            # Extract user content
            user_content = ""
            for item in prompt_list:
                if item.get('role') == 'user':
                    user_content = item.get('content', '')
                    break
            
            # Use positive response template for QA generation
            template_key = 'red_safety_012_response_positive'
            
            qa_prompt = prompt_template[template_key]['user'].format(
                tc260=policy,
                user_prompt=user_content
            )
            
            system_prompt = system_prompt_dict['qa']
            
            # Build conversation format
            full_prompt = [
                {"role": "system", "content": system_prompt},
                {'role': 'user', 'content': qa_prompt + ('' if args.think else '/no_think')}
            ]
            
            # Format using apply_chat_template
            formatted_prompt = tokenizer.apply_chat_template(
                full_prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs.append(formatted_prompt)
            user_contents.append(user_content)
        
        # Generate answers using LLM
        outputs = call_llm(
            model_name=args.qa_model_name,
            url=args.qa_model_url,
            prompts=inputs
        )
        
        # Process generation results
        for j, output in enumerate(outputs):
            try:
                generated_content = output.strip()
                print(f"Raw generated content: {generated_content}")
                
                # Build final data format
                generated_content = normalize_text(generated_content)
                generated_content = json_repair.loads(generated_content)
                negative_content = generated_content.get('negative_response')
                positive_content = generated_content.get('positive_response')
                reject_content = generated_content.get('rejective_response')
                
                # Use evaluation template
                template_key = 'red_safety_evaluation_en'
                system_prompt = SYSTEM_PROMPT["en"]["eval"]
                
                # Build evaluation prompts
                if 'tc260' in prompt_template[template_key]['user'] and 'special_issue' in prompt_template[template_key]['user']:
                    eval_negative_prompt = prompt_template[template_key]['user'].format(
                        tc260=policy,
                        special_issue=special_issue_content,
                        prompt=user_contents[j],
                        response=negative_content,
                    )
                    eval_positive_prompt = prompt_template[template_key]['user'].format(
                        tc260=policy,
                        special_issue=special_issue_content,
                        prompt=user_contents[j],
                        response=positive_content,
                    )
                    eval_reject_prompt = prompt_template[template_key]['user'].format(
                        tc260=policy,
                        special_issue=special_issue_content,
                        prompt=user_contents[j],
                        response=reject_content,
                    )
                else:
                    eval_negative_prompt = prompt_template[template_key]['user'].format(
                        prompt=user_contents[j],
                        response=negative_content,
                    )
                    eval_positive_prompt = prompt_template[template_key]['user'].format(
                        prompt=user_contents[j],
                        response=positive_content,
                    )
                    eval_reject_prompt = prompt_template[template_key]['user'].format(
                        prompt=user_contents[j],
                        response=reject_content,
                    )
                
                # Build data items: S (evil), P (helpful), R (safe)
                data_item = [
                    {
                        'prompt': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': eval_negative_prompt + ('/no_think' if not args.think else '')}
                        ],
                        'ground_truth': 'S'  # S corresponds to evil
                    },
                    {
                        'prompt': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': eval_positive_prompt + ('/no_think' if not args.think else '')}
                        ],
                        'ground_truth': 'P'  # P corresponds to helpful
                    },
                    {
                        'prompt': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': eval_reject_prompt + ('/no_think' if not args.think else '')}
                        ],
                        'ground_truth': 'R'  # R corresponds to safe
                    }
                ]
                
                generated_data.extend(data_item)
            except Exception as e:
                continue
    
    return generated_data

def preprocess_function_qa(example, args):
    example["prompt"][-1]["content"] = (
        example["prompt"][-1]["content"] + ("/no_think" if not args.think else "")
    )
    return example

def preprocess_function_eval(example, args):
    """Preprocess evaluation examples."""
    with open(args.prompt_fname, "r") as f:
        prompt_template = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_fname, "r", encoding="utf-8") as f:
        policy = f.read()
    with open(args.special_issue, "r", encoding="utf-8") as f:
        special_issue = f.read()
    
    prompt = example["prompt"]
    response = example["response"]
    
    # Use evaluation template
    template_key = 'red_safety_evaluation_en'
    system_prompt = SYSTEM_PROMPT["en"]["eval"]

    full = prompt_template[template_key]["user"].format(
        tc260=policy,
        special_issue=special_issue,
        prompt=prompt,
        response=response,
    )
    
    example["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full + ("/no_think" if not args.think else "")},
    ]
    return example


def preprocess_function_red(example, args):
    """Preprocess red examples."""
    content = example["prompt"][-1]["content"]

    system_prompt = SYSTEM_PROMPT["en"]["red"]
    example["prompt"][-1]["content"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content + ('' if args.think else '/no_think')},
    ]
    
    return example

def gen_qa_data(args):
    print("Training mode: Question-Answering")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    prompt_template = yaml.load(open(args.prompt_fname, 'r'), Loader=yaml.FullLoader)
    with open(args.policy_fname, "r", encoding="utf-8") as f:
        policy = f.read()

    if args.use_existing_dataset_for_training == "":
        temp_data_path = args.save_dir
        if not os.path.exists(temp_data_path):
            os.makedirs(temp_data_path, exist_ok=True)
        risky_prompts = generate_risky_prompts(
            tokenizer=tokenizer,
            args=args,
        )
        # Apply positive response template to all prompts
        for risk_prompt in risky_prompts:
            risk_prompt['prompt'][1]['content'] = prompt_template['red_safety_012_response_positive']['user'].format(
                tc260=policy,
                user_prompt=risk_prompt['prompt'][1]['content'],
            ) + ('/no_think' if not args.think else '')
        with open(os.path.join(temp_data_path, "risky_prompts_positive.json"), "w", encoding="utf-8",) as f:
            json.dump(risky_prompts, f, ensure_ascii=False, indent=4)
        print(
            f"Generated {len(risky_prompts)} risky prompts. "
            f"example: {risky_prompts[0] if risky_prompts else 'N/A (empty)'}"
        )
        print(f'Generating risky prompts have been saved to {os.path.join(temp_data_path, "risky_prompts.json")}')
    else:
        print(
            f"Using existing dataset for training evaluation: {args.use_existing_dataset_for_training}"
        )
        risky_prompts = load_jsonl(args.use_existing_dataset_for_training)
    traindataset = Dataset.from_list(risky_prompts)
    traindataset = traindataset.map(partial(preprocess_function_qa, args=args), num_proc=args.num_proc)
    return traindataset

def gen_eval_data(args):
    print("Training mode: Evaluation")
    prompt_template = yaml.load(open(args.prompt_fname, 'r'), Loader=yaml.FullLoader)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.use_existing_dataset_for_training == "":

        temp_data_path = args.save_dir

        if not os.path.exists(temp_data_path):
            os.makedirs(temp_data_path, exist_ok=True)
        if not args.skip_red_when_generating_eval:
            risky_prompts = generate_risky_prompts(
                tokenizer=tokenizer,
                args=args,
            )
        else:
            print("Skipping RED generation as per the flag --skip_red_when_generating_eval")
            risky_prompts = load_jsonl(args.traindataset_path)
        # Note: For eval mode, prompts are used as-is without template wrapping
        # The prompts are already formatted in the risky_prompts
        with open(
            os.path.join(temp_data_path, f"risky_prompts.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(risky_prompts, f, ensure_ascii=False, indent=4)

        print(
            f"Generated {len(risky_prompts)} risky prompts. "
            f"example: {risky_prompts[0] if risky_prompts else 'N/A (empty)'}"
        )

        prompts_and_answers = generate_answers_from_risky_prompts(
            tokenizer=tokenizer,
            prompts=risky_prompts,
            args=args,
        )

        with open(
            os.path.join(temp_data_path, "prompts_and_answers.json"),
            "w", encoding="utf-8",) as f:
            json.dump(prompts_and_answers, f, ensure_ascii=False, indent=4)

        print(
            f"Generated {len(prompts_and_answers)} prompts and answers. "
            f"example: {prompts_and_answers[0] if prompts_and_answers else 'N/A (empty)'}"
        )

    else:
        print(f"Using existing dataset for training evaluation: {args.use_existing_dataset_for_training}")
        prompts_and_answers = load_jsonl(args.use_existing_dataset_for_training)

    # Convert generated prompts and answers to Dataset format
    traindataset = Dataset.from_list(prompts_and_answers)
    # from functools import partial

    # traindataset = traindataset.map(
    #     partial(preprocess_function_eval, args=args),
    #     num_proc=args.num_proc
    # )
    return traindataset

def gen_red_data(args):
    print("Training mode: RED")
    traindataset = Dataset.from_json(args.traindataset_path)
    traindataset = traindataset.shuffle(seed=args.seed)
    if len(traindataset) > 0:
        print(traindataset[0])
    else:
        print("RED dataset is empty.")
    traindataset = traindataset.map(partial(preprocess_function_red, args=args), num_proc=args.num_proc)
    return traindataset

def main():
    args = parse_args()
    mode = args.mode
    if mode == "qa-positive":
        gen_qa_data(args)
    elif mode == "eval":
        gen_eval_data(args)
    elif mode == "red":
        gen_red_data(args)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    print(f'Data have been saved to {args.save_dir} Directory.')


if __name__ == "__main__":
    main()
