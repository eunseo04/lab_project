import argparse
import os

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from my_utils import (
    SaveBestRewardCallback,
    load_jsonl,
    setup_logger
)
from reward import (
    RewardModelHandler,
    RewardModelHandlerForRed,
    reward_func_for_eval_model,
)

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Self-RL Training Script")

    parser.add_argument("--mode", type=str, default="qa-positive", choices=["qa-positive", "red", "eval"], help="Training mode, can be qa-positive, red, or eval.")
    parser.add_argument('--iteration', type=int, default=1, help='Training iteration number, used for splitting and selecting dataset')
    parser.add_argument("--fsdp", type=lambda x: x.lower() == "true", help="Use FSDP for distributed training.", default=False)
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Use DeepSpeed for distributed training.')

    # Model related parameters
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Base model path or name')
    parser.add_argument('--qa_model_name', type=str, nargs='+', default='Qwen/Qwen2.5-0.5B-Instruct', help='QA model name')
    parser.add_argument('--eval_model_name', type=str, nargs='+', default='Qwen/Qwen2.5-0.5B-Instruct', help='Evaluation model name')

    # Data related parameters
    parser.add_argument("--prompt_fname", type=str, default="prompts/prompt_red_20250626.yaml", help="Prompt file path")
    parser.add_argument("--policy_fname", type=str, default="prompts/safety_spec_en.txt", help="Policy file path")
    parser.add_argument("--special_issue", type=str, default="prompts/special_issue_en.txt", help="Special issue file path")
    parser.add_argument("--traindataset_path", type=str, default="../self-rl/corpus/negetive_corpus.jsonl", help="Training dataset path")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty coefficient for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for generation")

    # Training related parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--use_vllm", type=lambda x: x.lower() == "true", default=True, help="Whether to use vLLM for rollout")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps, -1 means no limit")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_completion_length", type=int, default=8192, help="Maximum completion length")
    parser.add_argument('--max_prompt_length', type=int, default=1024, help='Maximum prompt length')
    parser.add_argument('--qa_max_tokens', type=int, default=4096, help="Maximum generation tokens for QA model")
    parser.add_argument('--eval_max_tokens', type=int, default=4096, help="Maximum generation tokens for evaluation model")

    # Output related parameters
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="./logs/selfrl-grpo-0731", help="Logging directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=5, help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total limit for saved models")
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument('--qa_model_url', type=str, nargs='+', default=None, help='QA model API URL, e.g., http://localhost:8000')
    parser.add_argument('--qa_model_weights', type=float, nargs='+', default=None, help='QA model weights, e.g., 0.5, 0.5')
    parser.add_argument('--eval_model_url', type=str, nargs='+', default=None, help='Evaluation model API URL, e.g., http://localhost:8000')

    parser.add_argument('--vllm_mode', type=str, default='server', choices=['server', 'local'], help='vLLM usage mode, server means using vLLM server, local means local vLLM call')
    parser.add_argument('--vllm_server_base_url', type=str, default=None, help='vLLM server base URL, e.g., http://localhost:8000')
    parser.add_argument('--vllm_server_timeout', type=int, default=60, help='Timeout for calling vLLM server in seconds')
    
    parser.add_argument('--how_to_save', type=str, default='best_reward', choices=['best_reward', 'steps'], help='Model saving method, best_reward means save model with highest reward, steps means save model by steps')
    parser.add_argument('--semantic_model_url', type=str, default=None, help='Semantic evaluation model API URL, e.g., http://localhost:1005')
    parser.add_argument('--semantic_model_name', type=str, default=None, help='Semantic evaluation model name, e.g., ../model/qwen/Qwen3-8B')
    parser.add_argument('--save_prompts_dir', type=str, default="output/red_prompts", help='Directory to save generated prompts, None means not saving')
    
    parser.add_argument('--bleu_reward_coef', type=float, default=-3.0, help='BLEU reward coefficient, default 0 means not using BLEU reward')
    parser.add_argument('--cossimemb_reward_coef', type=float, default=-3.0, help='Cosine similarity reward coefficient, default 0 means not using cosine similarity reward')
    
    parser.add_argument('--chunks', type=int, default=10, help='Number of parts to split training dataset, used for iterative dataset selection')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume training from specified checkpoint path, None means not resuming')
    parser.add_argument('--think', type=lambda x: x.lower() == "true", default=False, help="Whether to allow model to think before answering")
    
    parser.add_argument('--response_save_dir', type=str, help='Directory for red-qa to collect query-response pairs')
    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    _rk = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if _rk == 0:
        print(
            f"[train.py][entry] rank_env LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
            f"RANK={os.environ.get('RANK')} cwd={os.getcwd()} "
            f"output_dir(arg)={args.output_dir}",
            flush=True,
        )
    if args.mode == 'red' and args.response_save_dir is not None:
        response_file = os.path.join(args.response_save_dir, str(args.iteration), 'query_and_response.jsonl')
        if os.path.exists(response_file):
            # Clear existing file if it exists
            with open(response_file, 'w', encoding='utf-8') as f:
                pass

    if args.num_train_epochs < 1 and args.max_steps < 1:
        raise ValueError("Either num_train_epochs or max_steps must be set to a positive value.")
    elif args.num_train_epochs > 0 and args.max_steps > 0:
        raise ValueError("Only one of num_train_epochs or max_steps can be set to a positive value.")
    
    if args.fsdp and args.deepspeed_config is not None:
        raise ValueError("Cannot use both FSDP and DeepSpeed. Please choose one.")
    
    if len(args.qa_model_url) != len(args.qa_model_name):
        raise ValueError("Length of qa_model_url must match length of qa_model_name.")
    if len(args.eval_model_url) != len(args.eval_model_name):
        raise ValueError("Length of eval_model_url must match length of eval_model_name.")
    
    
    if args.deepspeed_config is None and not args.fsdp:
        print("Loading model locally since neither DeepSpeed nor FSDP is enabled.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype='auto',
            device_map='auto',
        )
        model.config.use_cache = False
    else:
        model = args.model_path
        
    logger = setup_logger(
        "selfrl",
        log_file=os.path.join(args.logging_dir, 'training.log'),
        to_console=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.mode == 'red':
        qa_tokeenizers = [
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) for model_name in args.qa_model_name
        ]
        for qa_tokenizer in qa_tokeenizers:
            qa_tokenizer.padding_side = "left"
            if qa_tokenizer.pad_token is None:
                qa_tokenizer.pad_token = qa_tokenizer.eos_token
    else:
        qa_tokeenizers = None
    
    if args.mode.startswith('qa'):
        eval_tokenizers = [
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) for
            model_name in args.eval_model_name
        ]
        for eval_tokenizer in eval_tokenizers:
            eval_tokenizer.padding_side = "left"
            if eval_tokenizer.pad_token is None:
                eval_tokenizer.pad_token = eval_tokenizer.eos_token
    else:
        eval_tokenizers = None
            
    traindataset = Dataset.from_json(args.traindataset_path)
    print(f"[train.py] loaded dataset size: {len(traindataset)} from {args.traindataset_path}")

    # Split traindataset into args.chunks parts, select one part based on (args.iteration - 1) % args.chunks for training
    if args.mode == 'red':
        split_size = len(traindataset) // args.chunks
        start = ((args.iteration - 1) % args.chunks) * split_size
        end = start + split_size if (args.iteration - 1) % args.chunks < args.chunks - 1 else len(traindataset)
        traindataset = traindataset.select([i for i in list(range(start, end))])
        print(
            f"[train.py] red split: chunks={args.chunks}, split_size={split_size}, "
            f"selected_range=[{start}, {end}), selected_size={len(traindataset)}"
        )

    if len(traindataset) == 0:
        raise ValueError(
            "Training dataset is empty after preprocessing/splitting. "
            "No optimization step will run, so no checkpoint can be saved. "
            "Please reduce chunks, change iteration, or verify traindataset_path."
        )

    #traindataset.shuffle(seed=int(time.time()) % 10000)

    # transformers>=4.40: TrainingArguments.fsdp must be str/list of options, not bool (DeepSpeed 时常为 False 触发 TypeError)
    fsdp_arg: str | list[str] = ["full_shard", "auto_wrap"] if args.fsdp else ""

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy='no' if args.how_to_save == 'best_reward' else 'steps',
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        fsdp=fsdp_arg,
        deepspeed=args.deepspeed_config if args.deepspeed_config is not None else None,
        mask_truncated_completions=True,
        epsilon_high=0.28,
        importance_sampling_level='sequence',
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_server_timeout=args.vllm_server_timeout,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        gradient_checkpointing=True if isinstance(model, str) else False,
        # distributed_backend='nccl',
        # world_size=torch.cuda.device_count() if args.ddp else 1,
    )

    if args.mode.startswith("qa"):
        reward_func = RewardModelHandler(
            reward_model_url=args.eval_model_url,
            prompt_file=args.prompt_fname,
            policy_file=args.policy_fname,
            special_issue=args.special_issue,
            tokenizer=tokenizer,
            reward_model_name=args.eval_model_name,
            max_tokens=args.eval_max_tokens,
            eval_tokenizers=eval_tokenizers,
            think=args.think
        )
    elif args.mode == "eval":
        reward_func = reward_func_for_eval_model

    elif args.mode == "red":
        reward_func = RewardModelHandlerForRed(
            qa_model_url=args.qa_model_url,
            eval_model_url=args.eval_model_url,
            prompt_file=args.prompt_fname,
            policy_file=args.policy_fname,
            special_issue=args.special_issue,
            tokenizer=tokenizer,
            qa_model_name=args.qa_model_name,
            eval_model_name=args.eval_model_name,
            qa_max_tokens=args.qa_max_tokens,
            eval_max_tokens=args.eval_max_tokens,
            semantic_model_url=args.semantic_model_url if args.semantic_model_url != "None" else None,
            semantic_model_name=args.semantic_model_name if args.semantic_model_name != "None" else None,
            qa_model_weights=args.qa_model_weights,
            iteration=args.iteration,
            save_prompts_dir=args.save_prompts_dir if args.save_prompts_dir != "None" else None,
            bleu_reward_coef=args.bleu_reward_coef,
            cossimemb_reward_coef=args.cossimemb_reward_coef,
            eval_tokenizers=eval_tokenizers,
            qa_tokenizers=qa_tokeenizers,
            think=args.think,
            response_save_dir=args.response_save_dir,
        )
    else:
        raise ValueError(
            f"Unsupported mode: {args.mode}"
        )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=traindataset,
        processing_class=tokenizer,
        callbacks=[SaveBestRewardCallback(top_k=args.save_total_limit, save_dir=args.output_dir, tokenizer=tokenizer)] if args.how_to_save == 'best_reward' else None,
    )
    
    if args.resume_from_checkpoint is not None:
        grpo_trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        grpo_trainer.train()
    print("Training completed and model saved.")
    _od = os.path.abspath(args.output_dir)
    _is_zero = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
    if _is_zero:
        print(
            f"[train.py][post-train] cwd={os.getcwd()} output_dir={_od} "
            f"exists={os.path.isdir(_od)} list={os.listdir(_od) if os.path.isdir(_od) else 'N/A'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
