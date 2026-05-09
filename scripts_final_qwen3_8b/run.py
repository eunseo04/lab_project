import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import torch
import yaml

# Get project root directory (parent of scripts_final_qwen3_8b)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deploy_vllm_server import start_vllm_server, start_trl_vllm_server, stop_vllm_server

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
total_gpus = torch.cuda.device_count()


def _list_checkpoints(output_dir: str) -> list[str]:
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
    )


def _abs_project_path(path: str) -> str:
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(project_root, p)
    return os.path.normpath(os.path.abspath(p))


def _pipeline_debug(phase: str, **info) -> None:
    """统一前缀，便于 grep: grep '\\[pipeline\\]' log"""
    parts = [f"[pipeline][{phase}]"]
    for k, v in info.items():
        parts.append(f"{k}={v!r}")
    print(" ".join(parts), flush=True)


def _hf_dir_ok(path: str) -> bool:
    ap = _abs_project_path(path)
    return os.path.isfile(os.path.join(ap, "config.json"))


def _guess_recent_run_dirs(train_output_root: str, mode: str, limit: int = 8) -> list[str]:
    """output_dir 缺失时，在 checkpoints 根下搜同类 run 目录便于人工对照"""
    base = _abs_project_path(train_output_root)
    pat = os.path.join(base, "*", mode, f"{mode}-*")
    cands = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return cands[:limit]


def allocate_gpu_for_model(used_gpu_mems: list[float], model_cfg: dict):
    """
    Allocate GPUs for a model using a greedy allocation strategy.
    
    Args:
        used_gpu_mems: List of GPU memory usage ratios (0.0-1.0) for each GPU
        model_cfg: Model configuration dictionary containing 'gpus' and 'gpu_mem' keys
    
    Returns:
        List of allocated GPU indices
    
    Raises:
        RuntimeError: If not enough GPU memory is available
    """
    gpus = model_cfg.get('gpus', 1)
    gpu_mem = model_cfg.get('gpu_mem', 0.9)
    total_gpus = len(used_gpu_mems)
    allocated_gpus = []
    
    for i in range(total_gpus):
        if len(allocated_gpus) < gpus:
            if used_gpu_mems[i] + gpu_mem <= 0.9:
                allocated_gpus.append(i)
                used_gpu_mems[i] += gpu_mem
    
    if len(allocated_gpus) < gpus:
        raise RuntimeError(f"Not enough GPU memory to allocate for model {model_cfg['model_path']}")
    
    model_cfg['allocated_gpus'] = ','.join(str(i) for i in allocated_gpus)
    return allocated_gpus


def auto_allocate_gpus(mode: str, total_gpus: int, config: dict):
    """
    Automatically allocate GPUs for different components based on training mode.
    Uses a greedy allocation strategy: only add a new GPU when accumulated gpu_mem exceeds 1.0.
    Remaining GPUs are used for training.
    
    Args:
        mode: Training mode ('red', 'qa-positive', or 'eval')
        total_gpus: Total number of available GPUs
        config: Configuration dictionary to modify
    
    Returns:
        Modified configuration dictionary with allocated GPU information
    """
    # used_gpu_mems represents the memory usage ratio for each GPU, initially all 0
    used_gpu_mems = [0.0] * total_gpus
    
    if mode == 'red':
        # For red mode: no gen_data section needed.
        # Reward section needs to deploy qa_model_list, eval_model, and semantic_model.
        # Rollout section needs to deploy red_model.
        for section in ['reward', 'rollout']:
            if section == 'rollout':
                allocate_gpu_for_model(used_gpu_mems, config[section])
            else:
                for key in ['qa', 'eval']:
                    cfgs = config[section][key]
                    if isinstance(cfgs, dict):
                        cfgs = [cfgs]
                    for model_cfg in cfgs:
                        # gpus: number of GPUs needed, gpu_mem: memory ratio per GPU
                        allocate_gpu_for_model(used_gpu_mems, model_cfg)
                allocate_gpu_for_model(used_gpu_mems, config[section]['semantic'])
    
    elif mode.startswith('qa'):
        # For qa phase: gen_data section needs to deploy red_model.
        # After gen_data, red_model memory will be released.
        # Reward section needs to deploy eval_model.
        # Rollout section needs to deploy qa_model.
        for section in ['gen_data', 'reward', 'rollout']:
            if section == 'gen_data':
                allocate_gpu_for_model(used_gpu_mems, config[section]['red'])
                # Release red_model memory after gen_data
                used_gpu_mems = [0.0] * total_gpus
            elif section == 'reward':
                cfgs = config[section]['eval']
                if isinstance(cfgs, dict):
                    cfgs = [cfgs]
                for model_cfg in cfgs:
                    allocate_gpu_for_model(used_gpu_mems, model_cfg)
            else:
                allocate_gpu_for_model(used_gpu_mems, config[section])
    
    else:
        # For eval phase: gen_data section needs to deploy red_model and qa_model.
        # After gen_data, red_model and qa_model memory will be released.
        # Reward section needs to deploy eval_model.
        # Rollout section needs to deploy eval_model.
        for section in ['gen_data', 'rollout']:
            if section == 'gen_data':
                # In gen_data section, red and qa models each take half of the GPUs
                config[section]['red']['allocated_gpus'] = ','.join(
                    str(i) for i in range(total_gpus // 2)
                )
                config[section]['red']['gpus'] = total_gpus // 2
                config[section]['qa']['allocated_gpus'] = ','.join(
                    str(i) for i in range(total_gpus // 2, total_gpus)
                )
                config[section]['qa']['gpus'] = total_gpus - total_gpus // 2
                used_gpu_mems = [0.0] * total_gpus
            else:
                allocate_gpu_for_model(used_gpu_mems, config[section])
    
    # Set GPUs for training section
    config['train']['allocated_gpus'] = ','.join(
        str(i) for i in range(total_gpus) if used_gpu_mems[i] < 0.2
    )
    return config

def training_mode(mode: str, skip_for_vllm: bool = False, kill_vllm: bool = True, iteration: int = 1):
    """
    Execute training pipeline for a specific mode.
    
    Args:
        mode: Training mode ('red', 'qa-positive', or 'eval')
        skip_for_vllm: Whether to skip vLLM server startup
        kill_vllm: Whether to kill vLLM servers after training
        iteration: Current iteration number
    
    Returns:
        Path to the last checkpoint directory
    """
    suffix = f'{mode}-{timestamp}'
    config_file = os.path.join(os.path.dirname(__file__), 'configs', mode, 'config.json')
    
    # Load and update configuration
    with open(config_file, "r") as f:
        config = json.load(f)
    
    auto_allocate_gpus(mode, total_gpus, config)
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    
    last_ckpt = None
    env = os.environ.copy()
    red_vllm_process = None
    rollout_vllm_process = None
    semantic_vllm_process = None
    save_dir = None
    eval_vllm_processes: list = []
    qa_vllm_processes: list = []

    _pipeline_debug(
        "training_mode_start",
        mode=mode,
        iteration_arg=iteration,
        config_iteration=config.get("general", {}).get("iteration"),
        suffix=suffix,
        project_root=project_root,
        cwd=os.getcwd(),
    )

    try:
        if not skip_for_vllm:
            # ==================== Generate Data ====================
            if mode == 'red' or config['gen_data']['use_existing_dataset_for_training'] != "":
                pass
            else:
                try:
                    save_dir = os.path.join(config['gen_data']['save_dir'], suffix)
                    env['CUDA_VISIBLE_DEVICES'] = config['gen_data']['red']['allocated_gpus']
                    
                    if mode != 'eval' or not config['gen_data'].get('skip_red_when_generating_eval', False):
                        _red_ckpt = config['gen_data']['red']['model_path']
                        _pipeline_debug(
                            "gen_data_red_vllm_model_path",
                            raw=_red_ckpt,
                            abs_path=_abs_project_path(_red_ckpt),
                            hf_config_json=_hf_dir_ok(_red_ckpt),
                        )
                        if not _hf_dir_ok(_red_ckpt):
                            raise FileNotFoundError(
                                f"gen_data 红队 vLLM 模型路径不是有效 HF 目录（缺少 config.json）: "
                                f"{_abs_project_path(_red_ckpt)}"
                            )
                        red_vllm_process = start_vllm_server(
                            model_path=_red_ckpt,
                            host=config['gen_data']['red']['url'],
                            port=config['gen_data']['red']['port'],
                            tp_size=config['gen_data']['red']['gpus'],
                            max_model_len=config['gen_data']['red']['max_model_len'],
                            gpu_mem=config['gen_data']['red']['gpu_mem'],
                            env=env
                        )
                        
                        gen_data_cmd = [
                            'python', os.path.join(project_root, 'gen_data.py'),
                            '--mode', mode,
                            '--traindataset_path', config['general']['traindataset_path'],
                            '--red_model_url', f"http://{config['gen_data']['red']['url']}:{config['gen_data']['red']['port']}",
                            '--red_model_name', config['gen_data']['red']['model_path'],
                            '--qa_model_url', f"http://{config['gen_data']['qa']['url']}:{config['gen_data']['qa']['port']}" if mode == 'eval' else "None",
                            '--qa_model_name', config['gen_data']['qa']['model_path'] if mode == 'eval' else "None",
                            '--tokenizer_path', config['general']['model_path'],
                            '--seed', str(config['general']['seed']),
                            '--save_dir', save_dir,
                            '--num_proc', str(4),
                            '--prompt_fname', config['general']['prompt_fname'],
                            '--policy_fname', config['general']['policy_fname'],
                            '--special_issue', config['general']['special_issue'],
                            '--chunks', str(config['general'].get('chunks', 1)),
                            '--iteration', str(config['general']['iteration'])
                        ]
                        
                        if config['general'].get('think', False):
                            gen_data_cmd += ['--think']
                        if config['gen_data'].get('skip_red_when_generating_eval', False):
                            gen_data_cmd += ['--skip_red_when_generating_eval']
                        
                        subprocess.run(gen_data_cmd, check=True, cwd=project_root)
                    
                    if mode == 'eval':
                        file_name = os.path.join(
                            config['general']['response_save_dir'], 
                            str(iteration), 
                            'query_and_response.jsonl'
                        )
                        # Use eval.py directly without environment switching
                        eval_cmd = ['python', os.path.join(project_root, 'eval.py'), '--file_name', file_name]
                        subprocess.run(eval_cmd, check=True, cwd=project_root)
                
                finally:
                    if kill_vllm and red_vllm_process is not None:
                        if mode != 'eval' or not config['gen_data'].get('skip_red_when_generating_eval', False):
                            stop_vllm_server(red_vllm_process)

            # ==================== Reward Models ====================
            if mode == 'eval':
                pass
            else:
                # Start eval model servers
                for cfg in config['reward']['eval']:
                    if cfg['gpus'] == 0:
                        continue
                    env['CUDA_VISIBLE_DEVICES'] = cfg['allocated_gpus']
                    eval_vllm_process = start_vllm_server(
                        model_path=cfg['model_path'],
                        host=cfg['url'],
                        port=cfg['port'],
                        tp_size=cfg['gpus'],
                        max_model_len=cfg['max_model_len'],
                        gpu_mem=cfg['gpu_mem'],
                        env=env
                    )
                    eval_vllm_processes.append(eval_vllm_process)
                
                # Start QA model servers for red mode
                if mode == 'red':
                    for cfg in config['reward']['qa']:
                        if cfg['gpus'] == 0:
                            continue
                        env['CUDA_VISIBLE_DEVICES'] = cfg['allocated_gpus']
                        qa_vllm_process = start_vllm_server(
                            model_path=cfg['model_path'],
                            host=cfg['url'],
                            port=cfg['port'],
                            tp_size=cfg['gpus'],
                            max_model_len=cfg['max_model_len'],
                            gpu_mem=cfg['gpu_mem'],
                            env=env
                        )
                        qa_vllm_processes.append(qa_vllm_process)
                    
                    # Start semantic model server for red mode
                    if config['reward']['semantic']['gpus'] > 0:
                        env['CUDA_VISIBLE_DEVICES'] = config['reward']['semantic']['allocated_gpus']
                        semantic_vllm_process = start_vllm_server(
                            model_path=config['reward']['semantic']['model_path'],
                            host=config['reward']['semantic']['url'],
                            port=config['reward']['semantic']['port'],
                            tp_size=config['reward']['semantic']['gpus'],
                            max_model_len=config['reward']['semantic']['max_model_len'],
                            gpu_mem=config['reward']['semantic']['gpu_mem'],
                            env=env
                        )

            # ==================== Rollout Model ====================
            if config['rollout']['use_vllm']:
                env['CUDA_VISIBLE_DEVICES'] = config['rollout']['allocated_gpus']
                rollout_vllm_process = start_trl_vllm_server(
                    model_path=config['general']['model_path'],
                    host=config['rollout']['url'],
                    port=config['rollout']['port'],
                    tp_size=config['rollout']['gpus'],
                    max_model_len=config['rollout']['max_model_len'],
                    gpu_mem=config['rollout']['gpu_mem'],
                    env=env
                )

        # ==================== Training ====================
        env['CUDA_VISIBLE_DEVICES'] = config['train']['allocated_gpus']
        
        # Determine training dataset path based on mode
        if mode == 'red':
            traindataset_path = config['general']['traindataset_path']
            qa_model_weights = [
                str(cfg['weight']) for cfg in config['reward']['qa'] if cfg['gpus'] > 0
            ]
        elif mode.startswith('qa'):
            if config['gen_data']['use_existing_dataset_for_training'] != "":
                traindataset_path = config['gen_data']['use_existing_dataset_for_training']
            else:
                if save_dir is None:
                    raise RuntimeError(
                        "qa 模式需要 gen_data 产物 save_dir，但 gen_data 被跳过；"
                        "不能构造 risky_prompts_positive.json 路径。"
                    )
                traindataset_path = os.path.join(save_dir, 'risky_prompts_positive.json')
        else:
            traindataset_path = os.path.join(
                config['general']['response_save_dir'], 
                str(iteration), 
                'query_and_response_labelled.jsonl'
            )

        output_dir = os.path.join(
            config['train']['output_dir'], 
            str(config['general']['iteration']), 
            mode, 
            suffix
        )
        output_dir_abs = _abs_project_path(output_dir)
        logging_dir_abs = _abs_project_path(
            os.path.join(
                config['train']['logging_dir'],
                str(config['general']['iteration']),
                mode,
                suffix,
            )
        )
        traindataset_path = _abs_project_path(traindataset_path)

        print(f"[train_mode={mode}] output_dir(rel)={output_dir}")
        print(f"[train_mode={mode}] output_dir(abs)={output_dir_abs}")
        _pipeline_debug(
            "pre_train_paths",
            traindataset_path=traindataset_path,
            traindataset_exists=os.path.isfile(traindataset_path),
            rollout_model_path=config["general"]["model_path"],
            rollout_hf_ok=_hf_dir_ok(config["general"]["model_path"]),
        )
        if not os.path.isfile(traindataset_path):
            raise FileNotFoundError(
                f"训练数据不存在，不会启动 train 子进程: {traindataset_path}"
            )

        # Prepare model URLs and names
        qa_model_url = [
            f'http://{cfg["url"]}:{cfg["port"]}' 
            for cfg in config['reward']['qa'] if cfg['gpus'] > 0
        ]
        eval_model_url = [
            f'http://{cfg["url"]}:{cfg["port"]}' 
            for cfg in config['reward']['eval'] if cfg['gpus'] > 0
        ]
        qa_model_name = [
            cfg['model_path'] for cfg in config['reward']['qa'] if cfg['gpus'] > 0
        ]
        eval_model_name = [
            cfg['model_path'] for cfg in config['reward']['eval'] if cfg['gpus'] > 0
        ]
        qa_max_tokens = str(max(cfg.get('max_tokens', 4096) for cfg in config['reward']['qa']))
        eval_max_tokens = str(max(cfg.get('max_tokens', 4096) for cfg in config['reward']['eval']))
        
        # Build training command arguments
        args_cmd = [
            '--mode', mode,
            '--iteration', str(config['general']['iteration']),
            '--model_path', config['general']['model_path'],
            '--traindataset_path', traindataset_path,
            '--per_device_train_batch_size', str(config['train']['per_device_train_batch_size']),
            '--gradient_accumulation_steps', str(config['train']['gradient_accumulation_steps']),
            '--learning_rate', str(config['train']['learning_rate']),
            '--num_train_epochs', str(config['train']['num_train_epochs']),
            '--max_steps', str(config['train']['max_steps']),
            '--warmup_steps', str(config['train']['warmup_steps']),
            '--weight_decay', str(config['train']['weight_decay']),
            '--max_completion_length', str(config['train']['max_completion_length']),
            '--max_prompt_length', str(config['train']['max_prompt_length']),
            '--logging_steps', str(config['train']['logging_steps']),
            '--save_steps', str(config['train']['save_steps']),
            '--save_total_limit', str(config['train']['save_total_limit']),
            '--output_dir', output_dir_abs,
            '--logging_dir', logging_dir_abs,
            '--seed', str(config['general']['seed']),
            '--prompt_fname', config['general']['prompt_fname'],
            '--policy_fname', config['general']['policy_fname'],
            '--special_issue', config['general']['special_issue'],
            '--qa_max_tokens', qa_max_tokens,
            '--eval_max_tokens', eval_max_tokens,
            '--qa_model_url', *qa_model_url,
            '--eval_model_url', *eval_model_url,
            '--qa_model_name', *qa_model_name,
            '--eval_model_name', *eval_model_name,
            '--how_to_save', config['train']['how_to_save'],
            '--use_vllm', str(config['rollout']['use_vllm']).lower(),
            '--vllm_mode', "server",
            '--vllm_server_base_url', f"http://{config['rollout']['url']}:{config['rollout']['port']}",
            '--vllm_server_timeout', str(config['rollout']['vllm_server_timeout']),
            '--repetition_penalty', str(config['general'].get('repetition_penalty', 1.0)),
            '--temperature', str(config['general'].get('temperature', 1.0)),
            '--semantic_model_url', f"http://{config['reward']['semantic']['url']}:{config['reward']['semantic']['port']}" if mode == 'red' and config['reward']['semantic']['gpus'] > 0 else "None",
            '--semantic_model_name', config['reward']['semantic']['model_path'] if mode == 'red' and config['reward']['semantic']['gpus'] > 0 else "None",
            '--save_prompts_dir', config['general'].get('save_prompts_dir'),
            '--bleu_reward_coef', str(config['reward'].get('bleu_reward_coef', 0.0)),
            '--cossimemb_reward_coef', str(config['reward'].get('cossimemb_reward_coef', 0.0)),
            '--chunks', str(config['general'].get('chunks', 1)),
        ]
        
        if mode == 'red':
            args_cmd += ['--response_save_dir', config['general']['response_save_dir']]
            args_cmd += ['--qa_model_weights', *qa_model_weights]
        
        # Build training command based on distributed training configuration
        train_script = os.path.join(project_root, 'train.py')
        if config['train']['fsdp']:
            fsdp_config_file = os.path.join(os.path.dirname(__file__), mode, 'fsdp_config.yaml')
            cmd = [
                'accelerate', 'launch',
                '--fsdp_config', fsdp_config_file,
                train_script,
            ] + args_cmd + ['--fsdp', 'true']
        elif config['train']['deepspeed']:
            deepspeed_config_file = os.path.join(
                os.path.dirname(__file__), 
                'configs', 
                mode, 
                'deepspeed_config.json'
            )
            cmd = [
                'deepspeed',
                train_script,
            ] + args_cmd + ['--deepspeed_config', deepspeed_config_file]
        else:
            cmd = ['python', train_script] + args_cmd

        _pipeline_debug(
            "launch_train_subprocess",
            cmd_head=" ".join(cmd[:6]) + " ...",
            cwd=project_root,
            cuda_visible=env.get("CUDA_VISIBLE_DEVICES"),
        )
        proc = subprocess.run(cmd, check=True, env=env, cwd=project_root)
        _pipeline_debug("train_subprocess_returned", returncode=getattr(proc, "returncode", 0))
        print(f"[train_mode={mode}] train subprocess finished.")
        out_ok = os.path.isdir(output_dir_abs)
        print(f"[train_mode={mode}] output_dir(abs) exists: {out_ok} path={output_dir_abs}")
        print(f"[train_mode={mode}] checkpoints: {_list_checkpoints(output_dir_abs)}")
        if not out_ok:
            guess = _guess_recent_run_dirs(config["train"]["output_dir"], mode)
            _pipeline_debug(
                "output_dir_missing_after_train",
                hint="若未见 [train.py][entry] 日志，多半是 train.py 未执行 main()（缺少 __main__ 入口）",
                recent_same_mode_dirs=guess,
            )
            raise RuntimeError(
                f"训练子进程已返回，但 output_dir 不存在: {output_dir_abs}。"
                f" 请确认 train.py 末尾有 if __name__ == '__main__': main()，并查看子进程日志。"
            )

        # Find and return the last checkpoint
        if os.path.exists(output_dir_abs):
            checkpoints = [
                os.path.join(output_dir_abs, d)
                for d in os.listdir(output_dir_abs)
                if os.path.isdir(os.path.join(output_dir_abs, d)) and d.startswith("checkpoint-")
            ]
            if checkpoints:
                last_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            else:
                last_ckpt = output_dir_abs
        else:
            last_ckpt = output_dir_abs

    finally:
        # Clean up vLLM servers
        if kill_vllm:
            if mode != 'eval':
                for eval_vllm_process in eval_vllm_processes:
                    stop_vllm_server(eval_vllm_process)
                if mode == 'red':
                    for qa_vllm_process in qa_vllm_processes:
                        stop_vllm_server(qa_vllm_process)
            if config['rollout']['use_vllm'] and rollout_vllm_process is not None:
                stop_vllm_server(rollout_vllm_process)
            if mode == 'red' and config['reward']['semantic']['gpus'] > 0 and semantic_vllm_process is not None:
                stop_vllm_server(semantic_vllm_process)

    return last_ckpt
