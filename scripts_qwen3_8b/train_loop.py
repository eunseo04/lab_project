import json
import os
import subprocess
import sys

import json_repair
import torch

from run import training_mode

# Configuration flags
skip_for_vllm = False
kill_vllm = True
start_mode = 'red'
modes = ['red', 'qa-positive', 'eval']

# Training steps configuration for each mode
steps_config = {
    'red': 1,
    'qa-positive': 1,
    'eval': 1
}

# Parse command line arguments
begin_iters = int(sys.argv[1])
max_iters = int(sys.argv[2])

# Save steps configuration for each mode
save_steps = {'red': 1, 'qa-positive': 1, 'eval': 1}

# Initial model paths for each mode
last_model_path = {
    'red': 'models/qwen/Qwen3-8B',
    'qa-positive': 'models/qwen/Qwen3-8B',
    'eval': 'models/qwen/Qwen3-8B'
}

# Centralized model path overrides (sparse style):
# - Only add keys you really want to override.
# - Missing keys keep the existing dynamic/default behavior.
model_paths_config = {
    'eval': {
        # Example: force fixed eval reward judge model in eval mode
        'reward.eval': ['models/qwen/Qwen3-14B'],
    }
}

# Per-device training batch size configuration
per_device_train_batch_size_config = {
    'red': 1,
    'qa-positive': 1,
    'eval': 1
}

# Manually define port configuration for each mode
# Format: {mode: {'gen_data.red': port, 'gen_data.qa': port, 'rollout': port, 
#                 'reward.qa': [port1, port2, ...], 'reward.eval': [port1, ...], 
#                 'reward.semantic': port}}
ports_config = {
    'red': {
        'gen_data.red': 1000,
        'gen_data.qa': 1001,
        'rollout': 1004,
        'reward.qa': [1002, 1006, 1007],  # If multiple QAs exist, fill in order
        'reward.eval': [1009],
        'reward.semantic': 1005
    },
    'qa-positive': {
        'gen_data.red': 2000,
        'gen_data.qa': 2001,
        'rollout': 2004,
        'reward.qa': [2002],
        'reward.eval': [2003]
    },
    'eval': {
        'gen_data.red': 9000,
        'gen_data.qa': 9001,
        'rollout': 9004,
        'reward.qa': [9002],
        'reward.eval': [9003]
    }
}


def set_ports(config, mode):
    """
    Set all port configurations in the config file based on ports_config.
    
    Args:
        config: Configuration dictionary to modify
        mode: Training mode ('red', 'qa-positive', or 'eval')
    """
    if mode not in ports_config:
        return
    
    port_cfg = ports_config[mode]
    
    # Set gen_data.red.port
    if 'gen_data' in config and 'red' in config['gen_data'] and 'gen_data.red' in port_cfg:
        config['gen_data']['red']['port'] = port_cfg['gen_data.red']
    
    # Set gen_data.qa.port
    if 'gen_data' in config and 'qa' in config['gen_data'] and 'gen_data.qa' in port_cfg:
        config['gen_data']['qa']['port'] = port_cfg['gen_data.qa']
    
    # Set rollout.port
    if 'rollout' in config and 'rollout' in port_cfg:
        config['rollout']['port'] = port_cfg['rollout']
    
    # Set reward.qa[].port
    if 'reward' in config and 'qa' in config['reward'] and 'reward.qa' in port_cfg:
        if isinstance(config['reward']['qa'], dict):
            config['reward']['qa'] = [config['reward']['qa']]
        qa_ports = port_cfg['reward.qa']
        for i, qa_cfg in enumerate(config['reward']['qa']):
            if i < len(qa_ports):
                qa_cfg['port'] = qa_ports[i]
    
    # Set reward.eval[].port
    if 'reward' in config and 'eval' in config['reward'] and 'reward.eval' in port_cfg:
        if isinstance(config['reward']['eval'], dict):
            config['reward']['eval'] = [config['reward']['eval']]
        eval_ports = port_cfg['reward.eval']
        for i, eval_cfg in enumerate(config['reward']['eval']):
            if i < len(eval_ports):
                eval_cfg['port'] = eval_ports[i]
    
    # Set reward.semantic.port (if exists)
    if 'reward' in config and 'semantic' in config['reward'] and 'reward.semantic' in port_cfg:
        config['reward']['semantic']['port'] = port_cfg['reward.semantic']


def apply_model_path_overrides(config, mode):
    """Apply sparse model-path overrides from model_paths_config."""
    mode_cfg = model_paths_config.get(mode, {})

    # Dynamic defaults
    config['general']['model_path'] = mode_cfg.get('general.model_path', last_model_path[mode])
    config['gen_data']['red']['model_path'] = mode_cfg.get('gen_data.red.model_path', last_model_path['red'])
    if mode == 'eval':
        config['gen_data']['qa']['model_path'] = mode_cfg.get('gen_data.qa.model_path', last_model_path['qa-positive'])
    elif 'gen_data.qa.model_path' in mode_cfg:
        config['gen_data']['qa']['model_path'] = mode_cfg['gen_data.qa.model_path']

    # reward.qa
    if isinstance(config['reward']['qa'], dict):
        config['reward']['qa'] = [config['reward']['qa']]
    qa_overrides = mode_cfg.get('reward.qa', [])
    for i, qa_model_cfg in enumerate(config['reward']['qa']):
        if i < len(qa_overrides):
            qa_model_cfg['model_path'] = qa_overrides[i]
        elif qa_model_cfg.get('is_training', False):
            qa_model_cfg['model_path'] = last_model_path['qa-positive']

    # reward.eval
    if isinstance(config['reward']['eval'], dict):
        config['reward']['eval'] = [config['reward']['eval']]
    eval_overrides = mode_cfg.get('reward.eval', [])
    for i, eval_model_cfg in enumerate(config['reward']['eval']):
        if i < len(eval_overrides):
            eval_model_cfg['model_path'] = eval_overrides[i]
        elif eval_model_cfg.get('is_training', False):
            eval_model_cfg['model_path'] = last_model_path['eval']

    # reward.semantic (optional)
    if 'semantic' in config['reward'] and 'reward.semantic.model_path' in mode_cfg:
        config['reward']['semantic']['model_path'] = mode_cfg['reward.semantic.model_path']


# Main training loop
total_gpus = torch.cuda.device_count()
for iteration in range(begin_iters, max_iters + 1):
    # Determine starting mode index
    if iteration == begin_iters:
        index = modes.index(start_mode)
    else:
        index = 0
    
    # Iterate through all modes
    while index < len(modes):
        mode = modes[index]
        config_file = os.path.join(os.path.dirname(__file__), 'configs', mode, 'config.json')
        
        # Load and update configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json_repair.load(f)
        
        print(config)
        
        # Set all ports uniformly
        set_ports(config, mode)
        
        # Update training configuration
        # Ensure at least one checkpoint is produced for the current run.
        # If max_steps is smaller than save_steps (e.g. debug with max_steps=1),
        # no checkpoint-* directory would be created and downstream vLLM loading fails.
        config['train']['save_steps'] = min(save_steps[mode], steps_config[mode])
        config['general']['iteration'] = iteration
        config['train']['per_device_train_batch_size'] = per_device_train_batch_size_config[mode]
        config['train']['max_steps'] = steps_config[mode]
        apply_model_path_overrides(config, mode)
        
        # Set directory paths (relative to project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config['general']['save_prompts_dir'] = os.path.join(project_root, 'output', 'red_prompts_qwen3_8b')
        config['general']['response_save_dir'] = os.path.join(project_root, 'output', 'response_qwen3_8b')
        config['train']['output_dir'] = 'checkpoints_qwen3_8b'
        config['general']['traindataset_path'] = os.path.join(project_root, 'prompts', 'red_corpus.jsonl')
        
        # Set semantic reward GPU configuration
        if 'semantic' in config['reward']:
            config['reward']['semantic']['gpus'] = 1
        
        # Set reward coefficients
        config['reward']['bleu_reward_coef'] = -2.0
        config['reward']['cossimemb_reward_coef'] = -4.0
        
        # Set training hyperparameters
        config['train']['learning_rate'] = 1e-6
        config['train']['logging_dir'] = f'selfrl_logs_qwen3_8b'
        
        # Set training dataset path for eval mode
        if mode == 'eval':
            config['general']['traindataset_path'] = os.path.join(
                config['general']['response_save_dir'], 
                str(iteration), 
                'query_and_response.jsonl'
            )
        
        # Save updated configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        # Set CUDA visible devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(total_gpus)])
        
        print(f'Iteration {iteration} - Mode: {mode} - Model: {last_model_path[mode]} - max_steps: {steps_config[mode]}')
        
        # Run training for current mode
        model_path = training_mode(
            mode, 
            skip_for_vllm=skip_for_vllm, 
            kill_vllm=kill_vllm, 
            iteration=iteration
        )
        last_model_path[mode] = model_path
        index += 1