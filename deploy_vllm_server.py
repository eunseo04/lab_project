#!/usr/bin/env python3
import argparse
import atexit
import os
import signal
import subprocess
import sys
import time

import psutil
import requests
debug = False
def start_vllm_server(model_path, host, port, tp_size, max_model_len, gpu_mem, env, log_dir="debug"):
    """
    Start vLLM server (runs in background), write logs to file and detect keywords.
    
    Args:
        model_path: Path to the model
        host: Host address
        port: Port number
        tp_size: Tensor parallel size
        max_model_len: Maximum model length
        gpu_mem: GPU memory utilization
        env: Environment variables
        log_dir: Directory for log files
    
    Returns:
        Process object
    """
    if debug:
        return
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vllm_{host}_{port}.log")

    # Open log file for writing
    log_file = open(log_path, "w", encoding="utf-8")

    process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", host,
            "--trust-remote-code",
            "--port", str(port),
            "--tensor_parallel_size", str(tp_size),
            "--max_model_len", str(max_model_len),
            "--gpu_memory_utilization", str(gpu_mem),
        ],
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,
        env=env
    )

    print(f"🚀 vLLM server is starting (pid={process.pid}), logs written to {log_path}")
    
    # Keyword detection
    ready_keywords = [
        "Application startup complete.",
    ]
    timeout = 300  # Maximum wait time: 5 minutes
    start_time = time.time()
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    print(line, end="")  # Also print key logs to terminal
                    if any(keyword in line for keyword in ready_keywords):
                        print("✅ vLLM server is ready!")
                        return process
                else:
                    time.sleep(0.5)  # Wait for new log writes

                if time.time() - start_time > timeout:
                    print("❌ Timeout waiting, vLLM server startup failed")
                    return process
    finally:
        return process


def start_trl_vllm_server(model_path, host, port, tp_size, max_model_len, gpu_mem, env, log_dir="debug"):
    """
    Start trl vLLM server (runs in background), write logs to file and detect keywords.
    
    Args:
        model_path: Path to the model
        host: Host address
        port: Port number
        tp_size: Tensor parallel size
        max_model_len: Maximum model length
        gpu_mem: GPU memory utilization
        env: Environment variables
        log_dir: Directory for log files
    
    Returns:
        Process object
    """
    if debug:
        return
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"trl_vllm_{host}_{port}.log")

    # Open log file for writing
    log_file = open(log_path, "w", encoding="utf-8")

    process = subprocess.Popen(
        [
            "trl", "vllm-serve",
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--trust-remote-code",
            "--tensor_parallel_size", str(tp_size),
            "--max_model_len", str(max_model_len),
            "--gpu_memory_utilization", str(gpu_mem),
        ],
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,
        env=env
    )
    print(f"🚀 trl vLLM server is starting (pid={process.pid}), logs written to {log_path}")

    # Keyword detection
    ready_keywords = [
        "Application startup complete.",
    ]
    timeout = 300  # Maximum wait time: 5 minutes
    start_time = time.time()

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    print(line, end="")  # Also print key logs to terminal
                    if any(keyword in line for keyword in ready_keywords):
                        print("✅ trl vLLM server is ready!")
                        return process
                else:
                    time.sleep(0.5)  # Wait for new log writes

                if time.time() - start_time > timeout:
                    print("❌ Timeout waiting, trl vLLM server startup failed")
                    return process
    except:
        return process


def stop_vllm_server(process):
    """
    Safely stop vLLM server process and its child processes.
    
    Args:
        process: Process object to stop
    """
    if process is None:
        return
    
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return

    # Kill child processes first
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # Then kill main process
    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass

    process.wait()
    print("✅ vLLM server stopped, GPU memory should be released")
