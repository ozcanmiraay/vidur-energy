# vidur_runner.py
import os
import subprocess
from typing import Dict, Any

# We translate the user’s config dictionary into CLI flags for Vidur.
# We specify --simulation_output_dir so that the results land in output_dir/chunk_{n}.
# This function returns nothing: it just runs Vidur.

def run_vidur_chunk(output_dir: str, config: Dict[str, Any], num_requests: int):
    """
    Runs a single Vidur simulation chunk with the specified config.

    :param output_dir: Where Vidur simulation results (logs, CSVs) will be placed
    :param config: A dictionary specifying model, batch_size, qps, etc.
    :param num_requests: number of requests to run for this chunk
    """
    # Example mapping from config to CLI arguments
    model_name = config.get("model_name", "Llama2-7b")
    batch_size = config.get("batch_size", 16)
    qps = config.get("qps", 20)

    # Build the command
    # You can adapt these based on how your code expects them
    cmd = [
        "python", "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", model_name,
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--request_generator_config_type", "synthetic",
        f"--synthetic_request_generator_config_num_requests={num_requests}",
        "--length_generator_config_type", "zipf",
        "--interval_generator_config_type", "poisson",
        f"--poisson_request_interval_generator_config_qps={qps}",
        "--replica_scheduler_config_type", "vllm",
        f"--vllm_scheduler_config_batch_size_cap={batch_size}",
        "--vllm_scheduler_config_max_tokens_in_batch", "4096",
        "--metrics_config_store_utilization_metrics",
        "--execution_time_predictor_config_type", "random_forrest",
        f"--metrics_config_output_dir={output_dir}"
    ]

    print(f"Running Vidur chunk in {output_dir}:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/Users/mirayozcan/Desktop/vidur_copy/vidur")