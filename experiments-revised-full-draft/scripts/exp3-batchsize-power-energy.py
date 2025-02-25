import os
import subprocess
import json
import pandas as pd
import numpy as np

# Path to the stats extraction script
EXTRACTION_SCRIPT = "/Users/mirayozcan/Desktop/vidur_copy/vidur/vidur/config_optimizer/analyzer/stats_extractor_energy.py"

# Base directory for experiment results
base_output_dir = "/Users/mirayozcan/Desktop/vidur_copy/vidur/simulator_output/exp3-batchsize-power-energy"
os.makedirs(base_output_dir, exist_ok=True)

# Define the range of max_batch_size values to test
max_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

# Fixed parameters
num_requests = 1024  # Constant workload
model_name = "meta-llama/Meta-Llama-3-8B"
model_params = {"meta-llama/Meta-Llama-3-8B": 8e9}  # 8B params
model_config = {"TP": 1, "PP": 1}  # Fixed model parallelism

# Base command template (Zipf-based request length generator)
base_command = (
    "python -m vidur.main "
    "--replica_config_device a100 "
    "--replica_config_model_name {model_name} "
    "--cluster_config_num_replicas 1 "
    "--replica_config_tensor_parallel_size {TP} "
    "--replica_config_num_pipeline_stages {PP} "
    "--request_generator_config_type synthetic "
    "--synthetic_request_generator_config_num_requests {num_requests} "
    "--length_generator_config_type zipf "
    "--interval_generator_config_type poisson "
    "--poisson_request_interval_generator_config_qps 6.45 "
    "--global_scheduler_config_type round_robin "
    "--replica_scheduler_config_type vllm "
    "--vllm_scheduler_config_batch_size_cap {max_batch_size} "
    "--metrics_config_store_utilization_metrics "
    "--metrics_config_output_dir {output_dir} "
    "--execution_time_predictor_config_type random_forrest"
)

# Experiment results collection
experiment_results = []

def find_latest_timestamped_dir(exp_dir):
    """Finds the latest timestamped directory within the experiment folder."""
    subdirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if not subdirs:
        print(f"No timestamped directory found in {exp_dir}. Skipping...")
        return None
    latest_dir = sorted(subdirs)[-1]
    return os.path.join(exp_dir, latest_dir)

# Run experiments
for max_batch_size in max_batch_sizes:
    print(f"\nRunning experiment: Model={model_name}, Max Batch Size={max_batch_size}...")

    sim_output_dir = os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_batch_{max_batch_size}")
    os.makedirs(sim_output_dir, exist_ok=True)

    command = base_command.format(
        model_name=model_name,
        TP=model_config["TP"],
        PP=model_config["PP"],
        num_requests=num_requests,
        output_dir=sim_output_dir,
        max_batch_size=max_batch_size
    )

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Simulation complete for {model_name}, Max Batch Size={max_batch_size}.")

        # Find latest timestamped directory
        actual_sim_dir = find_latest_timestamped_dir(sim_output_dir)
        if not actual_sim_dir:
            print(f"No valid output found for {model_name}, Max Batch Size={max_batch_size}. Skipping...")
            continue

        # Run energy extraction script
        extraction_cmd = [
            "python", "-m", "vidur.config_optimizer.analyzer.stats_extractor_energy",
            "--sim-results-dir", actual_sim_dir
        ]
        subprocess.run(extraction_cmd, check=True)
        print(f"Energy analysis complete for {model_name}, Max Batch Size={max_batch_size}.")

        # Read analysis results
        analysis_file = os.path.join(actual_sim_dir, "analysis", "simulation_stats_with_energy.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                stats = json.load(f)

            experiment_results.append({
                "model": model_name,
                "num_parameters": model_params[model_name],
                "num_requests": num_requests,
                "max_batch_size": max_batch_size,
                "actual_batch_size_weighted_mean": stats.get("actual_batch_size_weighted_mean", None),
                "actual_batch_size_std": stats.get("actual_batch_size_std", None),
                "mfu_mean": stats.get("mfu_mean", None),
                "average_power_watts": stats.get("average_power_watts", None),
                "peak_power_watts": stats.get("peak_power_watts", None),
                "total_energy_kwh": stats.get("total_energy_kwh", None),
                "tokens_per_second": stats.get("tokens_per_second", None),
                "execution_time_s": stats.get("sim_time", None)
            })
            print(f"Data saved for {model_name}, Max Batch Size={max_batch_size}.")

    except subprocess.CalledProcessError as e:
        print(f"Simulation failed for {model_name}, Max Batch Size={max_batch_size}: {e}")
    except Exception as e:
        print(f"Error processing results for {model_name}, Max Batch Size={max_batch_size}: {e}")

# Save experiment results
if experiment_results:
    results_json_file = os.path.join(base_output_dir, "experiment_results_summary.json")
    results_csv_file = os.path.join(base_output_dir, "experiment_results_summary.csv")

    # Save as JSON
    with open(results_json_file, "w") as f:
        json.dump(experiment_results, f, indent=4)

    # Save as CSV
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(results_csv_file, index=False)

    print(f"\nExperiment complete! Results saved to:\n{results_json_file}\n{results_csv_file}")