import os
import subprocess
import json
import pandas as pd
import itertools

# Path to stats_extractor_energy.py
EXTRACTION_SCRIPT = "/Users/mirayozcan/Desktop/vidur_copy/vidur/vidur/config_optimizer/analyzer/stats_extractor_energy.py"

# Base directory for experiment results
base_output_dir = "/Users/mirayozcan/Desktop/vidur_copy/vidur/simulator_output/exp5-parallelism-power-lenergy"
os.makedirs(base_output_dir, exist_ok=True)

# Change to the correct working directory
os.chdir("/Users/mirayozcan/Desktop/vidur_copy/vidur")

# Fixed model for all tests
model_name = "codellama/CodeLlama-34b-Instruct-hf"
model_params = {
    "codellama/CodeLlama-34b-Instruct-hf": 34e9
}  # Model parameter dictionary

# Number of requests (fixed workload)
num_requests = 1024

# Fixed QPS for all tests
qps = 5

# Define parallelization configurations (Pairwise NVLink Node - 4 GPUs)
parallelization_settings = [
    {"TP": tp, "PP": pp} for tp, pp in itertools.product([1, 2, 4], repeat=2)
]

# Base command template for running simulations
base_command = (
    "python -m vidur.main "
    "--replica_config_device a100 "
    "--replica_config_model_name {model_name} "
    "--cluster_config_num_replicas 1 "
    "--replica_config_tensor_parallel_size {TP} "
    "--replica_config_num_pipeline_stages {PP} "
    "--request_generator_config_type synthetic "
    "--synthetic_request_generator_config_num_requests {num_requests} "
    "--interval_generator_config_type poisson "
    "--poisson_request_interval_generator_config_qps {qps} "
    "--global_scheduler_config_type round_robin "
    "--replica_scheduler_config_type vllm "
    "--metrics_config_store_utilization_metrics "
    "--metrics_config_output_dir {output_dir} "
    "--execution_time_predictor_config_type random_forrest"
)

# Experiment results collection
experiment_results = []


def find_latest_timestamped_dir(exp_dir):
    """Finds the latest timestamped directory within the experiment folder."""
    subdirs = [
        d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))
    ]
    if not subdirs:
        print(f"No timestamped directory found in {exp_dir}. Skipping...")
        return None
    latest_dir = sorted(subdirs)[-1]
    return os.path.join(exp_dir, latest_dir)


# Run experiments
for setting in parallelization_settings:
    TP = setting["TP"]
    PP = setting["PP"]

    print(f"\nRunning experiment: Model={model_name}, TP={TP}, PP={PP}...")

    sim_output_dir = os.path.join(
        base_output_dir, f"{model_name.replace('/', '_')}_TP_{TP}_PP_{PP}"
    )
    os.makedirs(sim_output_dir, exist_ok=True)

    command = base_command.format(
        model_name=model_name,
        TP=TP,
        PP=PP,
        num_requests=num_requests,
        qps=qps,
        output_dir=sim_output_dir,
    )

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Simulation complete for TP={TP}, PP={PP}.")

        # Find latest timestamped directory
        actual_sim_dir = find_latest_timestamped_dir(sim_output_dir)
        if not actual_sim_dir:
            print(f"No valid output found for TP={TP}, PP={PP}. Skipping...")
            continue

        # Run energy extraction script using the correct path
        extraction_cmd = [
            "python",
            "-m",
            "vidur.config_optimizer.analyzer.stats_extractor_energy",
            "--sim-results-dir",
            actual_sim_dir,
        ]
        subprocess.run(extraction_cmd, check=True)
        print(f"Energy analysis complete for TP={TP}, PP={PP}.")

        # Read and log analysis results
        analysis_file = os.path.join(
            actual_sim_dir, "analysis", "simulation_stats_with_energy.json"
        )
        if os.path.exists(analysis_file):
            with open(analysis_file, "r") as f:
                stats = json.load(f)

            experiment_results.append(
                {
                    "model": model_name,
                    "num_parameters": model_params[model_name],
                    "num_requests": num_requests,
                    "TP": TP,
                    "PP": PP,
                    "mfu_mean": stats.get("mfu_mean", None),
                    "average_power_watts": stats.get("average_power_watts", None),
                    "peak_power_watts": stats.get("peak_power_watts", None),
                    "total_energy_kwh": stats.get("total_energy_kwh", None),
                    "tokens_per_second": stats.get("tokens_per_second", None),
                    "requests_per_second": stats.get("requests_per_second", None),
                    "latency_95p": stats.get("e2e_latency_95p", None),
                    "execution_time_s": stats.get("sim_time", None),
                }
            )
            print(f"Data saved for TP={TP}, PP={PP}.")

    except subprocess.CalledProcessError as e:
        print(f"Simulation failed for TP={TP}, PP={PP}: {e}")
    except Exception as e:
        print(f"Error processing results for TP={TP}, PP={PP}: {e}")

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

    print(
        f"\nExperiment complete! Results saved to:\n{results_json_file}\n{results_csv_file}"
    )
