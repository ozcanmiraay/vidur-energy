import os
import subprocess
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "VIDUR_OUTPUT_DIR",
            os.path.join(Path(__file__).parent.parent.parent, "simulator_output")
        ),
        help="Base directory for simulation output"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    base_output_dir = os.path.join(args.output_dir, "exp4-qps-power-energy")
    os.makedirs(base_output_dir, exist_ok=True)

    num_points_per_range = 8  # Ensure the same number of values <1 and >1

    # Generate log-space values
    qps_values_below_1 = np.round(np.logspace(np.log10(0.1), np.log10(1), num=num_points_per_range), 2).tolist()
    qps_values_above_1 = np.round(np.logspace(np.log10(2), np.log10(50), num=num_points_per_range), 1).tolist()

    qps_values = qps_values_below_1 + qps_values_above_1

    # Fixed parameters
    num_requests = 2**14  # Constant workload
    model_name = "meta-llama/Meta-Llama-3-8B"
    model_params = {"meta-llama/Meta-Llama-3-8B": 8e9}  # 8B params
    model_config = {"TP": 1, "PP": 1}  # Fixed model parallelism

    # Base command template
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
        "--poisson_request_interval_generator_config_qps {qps} "
        "--global_scheduler_config_type round_robin "
        "--replica_scheduler_config_type vllm "
        "--metrics_config_store_utilization_metrics "
        "--metrics_config_output_dir {output_dir} "
        "--no-metrics_config_enable_chrome_trace "
        "--no-metrics_config_store_plots "
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
    for qps in qps_values:
        print(f"\nRunning experiment: Model={model_name}, QPS={qps}...")

        sim_output_dir = os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_qps_{qps}")
        os.makedirs(sim_output_dir, exist_ok=True)

        command = base_command.format(
            model_name=model_name,
            TP=model_config["TP"],
            PP=model_config["PP"],
            num_requests=num_requests,
            qps=qps,
            output_dir=sim_output_dir
        )

        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Simulation complete for {model_name}, QPS={qps}.")

            # Find latest timestamped directory
            actual_sim_dir = find_latest_timestamped_dir(sim_output_dir)
            if not actual_sim_dir:
                print(f"No valid output found for {model_name}, QPS={qps}. Skipping...")
                continue

            # Add this block back - Run energy extraction
            extraction_cmd = [
                "python", "-m", "vidur.config_optimizer.analyzer.stats_extractor_energy",
                "--sim-results-dir", actual_sim_dir
            ]
            subprocess.run(extraction_cmd, check=True)
            print(f"Energy analysis complete for {model_name}, QPS={qps}.")

            # Read analysis results
            analysis_file = os.path.join(actual_sim_dir, "analysis", "simulation_stats_with_energy.json")
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    stats = json.load(f)

                experiment_results.append({
                    "model": model_name,
                    "num_parameters": model_params[model_name],
                    "num_requests": num_requests,
                    "qps": qps,
                    "mfu_weighted_mean": stats.get("mfu_mean", None),
                    "average_power_watts": stats.get("average_power_watts", None),
                    "total_energy_kwh": stats.get("total_energy_kwh", None),
                    "execution_time_s": stats.get("sim_time", None)
                })
                print(f"Data saved for {model_name}, QPS={qps}.")

        except subprocess.CalledProcessError as e:
            print(f"Simulation failed for {model_name}, QPS={qps}: {e}")
        except Exception as e:
            print(f"Error processing results for {model_name}, QPS={qps}: {e}")

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

if __name__ == "__main__":
    main()