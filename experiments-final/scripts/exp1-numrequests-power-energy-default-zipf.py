import os
import subprocess
import json
import pandas as pd
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
    base_output_dir = os.path.join(args.output_dir, "exp1-numrequests-power-energy-default-zipf")
    os.makedirs(base_output_dir, exist_ok=True)

    # Define request sizes
    request_sizes = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 65536]

    # Hardcoded model parameter sizes (in billions)
    model_params = {
        "meta-llama/Meta-Llama-3-8B": 8e9,  # 8B parameters
        "meta-llama/Llama-2-7b-hf": 7e9,  # 7B parameters
        "meta-llama/Llama-2-70b-hf": 70e9,  # 70B parameters
        "meta-llama/Meta-Llama-3-70B": 70e9,  # 70B parameters
        "microsoft/phi-2": 2.7e9,  # 2.7B parameters
        "internlm/internlm-20b": 20e9,  # 20B parameters
        "codellama/CodeLlama-34b-Instruct-hf": 34e9,  # 34B parameters
        "Qwen/Qwen-72B": 72e9  # 72B parameters
    }

    # Define model configurations (single-GPU and multi-GPU)
    models_single_gpu = {
        "meta-llama/Meta-Llama-3-8B": {"TP": 1, "PP": 1},
        "meta-llama/Llama-2-7b-hf": {"TP": 1, "PP": 1},
        "microsoft/phi-2": {"TP": 1, "PP": 1},
        "internlm/internlm-20b": {"TP": 1, "PP": 1},
        "codellama/CodeLlama-34b-Instruct-hf": {"TP": 1, "PP": 1}
    }

    models_multi_gpu = {
        "meta-llama/Llama-2-70b-hf": {"TP": 2, "PP": 2},
        "meta-llama/Meta-Llama-3-70B": {"TP": 2, "PP": 2},
        "Qwen/Qwen-72B": {"TP": 2, "PP": 2}
    }

    # Combine both single and multi-GPU models
    all_models = {**models_single_gpu, **models_multi_gpu}

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
        "--length_generator_config_type zipf "
        "--interval_generator_config_type poisson "
        "--poisson_request_interval_generator_config_qps 6.45 "
        "--global_scheduler_config_type round_robin "
        "--replica_scheduler_config_type vllm "
        "--vllm_scheduler_config_batch_size_cap 128 "
        "--vllm_scheduler_config_block_size 16 "
        "--vllm_scheduler_config_max_tokens_in_batch 4096 "
        "--vllm_scheduler_config_watermark_blocks_fraction 0.01 "
        "--vllm_scheduler_config_num_blocks 512 "
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
        latest_dir = sorted(subdirs)[-1]  # Get the most recent timestamped folder
        return os.path.join(exp_dir, latest_dir)

    # Run experiments
    for model_name, config in all_models.items():
        for num_requests in request_sizes:
            print(f"Running experiment: Model={model_name}, Requests={num_requests}...")

            # Set output directory for this simulation
            sim_output_dir = os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_req_{num_requests}")
            os.makedirs(sim_output_dir, exist_ok=True)

            # Generate simulation command
            command = base_command.format(
                model_name=model_name,
                TP=config["TP"],
                PP=config["PP"],
                num_requests=num_requests,
                output_dir=sim_output_dir
            )

            try:
                # Run simulation
                subprocess.run(command, shell=True, check=True)
                print(f"Simulation complete for {model_name} with {num_requests} requests.")

                # Find latest timestamped directory
                actual_sim_dir = find_latest_timestamped_dir(sim_output_dir)
                if not actual_sim_dir:
                    print(f"No valid output found for {model_name}, Requests={num_requests}. Skipping...")
                    continue

                # Run energy extraction script using the correct path
                extraction_cmd = [
                    "python", "-m", "vidur.config_optimizer.analyzer.stats_extractor_energy",
                    "--sim-results-dir", actual_sim_dir
                ]
                subprocess.run(extraction_cmd, check=True)
                print(f"Energy analysis complete for {model_name}, Requests={num_requests}.")

                # Read and log analysis results
                analysis_file = os.path.join(actual_sim_dir, "analysis", "simulation_stats_with_energy.json")
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as f:
                        stats = json.load(f)

                    experiment_results.append({
                        "model": model_name,
                        "num_parameters": model_params[model_name],
                        "num_requests": num_requests,
                        "mfu_mean": stats.get("mfu_mean", None),
                        "average_power_watts": stats.get("average_power_watts", None),
                        "total_energy_kwh": stats.get("total_energy_kwh", None),
                        "average_energy_per_request": stats.get("average_energy_per_request", None),
                        "total_gpu_hrs": stats.get("total_gpu_hrs", None),
                        "execution_time_s": stats.get("sim_time", None)
                    })
                    print(f"Data saved for {model_name}, Requests={num_requests}.")

            except subprocess.CalledProcessError as e:
                print(f"Simulation failed for {model_name} with {num_requests} requests: {e}")
            except Exception as e:
                print(f"Error processing results for {model_name} with {num_requests} requests: {e}")

    # Save experiment results
    if experiment_results:
        results_file = os.path.join(base_output_dir, "experiment_results_summary.json")
        with open(results_file, "w") as f:
            json.dump(experiment_results, f, indent=4)

        # Also save as CSV for easier analysis
        results_df = pd.DataFrame(experiment_results)
        csv_file = os.path.join(base_output_dir, "experiment_results_summary.csv")
        results_df.to_csv(csv_file, index=False)

        print(f"Experiment complete! Results saved to: {results_file} and {csv_file}")

if __name__ == "__main__":
    main()