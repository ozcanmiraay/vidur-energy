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
    base_output_dir = os.path.join(args.output_dir, "exp2-pdratio-inpseq-power-energy")
    os.makedirs(base_output_dir, exist_ok=True)

    # Fixed request sizes (Tokens per request)
    fixed_request_sizes = [128, 256, 512, 1024, 2048, 4096]  

    # Define number of requests to test
    num_requests = 1024  # Keeping constant workload

    # Define Prefill:Decode ratios to test (logarithmic scaling from 50:1 to 1:50)
    prefill_decode_ratios = np.round(np.logspace(np.log10(50), np.log10(0.02), num=10), 4).tolist()  

    # Selected Model for this experiment (Controlled Condition)
    model_name = "meta-llama/Meta-Llama-3-8B"
    model_params = {"meta-llama/Meta-Llama-3-8B": 8e9}  # Dictionary for consistency

    # Fixed model configuration (Single GPU)
    model_config = {"TP": 1, "PP": 1}

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
        "--length_generator_config_type fixed "
        "--fixed_request_length_generator_config_max_tokens {fixed_request_size} "
        "--fixed_request_length_generator_config_prefill_tokens {prefill_tokens} "
        "--fixed_request_length_generator_config_decode_tokens {decode_tokens} "
        "--interval_generator_config_type poisson "
        "--poisson_request_interval_generator_config_qps 6.45 "
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
    for fixed_request_size in fixed_request_sizes:
        for ratio in prefill_decode_ratios:
            # Ensure valid prefill/decode values
            prefill_tokens = int(round((fixed_request_size * ratio) / (1 + ratio)))
            decode_tokens = fixed_request_size - prefill_tokens

            # Sanity check to avoid negative decode_tokens
            if decode_tokens < 0 or prefill_tokens < 0:
                print(f"⚠️ Skipping invalid ratio {ratio} for request size {fixed_request_size} (prefill={prefill_tokens}, decode={decode_tokens})")
                continue

            print(f"\n Running experiment: Model={model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}...")

            sim_output_dir = os.path.join(base_output_dir, f"{model_name.replace('/', '_')}_len_{fixed_request_size}_PD_{ratio}")
            os.makedirs(sim_output_dir, exist_ok=True)

            command = base_command.format(
                model_name=model_name,
                TP=model_config["TP"],
                PP=model_config["PP"],
                num_requests=num_requests,
                fixed_request_size=fixed_request_size,  
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                output_dir=sim_output_dir
            )

            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Simulation complete for {model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}.")

                # Find latest timestamped directory
                actual_sim_dir = find_latest_timestamped_dir(sim_output_dir)
                if not actual_sim_dir:
                    print(f"⚠️ No valid output found for {model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}. Skipping...")
                    continue

                # Run energy extraction script using the correct path
                extraction_cmd = [
                    "python", "-m", "vidur.config_optimizer.analyzer.stats_extractor_energy",
                    "--sim-results-dir", actual_sim_dir
                ]
                subprocess.run(extraction_cmd, check=True)
                print(f"Energy analysis complete for {model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}.")

                # Read and log analysis results
                analysis_file = os.path.join(actual_sim_dir, "analysis", "simulation_stats_with_energy.json")
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as f:
                        stats = json.load(f)

                    experiment_results.append({
                        "model": model_name,
                        "num_parameters": model_params[model_name],
                        "num_requests": num_requests,
                        "request_length": fixed_request_size,
                        "prefill_decode_ratio": ratio,
                        "prefill_tokens": prefill_tokens,
                        "decode_tokens": decode_tokens,
                        "mfu_mean": stats.get("mfu_mean", None),
                        "average_power_watts": stats.get("average_power_watts", None),
                        "peak_power_watts": stats.get("peak_power_watts", None),
                        "total_energy_kwh": stats.get("total_energy_kwh", None),
                        "tokens_per_second": stats.get("tokens_per_second", None),
                        "execution_time_s": stats.get("sim_time", None)
                    })
                    print(f"Data saved for {model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}.")

            except subprocess.CalledProcessError as e:
                print(f"Simulation failed for {model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}: {e}")
            except Exception as e:
                print(f"Error processing results for {model_name}, Request Length={fixed_request_size}, P:D Ratio={ratio}: {e}")

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