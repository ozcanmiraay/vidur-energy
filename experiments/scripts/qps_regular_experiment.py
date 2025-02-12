import os
import subprocess
import json
from datetime import datetime
import pandas as pd
import numpy as np

def run_simulation(qps: float, output_dir: str):
    """Run a single simulation with specified QPS."""
    cmd = [
        "python", "-m", "vidur.main",
        "--request_generator_config_type", "synthetic",
        # Length generator config (Zipf)
        "--length_generator_config_type", "zipf",
        "--zipf_request_length_generator_config_min_tokens", "32",
        "--zipf_request_length_generator_config_max_tokens", "4096",
        "--zipf_request_length_generator_config_theta", "0.99",
        # Interval generator config (Poisson)
        "--interval_generator_config_type", "poisson",
        f"--poisson_request_interval_generator_config_qps", str(qps),
        # Number of requests
        "--synthetic_request_generator_config_num_requests", "10000",
        # Rest of the configuration
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--replica_config_device", "a100",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--cluster_config_num_replicas", "1",
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm",
        "--vllm_scheduler_config_batch_size_cap", "128",
        "--vllm_scheduler_config_block_size", "16",
        "--vllm_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_scheduler_config_watermark_blocks_fraction", "0.01",
        "--metrics_config_output_dir", output_dir,
        "--metrics_config_store_utilization_metrics"
    ]
    
    print(f"\nRunning simulation with QPS={qps}")
    subprocess.run(cmd, check=True)

def get_timestamped_dir(base_dir: str) -> str:
    """Find the timestamped directory within the simulation output directory."""
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise ValueError(f"No timestamped directory found in {base_dir}")
    return os.path.join(base_dir, sorted(subdirs)[-1])

def run_energy_analysis(sim_dir: str):
    """Run the existing energy analysis script on simulation results."""
    timestamped_dir = get_timestamped_dir(sim_dir)
    cmd = [
        "python", "-m", "vidur.config_optimizer.analyzer.stats_extractor_energy",
        "--sim-results-dir", timestamped_dir
    ]
    print(f"\nRunning energy analysis for {timestamped_dir}")
    subprocess.run(cmd, check=True)
    return timestamped_dir

def main():
    # Experiment configuration
    qps_values = np.linspace(1, 20, 20)  # 20 evenly spaced points from 1 to 20
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("simulator_output", f"qps_regular_study_{timestamp}")
    
    # Results collection
    results = []
    
    for qps in qps_values:
        # Create directory for this QPS value
        sim_dir = os.path.join(base_dir, f"qps_{qps:.2f}")
        os.makedirs(sim_dir, exist_ok=True)
        
        try:
            # Run simulation
            run_simulation(qps, sim_dir)
            
            # Run energy analysis using the existing script
            timestamped_dir = run_energy_analysis(sim_dir)
            
            # Collect results
            stats_file = os.path.join(timestamped_dir, "analysis", "simulation_stats_with_energy.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    stats['qps'] = qps
                    results.append(stats)
                print(f"Successfully collected results for QPS={qps}")
            
        except Exception as e:
            print(f"Error processing QPS={qps}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save aggregated results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(base_dir, "aggregated_results.csv"), index=False)
        with open(os.path.join(base_dir, "aggregated_results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {base_dir}")

if __name__ == "__main__":
    main() 