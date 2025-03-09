import os
import subprocess
import json
from datetime import datetime
import pandas as pd

def run_simulation(batch_size: int, output_dir: str):
    """Run a single simulation with specified batch size."""
    cmd = [
        "python", "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "1000",
        "--length_generator_config_type", "zipf",
        "--zipf_request_length_generator_config_min_tokens", "16",
        "--zipf_request_length_generator_config_max_tokens", "4096",
        "--zipf_request_length_generator_config_theta", "1.1",
        "--zipf_request_length_generator_config_prefill_to_decode_ratio", "0.5",
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "10",
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm",
        f"--vllm_scheduler_config_batch_size_cap", str(batch_size),
        "--vllm_scheduler_config_block_size", "16",
        "--vllm_scheduler_config_max_tokens_in_batch", "4096",
        "--vllm_scheduler_config_watermark_blocks_fraction", "0.01",
        "--vllm_scheduler_config_num_blocks", "512",
        "--metrics_config_store_utilization_metrics",
        "--execution_time_predictor_config_type", "random_forrest",
        "--metrics_config_output_dir", output_dir
    ]
    
    print(f"\nRunning simulation with batch_size={batch_size}")
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
    batch_sizes = list(range(1, 21))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("simulator_output", f"batch_size_study_{timestamp}")
    
    # Results collection
    results = []
    
    for batch_size in batch_sizes:
        # Create directory for this batch size
        sim_dir = os.path.join(base_dir, f"batch_{batch_size}")
        os.makedirs(sim_dir, exist_ok=True)
        
        try:
            # Run simulation
            run_simulation(batch_size, sim_dir)
            
            # Run energy analysis using the existing script
            timestamped_dir = run_energy_analysis(sim_dir)
            
            # Collect results
            stats_file = os.path.join(timestamped_dir, "analysis", "simulation_stats_with_energy.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    stats['batch_size'] = batch_size
                    results.append(stats)
                print(f"Successfully collected results for batch_size={batch_size}")
            
        except Exception as e:
            print(f"Error processing batch_size={batch_size}: {e}")
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