import os
import subprocess
import json
from datetime import datetime
import pandas as pd
import itertools

def run_simulation(tp_size: int, pp_stages: int, output_dir: str):
    """Run simulation with different tensor parallel and pipeline parallel configurations."""
    cmd = [
        "python", "-m", "vidur.main",
        "--request_generator_config_type", "synthetic",
        # Length generator config
        "--length_generator_config_type", "zipf",
        "--zipf_request_length_generator_config_min_tokens", "32",
        "--zipf_request_length_generator_config_max_tokens", "4096",
        "--zipf_request_length_generator_config_theta", "0.99",
        # Rest of configuration
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", "5",
        "--synthetic_request_generator_config_num_requests", "1000",  # Reduced from 10000
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--replica_config_device", "a100",
        # Parallelism configuration
        "--replica_config_tensor_parallel_size", str(tp_size),
        "--replica_config_num_pipeline_stages", str(pp_stages),
        "--cluster_config_num_replicas", "1",  # Keep this at 1
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm",
        "--vllm_scheduler_config_batch_size_cap", "128",
        "--metrics_config_output_dir", output_dir,
        "--metrics_config_store_utilization_metrics",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_plots"
    ]
    
    print(f"\nRunning simulation with TP={tp_size}, PP={pp_stages}")
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
    tp_sizes = [1, 2, 4]  # Tensor Parallel sizes
    pp_stages = [1, 2, 4]    # Pipeline Parallel stages
    
    # Generate all valid combinations
    configs = []
    for tp, pp in itertools.product(tp_sizes, pp_stages):
        # Skip configurations requiring too many GPUs
        if tp * pp <= 16:  # Assuming max 8 GPUs available
            configs.append((tp, pp))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("simulator_output", f"parallelism_study_{timestamp}")
    
    # Results collection
    results = []
    
    for tp_size, pp_stages in configs:
        # Create directory for this configuration
        sim_dir = os.path.join(base_dir, f"tp{tp_size}_pp{pp_stages}")
        os.makedirs(sim_dir, exist_ok=True)
        
        try:
            # Run simulation
            run_simulation(tp_size, pp_stages, sim_dir)
            
            # Run energy analysis using the existing script
            timestamped_dir = run_energy_analysis(sim_dir)
            
            # Collect results
            stats_file = os.path.join(timestamped_dir, "analysis", "simulation_stats_with_energy.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    stats['tp_size'] = tp_size
                    stats['pp_stages'] = pp_stages
                    stats['total_gpus'] = tp_size * pp_stages
                    results.append(stats)
                print(f"Successfully collected results for TP={tp_size}, PP={pp_stages}")
            
        except Exception as e:
            print(f"Error processing TP={tp_size}, PP={pp_stages}: {e}")
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