import argparse
import glob
import json
import re
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from vidur.logger import init_logger
from vidur.config_optimizer.analyzer.constants import CPU_MACHINE_COST, GPU_COSTS
from datetime import datetime, timedelta
from vidur.config_optimizer.analyzer.stats_extractor_energy_reporting.config.region_configs import REGIONAL_ENERGY_CONFIGS
from vidur.config_optimizer.analyzer.stats_extractor_energy_reporting.config.gpu_configs import GPU_POWER_CONFIGS

logger = init_logger(__name__)

def get_gpu_config(sim_dir: str):
    """Get GPU configuration from simulation config file."""
    try:
        config_file = os.path.join(sim_dir, "config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        gpu_type = config['cluster_config']['replica_config']['device'].lower()
        return GPU_POWER_CONFIGS[gpu_type]
    except Exception as e:
        logger.error(f"Error reading GPU config: {e}")
        return None

def read_mfu_data(sim_dir: str):
    """Read MFU (Model FLOPs Utilization) data from simulation results."""
    try:
        mfu_file = os.path.join(sim_dir, "plots", "replica_1_stage_1_mfu.json")
        with open(mfu_file, 'r') as f:
            data = json.load(f)
            
        # Extract time series data from the distribution, excluding placeholder zeros
        mfu_series = []
        for entry in data['replica_1_stage_1_mfu_distribution']:
            # Only include entries with non-zero MFU and execution time
            if entry.get('mfu', 0) > 0 and 'execution_time' in entry:
                mfu_series.append({
                    'time': entry['time'],
                    'mfu': entry['mfu'],
                    'execution_time': entry['execution_time'],
                    'num_tokens': entry.get('num_tokens', 0),
                    'batch_size': entry.get('batch_size', 0)
                })
            
        return pd.DataFrame(mfu_series)

    except Exception as e:
        logger.error(f"Error reading MFU data: {e}")
        return None

def calculate_power_usage(mfu: float, gpu_config) -> float:
    """Calculate GPU power usage based on MFU."""
    # Linear interpolation between idle and max power based on MFU
    power = gpu_config.idle + (gpu_config.max_util - gpu_config.idle) * mfu
    return power

def calculate_average_mfu(mfu_data):
    """Get the pre-calculated weighted average MFU."""
    return mfu_data.get('replica_1_stage_1_mfu_weighted_mean', 0.0)

def calculate_energy_metrics(sim_dir: str, region: str = "california"):
    """Calculate energy consumption and related metrics."""
    try:
        # Get GPU configuration
        gpu_config = get_gpu_config(sim_dir)
        if gpu_config is None:
            return None

        # Read MFU data from the JSON file that contains the pre-calculated weighted average
        mfu_file = os.path.join(sim_dir, "plots", "replica_1_stage_1_mfu.json")
        with open(mfu_file, 'r') as f:
            mfu_data = json.load(f)

        # Use the pre-calculated weighted average
        weighted_avg_mfu = calculate_average_mfu(mfu_data)
        
        # Calculate power and energy metrics using the raw time series
        mfu_df = read_mfu_data(sim_dir)
        if mfu_df is None:
            return None

        # Calculate power and energy for each time point
        energy_metrics = []
        for _, row in mfu_df.iterrows():
            power = calculate_power_usage(row['mfu'], gpu_config)
            time_hours = row['time'] / 3600  # Convert seconds to hours
            energy_kwh = power * time_hours / 1000  # Convert watt-hours to kilowatt-hours
            
            energy_metrics.append({
                'time': row['time'],
                'mfu': row['mfu'],
                'power_watts': power,
                'energy_kwh': energy_kwh
            })

        # Create and save energy metrics DataFrame
        energy_df = pd.DataFrame(energy_metrics)
        
        # Create analysis directory
        analysis_dir = os.path.join(sim_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save detailed metrics
        energy_df.to_csv(os.path.join(analysis_dir, "energy_metrics.csv"), index=False)
        
        # Calculate summary statistics
        stats = {
            'total_energy_kwh': energy_df['energy_kwh'].sum(),
            'avg_power_watts': energy_df['power_watts'].mean(),
            'peak_power_watts': energy_df['power_watts'].max(),
            'avg_mfu': weighted_avg_mfu,  # Using our new weighted MFU calculation
            'total_time_hours': energy_df['time'].max() / 3600
        }
        
        # Save summary stats
        with open(os.path.join(analysis_dir, "simulation_stats_with_energy.json"), 'w') as f:
            json.dump(stats, f, indent=4)
            
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating energy metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate energy metrics for simulation results")
    parser.add_argument("--sim-results-dir", required=True, help="Path to simulation results directory")
    parser.add_argument("--region", default="california", choices=list(REGIONAL_ENERGY_CONFIGS.keys()),
                       help="Region for energy calculations")
    args = parser.parse_args()

    stats = calculate_energy_metrics(args.sim_results_dir, args.region)
    if stats:
        print("Energy analysis completed successfully")
        print(f"Total energy consumption: {stats['total_energy_kwh']:.2f} kWh")
        print(f"Average power usage: {stats['avg_power_watts']:.2f} watts")
        print(f"Average MFU: {stats['avg_mfu']:.2%}")

if __name__ == "__main__":
    main()