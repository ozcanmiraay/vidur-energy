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

def interpolate_power_v2(mfu: float, gpu_type: str, memory_util: float = None, batch_size: int = None) -> float:
    """Linear interpolation between idle and max utilization power states, with memory and batch effects."""
    gpu_config = GPU_POWER_CONFIGS[gpu_type]
    
    # 1. Base compute power (same as original)
    compute_power = gpu_config.idle + (gpu_config.max_util - gpu_config.idle) * mfu
    
    # 2. Memory power component (if memory utilization is provided)
    if memory_util is not None:
        MEMORY_POWER_FACTOR = 0.25
        memory_power = compute_power * MEMORY_POWER_FACTOR * memory_util
        compute_power += memory_power
    
    # 3. Batch efficiency (if batch size is provided)
    if batch_size is not None:
        batch_factor = 1.0 - (0.1 * min(batch_size/512, 1.0))
        compute_power *= batch_factor
    
    return compute_power

def get_gpu_power(sim_results_dir):
    """Get GPU config based on type specified in simulation config."""
    config_file = f"{sim_results_dir}/config.json"
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        gpu_type = config_data['cluster_config']['replica_config']['device'].lower()
        return GPU_POWER_CONFIGS[gpu_type]
    except Exception as e:
        logger.error(f"Error reading config file for GPU power: {e}")
        return GPU_POWER_CONFIGS["a100"]  # Default to A100 if there's an error

def calculate_energy_consumption(gpu_hrs, power_values, mfu, gpu_type, memory_util=None, batch_size=None):
    """Calculate energy consumption with enhanced power model."""
    effective_power = interpolate_power_v2(
        mfu=mfu,
        gpu_type=gpu_type,
        memory_util=memory_util,
        batch_size=batch_size
    )
    return effective_power * gpu_hrs / 1000

def calculate_total_energy_with_pue(energy_gpu, pue):
    """
    Adjust energy usage with the PUE (Power Usage Effectiveness) factor.
    """
    return energy_gpu * pue  # PUE adjusts for energy overhead in data centers

def calculate_carbon_emissions(total_energy_kwh, carbon_intensity, manufacturing_emissions, gpu_hours):
    """
    Calculate the carbon emissions based on total energy, carbon intensity, and manufacturing emissions.

    Args:
        total_energy_kwh (float): Total energy consumption in kWh.
        carbon_intensity (float): Carbon intensity (gCO2eq/kWh).
        manufacturing_emissions (float): Scope 3 emissions (gCO2eq) per GPU-hour.
        gpu_hours (float): GPU usage in hours.

    Returns:
        float: Total carbon emissions in gCO2eq.
    """
    # Scope 2 emissions
    scope_2_emissions = total_energy_kwh * carbon_intensity
    # Scope 3 emissions (manufacturing)
    scope_3_emissions = manufacturing_emissions * gpu_hours
    # Total emissions
    return scope_2_emissions + scope_3_emissions

def process_mfu_energy(run_dir: str, power_values: dict):
    """
    Extract MFU values and calculate energy consumption for each stage across all replicas.
    Now includes memory utilization and batch size effects.
    """
    # Get GPU type from config
    config_file = f"{run_dir}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    gpu_type = config["cluster_config"]["replica_config"]["device"].lower()

    # Get memory usage data
    memory_file = f"{run_dir}/plots/replica_1_memory_usage.json"
    with open(memory_file, 'r') as f:
        memory_data = json.load(f)
    memory_df = pd.DataFrame(memory_data['replica_1_memory_usage_distribution'])
    
    # Rest of setup (num_gpus calculation etc.) stays the same
    mfu_files = glob.glob(f"{run_dir}/plots/replica_*_stage_*_mfu.json")
    if not mfu_files:
        logger.warning("No MFU files found in the directory.")
        return

    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)
        num_gpus = (
            config_data["cluster_config"]["num_replicas"] *
            config_data["cluster_config"]["replica_config"]["tensor_parallel_size"] *
            config_data["cluster_config"]["replica_config"]["num_pipeline_stages"]
        )
    except KeyError:
        logger.warning("Number of GPUs could not be determined. Defaulting to 1 GPU.")
        num_gpus = 1

    mfu_energy_power_data = []
    start_time = datetime.now()

    for mfu_file in mfu_files:
        with open(mfu_file, "r") as f:
            mfu_data = json.load(f)

        replica_stage_match = re.search(r"replica_(\d+)_stage_(\d+)", mfu_file)
        if not replica_stage_match:
            logger.warning(f"Failed to parse replica and stage from {mfu_file}")
            continue

        replica_id = int(replica_stage_match.group(1))
        stage_id = int(replica_stage_match.group(2))

        key = f"replica_{replica_id}_stage_{stage_id}_mfu_distribution"
        if key not in mfu_data:
            logger.warning(f"Key {key} not found in {mfu_file}")
            continue

        entries = mfu_data[key]
        df = pd.DataFrame(entries)
        df = df[df["mfu"] > 0]
        df = df.sort_values(by="time")

        for i in range(1, len(df)):
            time_seconds = df.iloc[i]["time"]
            current_datetime = start_time + timedelta(seconds=time_seconds)
            formatted_time = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
            
            time_previous = df.iloc[i - 1]["time"]
            mfu = df.iloc[i]["mfu"] / 100
            batch_size = df.iloc[i].get("batch_size", 1)
            execution_time = time_seconds - time_previous

            if execution_time <= 0:
                logger.warning(f"Execution time is non-positive: {execution_time} at time {formatted_time}")
                continue

            # Find closest memory usage timestamp and convert percentage to ratio
            memory_usage = memory_df.iloc[
                (memory_df['time'] - time_seconds).abs().argsort()[0]
            ]['memory_usage']
            memory_util = memory_usage / 100  # Convert percentage to ratio (0-1)

            gpu_hrs = (execution_time / 3600) * num_gpus
            
            # Calculate power and energy with all factors
            effective_power = interpolate_power_v2(
                mfu=mfu,
                gpu_type=gpu_type,
                memory_util=memory_util,
                batch_size=batch_size
            )
            
            energy_consumption = calculate_energy_consumption(
                gpu_hrs=gpu_hrs,
                power_values=power_values,
                mfu=mfu,
                gpu_type=gpu_type,
                memory_util=memory_util,
                batch_size=batch_size
            )

            mfu_energy_power_data.append({
                "time": time_seconds,
                "time_extended": formatted_time,
                "replica": replica_id,
                "stage": stage_id,
                "mfu": mfu,
                "memory_util": memory_util,
                "batch_size": batch_size,
                "energy": energy_consumption,
                "gpu_hrs": gpu_hrs,
                "effective_power": effective_power
            })

    # Save results with additional columns
    output_file = os.path.join(run_dir, "analysis/mfu_energy_power_data.csv")
    pd.DataFrame(mfu_energy_power_data).to_csv(output_file, index=False)
    logger.info(f"MFU and energy data saved to {output_file}")

def extract_stat_from_request_metrics(
    request_metrics_df: pd.DataFrame,
    stat_name: str,
    stat_short_name: str = None,
):
    if stat_short_name is None:
        stat_short_name = stat_name

    stats = request_metrics_df[stat_name].describe().to_dict()
    # add 95th and 99th percentile
    stats["90%"] = request_metrics_df[stat_name].quantile(0.90)
    stats["95%"] = request_metrics_df[stat_name].quantile(0.95)
    stats["99%"] = request_metrics_df[stat_name].quantile(0.99)

    stats_dict = {f"{stat_short_name}_{k}": v for k, v in stats.items()}
    return stats_dict


def extract_stats_from_cdf_df(
    cdf_df: pd.DataFrame,
    stat_name: str,
    stat_short_name: str = None,
    extract_all: bool = False,
):
    if stat_short_name is None:
        stat_short_name = stat_name

    if extract_all:
        cdf_df["cdf_rounded"] = cdf_df["cdf"].round(2)
        cdf_df = cdf_df.drop_duplicates(subset="cdf_rounded", keep="first")
        return {f"{stat_short_name}_cdf": cdf_df[stat_name].tolist()[1:]}

    percentile_map = {
        "min": 0.0,
        "25%": 0.25,
        "50%": 0.5,
        "75%": 0.75,
        "90%": 0.90,
        "95%": 0.95,
        "99%": 0.99,
        "max": 1.0,
    }
    stats = {
        k: cdf_df[cdf_df["cdf"] == v][stat_name].iloc[0]
        for k, v in percentile_map.items()
    }
    stats_dict = {f"{stat_short_name}_{k}": v for k, v in stats.items()}
    return stats_dict


def extract_utilization_stats(run_dir: str, stat_name: str):
    stat_files = glob.glob(f"{run_dir}/plots/replica_*{stat_name}.json")
    vals = []
    for stat_file in stat_files:
        stat = json.load(open(stat_file))
        for k, v in stat.items():
            if k.endswith("weighted_mean"):
                vals.append(v)

    if len(vals) == 0:
        return {f"{stat_name}_mean": np.nan}

    return {f"{stat_name}_mean": sum(vals) / len(vals)}


def process_run(run_dir: str):
    config_file = f"{run_dir}/config.json"  # Adjusted to json
    request_metrics_file = f"{run_dir}/request_metrics.csv"
    tbt_file = f"{run_dir}/plots/batch_execution_time.csv"
    ttft_file = f"{run_dir}/plots/prefill_e2e_time.csv"
    batch_size_file = f"{run_dir}/plots/batch_size.csv"
    batch_num_tokens_file = f"{run_dir}/plots/batch_num_tokens.csv"
    request_completion_time_series_file = (
        f"{run_dir}/plots/request_completion_time_series.csv"
    )

    try:
        # Load config file
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file {config_file} not found.")

        # Load request metrics
        if os.path.exists(request_metrics_file):
            request_metrics_df = pd.read_csv(request_metrics_file)
        else:
            raise FileNotFoundError(f"Request metrics file {request_metrics_file} not found.")

        # Load tbt_file (optional, may not exist)
        tbt_df = pd.read_csv(tbt_file) if os.path.exists(tbt_file) else None

        # Load ttft_file (optional, may not exist)
        ttft_df = pd.read_csv(ttft_file) if os.path.exists(ttft_file) else None

        # Load batch_size_file (optional, may not exist)
        batch_size_df = pd.read_csv(batch_size_file) if os.path.exists(batch_size_file) else None

        # Load batch_num_tokens_file (optional, may not exist)
        batch_num_tokens_df = pd.read_csv(batch_num_tokens_file) if os.path.exists(batch_num_tokens_file) else None

        # Load request_completion_time_series_file (optional, may not exist)
        request_completion_time_series_df = pd.read_csv(request_completion_time_series_file) if os.path.exists(request_completion_time_series_file) else None

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

    # Check if replica_scheduler_config exists and access the scheduler name
    if "replica_scheduler_config" in config:
        scheduler_name = config["replica_scheduler_config"].get("name", None)
        if scheduler_name == "sarathi" and config["replica_scheduler_config"].get("chunk_size", None) == 4096:
            config["replica_scheduler_config"]["name"] = "orca+"
    else:
        logger.warning("'replica_scheduler_config' not found in config")

    request_scheduling_delay_stats = extract_stat_from_request_metrics(
        request_metrics_df, "request_scheduling_delay"
    )
    request_e2e_time_normalized_stats = extract_stat_from_request_metrics(
        request_metrics_df, "request_e2e_time_normalized"
    )
    ttft_stats = extract_stat_from_request_metrics(
        request_metrics_df, "prefill_e2e_time", "ttft"
    )
    ttft_cdf = extract_stats_from_cdf_df(
        ttft_df, "prefill_e2e_time", "ttft", extract_all=True
    )
    tbt_stats = extract_stats_from_cdf_df(tbt_df, "batch_execution_time", "tbt")
    tbt_cdf = extract_stats_from_cdf_df(
        tbt_df, "batch_execution_time", "tbt", extract_all=True
    )
    batch_size_stats = extract_stats_from_cdf_df(batch_size_df, "batch_size")
    batch_size_cdf = extract_stats_from_cdf_df(
        batch_size_df, "batch_size", extract_all=True
    )
    batch_num_tokens_cdf = extract_stats_from_cdf_df(
        batch_num_tokens_df, "batch_num_tokens", extract_all=True
    )
    memory_usage_stats = extract_utilization_stats(run_dir, "memory_usage")
    mfu_stats = extract_utilization_stats(run_dir, "mfu")
    busy_time_percent_stats = extract_utilization_stats(run_dir, "busy_time_percent")
    runtime = request_completion_time_series_df["Time (sec)"].max() 

    config.update(
        {
            **request_scheduling_delay_stats,
            **request_e2e_time_normalized_stats,
            **tbt_stats,
            **ttft_stats,
            **memory_usage_stats,
            **mfu_stats,
            **busy_time_percent_stats,
            **ttft_cdf,
            **tbt_cdf,
            **batch_size_stats,
            **batch_size_cdf,
            **batch_num_tokens_cdf,
            "runtime": runtime,
        }
    ) 
    return config

def get_sim_time_from_request_completion(run_dir: str):
    request_completion_file = f"{run_dir}/plots/request_completion_time_series.csv"
    
    if os.path.exists(request_completion_file):
        request_completion_df = pd.read_csv(request_completion_file)
        return request_completion_df["Time (sec)"].max()  # Get the last timestamp in the file
    else:
        logger.warning(f"request_completion_time_series.csv not found in {run_dir}")
        return np.nan 

def process_trace(sim_results_dir: str, region: str = "california"):
    analysis_dir = f"{sim_results_dir}/analysis"

    if os.path.exists(f"{analysis_dir}/stats.csv") and os.path.exists(
        f"{analysis_dir}/simulation_stats.yml"
    ):
        return

    os.makedirs(analysis_dir, exist_ok=True)

    run_dirs = [sim_results_dir]

    num_cores = os.cpu_count() - 2

    with Pool(num_cores) as p:
        all_results = p.map(process_run, run_dirs)

    all_results = [r for r in all_results if r is not None]
    logger.info(f"Total number of runs: {len(run_dirs)} valid runs: {len(all_results)}")

    df = pd.DataFrame(all_results)

    df["num_gpus"] = (
        df["cluster_config"].apply(lambda x: x["num_replicas"])
    * df["cluster_config"].apply(lambda x: x["replica_config"]["tensor_parallel_size"])
    * df["cluster_config"].apply(lambda x: x["replica_config"]["num_pipeline_stages"])
    )
    df["cost"] = (
        df["runtime"] * df["num_gpus"] * df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS) / 3600
    )

    # Get GPU type from config once
    config_file = f"{sim_results_dir}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    gpu_type = config["cluster_config"]["replica_config"]["device"].lower()
    
    mfu_stats = extract_utilization_stats(sim_results_dir, "mfu")
    memory_stats = extract_utilization_stats(sim_results_dir, "memory_usage")
    
    # Load batch size file
    batch_size_file = f"{sim_results_dir}/plots/batch_size.csv"
    batch_size_df = pd.read_csv(batch_size_file) if os.path.exists(batch_size_file) else None
    batch_size_stats = extract_stats_from_cdf_df(batch_size_df, "batch_size") if batch_size_df is not None else {"batch_size_mean": 1}
    
    df["mfu_mean"] = mfu_stats["mfu_mean"]
    df["memory_util"] = memory_stats["memory_usage_mean"] / 100  # Convert percentage to ratio (0-1)
    df["batch_size"] = batch_size_stats.get("batch_size_mean", 1)
    df["gpu_hrs"] = df["runtime"] * df["num_gpus"] / 3600
    
    # Get GPU type and power values
    power_values = get_gpu_power(sim_results_dir)
    
    # Generate mfu_energy_power_data.csv first!
    process_mfu_energy(sim_results_dir, power_values)
    manufacturing_emissions = power_values.manufacturing_emissions

    # Then do the rest of the calculations
    df["energy_gpu_kwh"] = df.apply(
        lambda row: calculate_energy_consumption(
            row["gpu_hrs"], 
            power_values, 
            row["mfu_mean"], 
            gpu_type,
            row["memory_util"],
            row["batch_size"]
        ), axis=1
    )

    df["total_energy_kwh"] = df["energy_gpu_kwh"] * REGIONAL_ENERGY_CONFIGS[region].pue

    df["carbon_emissions_gco2eq"] = df.apply(
        lambda row: calculate_carbon_emissions(
            row["total_energy_kwh"], REGIONAL_ENERGY_CONFIGS[region].carbon_intensity, manufacturing_emissions, row["gpu_hrs"]
        ),
        axis=1
    )
    
    df.to_csv(f"{analysis_dir}/stats_with_energy.csv", index=False)

    gpu_cost = df["cost"].sum()
    total_gpu_hrs = df["gpu_hrs"].sum()
    mfu_mean = df["mfu_mean"].mean()

    sim_time = get_sim_time_from_request_completion(sim_results_dir)
    cpu_hrs = sim_time / 3600
    cpu_cost = cpu_hrs * CPU_MACHINE_COST

    total_gpu_hrs = df["gpu_hrs"].sum()
    total_energy_kwh = df["total_energy_kwh"].sum()
    total_carbon_emissions = df["carbon_emissions_gco2eq"].sum()

    simulation_stats = {
        "gpu_cost": gpu_cost,
        "sim_cpu_cost": cpu_cost,
        "total_gpu_hrs": total_gpu_hrs,
        "sim_time": sim_time,
        "total_runs": len(run_dirs),
        "valid_runs": len(all_results),
        "mfu_mean": mfu_mean,
        "total_energy_kwh": total_energy_kwh,
        "total_carbon_emissions_gco2eq": total_carbon_emissions,
    }

    json.dump(
        simulation_stats, open(f"{analysis_dir}/simulation_stats_with_energy.json", "w"), indent=4
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results-dir", type=str, required=True)
    parser.add_argument("--region", type=str, default="california", 
                       choices=list(REGIONAL_ENERGY_CONFIGS.keys()),
                       help="Region for energy calculations (default: california)")
    args = parser.parse_args()
    process_trace(args.sim_results_dir, region=args.region)

if __name__ == "__main__":
    main() 