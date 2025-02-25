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

def interpolate_power(mfu: float, gpu_type: str) -> float:
    """
    Power interpolation considering that max GPU utilization (and thus max power draw)
    can occur even at lower MFU values.
    
    Args:
        mfu: Model FLOP Utilization (0-1)
        gpu_type: Type of GPU (e.g., "a100")
    
    Notes:
        - 100% GPU utilization (max power draw) can occur at MFU values as low as 20-45%
        - Power should never exceed max_util (100% GPU utilization power)
        - We use a more aggressive scaling for lower MFU values to reflect this
    """
    gpu_config = GPU_POWER_CONFIGS[gpu_type]
    
    # Define the MFU threshold where we might hit max GPU utilization
    TYPICAL_HIGH_MFU = 0.45  # Based on typical high MFU values for inference
    
    # More aggressive power scaling for lower MFU values
    # This reflects that we can hit high GPU utilization even at lower MFU
    utilization_factor = min(1.0, (mfu / TYPICAL_HIGH_MFU) ** 0.7)  # Power of 0.7 makes scaling more aggressive
    
    # Calculate power, capped at max_util
    power = gpu_config.idle + (gpu_config.max_util - gpu_config.idle) * utilization_factor
    return power

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

def calculate_energy_consumption(gpu_hrs, power_values, mfu, gpu_type):
    """
    Calculate the total energy consumption based on GPU hours, interpolated power, and MFU.

    Args:
        gpu_hrs (float): Number of GPU hours.
        power_values (dict): Power consumption values (idle, 10%, 50%, 100%).
        mfu (float): Model FLOP Utilization (0 to 100).
        gpu_type (str): GPU type.

    Returns:
        float: Energy consumption in kilowatt-hours (kWh).
    """
    effective_power = interpolate_power(mfu, gpu_type)  # Pass gpu_type here
    energy_kwh = effective_power * gpu_hrs / 1000  # Convert watts to kilowatt-hours
    return energy_kwh

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
    Extract MFU values and calculate energy consumption for each stage across all replicas,
    considering the number of GPUs involved.

    Args:
        run_dir (str): Simulation output directory containing the MFU distribution files.
        power_values (dict): GPU-specific power consumption values.

    Returns:
        None
    """
    # Get GPU type from config
    config_file = f"{run_dir}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    gpu_type = config["cluster_config"]["replica_config"]["device"].lower()

    # Locate all MFU files across replicas and stages
    mfu_files = glob.glob(f"{run_dir}/plots/replica_*_stage_*_mfu.json")
    if not mfu_files:
        logger.warning("No MFU files found in the directory.")
        return

    # Retrieve the number of GPUs from the config file
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

    # Get the start time of the simulation
    start_time = datetime.now()  # or use a specific start time if needed
    
    # Process each MFU file
    for mfu_file in mfu_files:
        with open(mfu_file, "r") as f:
            mfu_data = json.load(f)

        # Dynamically extract replica and stage information from the file name
        replica_stage_match = re.search(r"replica_(\d+)_stage_(\d+)", mfu_file)
        if not replica_stage_match:
            logger.warning(f"Failed to parse replica and stage from {mfu_file}")
            continue

        replica_id = int(replica_stage_match.group(1))
        stage_id = int(replica_stage_match.group(2))

        # Get the correct MFU distribution key dynamically
        key = f"replica_{replica_id}_stage_{stage_id}_mfu_distribution"
        if key not in mfu_data:
            logger.warning(f"Key {key} not found in {mfu_file}")
            continue

        # Extract the "distribution" field
        entries = mfu_data[key]

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(entries)

        # Filter out rows with 0 MFU values
        df = df[df["mfu"] > 0]

        # Sort by time
        df = df.sort_values(by="time")

        # Calculate energy consumption
        for i in range(1, len(df)):
            time_seconds = df.iloc[i]["time"]
            # Convert seconds to datetime
            current_datetime = start_time + timedelta(seconds=time_seconds)
            # Format datetime as string: YYYY-MM-DDThh:mm:ss
            formatted_time = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")
            
            time_previous = df.iloc[i - 1]["time"]
            mfu = df.iloc[i]["mfu"] / 100  # Convert percentage to ratio
            execution_time = time_seconds - time_previous

            # Skip invalid execution times
            if execution_time <= 0:
                logger.warning(f"Execution time is non-positive: {execution_time} at time {formatted_time}")
                continue

            # Convert execution time from seconds to hours and account for the number of GPUs
            gpu_hrs = (execution_time / 3600) * num_gpus
            
            # Calculate effective power for this MFU value
            effective_power = interpolate_power(mfu, gpu_type)
            energy_consumption = calculate_energy_consumption(gpu_hrs, power_values, mfu, gpu_type)

            mfu_energy_power_data.append({
                "time": time_seconds,  # Original time in seconds
                "time_extended": formatted_time,  # New column with ISO format datetime
                "replica": replica_id,
                "stage": stage_id,
                "mfu": mfu,
                "energy": energy_consumption,
                "gpu_hrs": gpu_hrs,
                "effective_power": effective_power
            })

    # Save results to a CSV
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
    max_vals = []  # New list to store peak values

    for stat_file in stat_files:
        stat = json.load(open(stat_file))
        for k, v in stat.items():
            if k.endswith("weighted_mean"):
                vals.append(v)
            if k.endswith("_max"):  # Extract max power
                max_vals.append(v)

    if len(vals) == 0:
        return {f"{stat_name}_mean": np.nan, f"{stat_name}_peak": np.nan}

    return {
        f"{stat_name}_mean": sum(vals) / len(vals),
        f"{stat_name}_peak": max(max_vals) if max_vals else np.nan  # Use max if available
    }



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
    
    # Extract Prefill and Decode token configuration
    try:
        prefill_tokens_per_request = config["request_generator_config"]["length_generator_config"]["prefill_tokens"]
        decode_tokens_per_request = config["request_generator_config"]["length_generator_config"]["decode_tokens"]
        total_tokens_processed = prefill_tokens_per_request + decode_tokens_per_request
    except KeyError:
        logger.warning("Prefill and decode token counts not found in config. Setting to default values.")
        prefill_tokens_per_request = np.nan
        decode_tokens_per_request = np.nan
        total_tokens_processed = np.nan

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

    def extract_batch_size_stats(run_dir):
        """
        Extract actual batch size statistics by reading from the 'mfu_distribution' entries in MFU JSON files.
        """
        batch_size_files = glob.glob(f"{run_dir}/plots/replica_*_stage_*_mfu.json")

        all_batch_sizes = []
        batch_stage_counts = []

        for file in batch_size_files:
            with open(file, "r") as f:
                data = json.load(f)

            # Dynamically get the correct key
            for key, value in data.items():
                if key.endswith("_mfu_distribution") and isinstance(value, list):
                    # Extract batch_size from valid entries
                    batch_sizes = [entry["batch_size"] for entry in value if "batch_size" in entry]

                    if batch_sizes:
                        all_batch_sizes.extend(batch_sizes)
                        batch_stage_counts.extend([1] * len(batch_sizes))  # Assign weight of 1 per batch size

        if not all_batch_sizes:
            print("⚠️ No batch sizes found in any files!")
            return {"actual_batch_size_weighted_mean": np.nan, "actual_batch_size_std": np.nan}

        # Convert lists to NumPy arrays
        batch_sizes_array = np.array(all_batch_sizes)
        batch_stage_counts_array = np.array(batch_stage_counts)

        # Ensure the arrays have the same shape
        if batch_sizes_array.shape != batch_stage_counts_array.shape:
            print(f"Shape mismatch detected: batch_sizes {batch_sizes_array.shape}, batch_stage_counts {batch_stage_counts_array.shape}")
            batch_stage_counts_array = np.ones_like(batch_sizes_array)  # Assign equal weights if mismatch occurs

        # Compute weighted mean batch size
        weighted_mean_batch_size = np.average(batch_sizes_array, weights=batch_stage_counts_array)
        batch_size_std = np.std(batch_sizes_array)

        return {
            "actual_batch_size_weighted_mean": weighted_mean_batch_size,
            "actual_batch_size_std": batch_size_std,
        }


    
    memory_usage_stats = extract_utilization_stats(run_dir, "memory_usage")
    mfu_stats = extract_utilization_stats(run_dir, "mfu")
    busy_time_percent_stats = extract_utilization_stats(run_dir, "busy_time_percent")
    runtime = request_completion_time_series_df["Time (sec)"].max()
    batch_size_stats = extract_batch_size_stats(run_dir)

    config.update(
        {
            **batch_size_stats,
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
            "prefill_tokens_per_request": prefill_tokens_per_request,
            "decode_tokens_per_request": decode_tokens_per_request,
            "total_tokens_processed": total_tokens_processed,
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


    # Extract new fields from df safely
    prefill_tokens_per_request = df["prefill_tokens_per_request"].iloc[0] if "prefill_tokens_per_request" in df.columns else np.nan
    decode_tokens_per_request = df["decode_tokens_per_request"].iloc[0] if "decode_tokens_per_request" in df.columns else np.nan
    total_tokens_processed = df["total_tokens_processed"].sum() if "total_tokens_processed" in df.columns else np.nan
    df["actual_batch_size_weighted_mean"] = df["actual_batch_size_weighted_mean"]
    df["actual_batch_size_std"] = df["actual_batch_size_std"]


    df["num_gpus"] = (
        df["cluster_config"].apply(lambda x: x["num_replicas"])
    * df["cluster_config"].apply(lambda x: x["replica_config"]["tensor_parallel_size"])
    * df["cluster_config"].apply(lambda x: x["replica_config"]["num_pipeline_stages"])
    )
    df["cost"] = (
        df["runtime"] * df["num_gpus"] * df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS) / 3600
    )

    mfu_stats = extract_utilization_stats(sim_results_dir, "mfu")
    power_stats = extract_utilization_stats(sim_results_dir, "power")
    df["mfu_mean"] = mfu_stats["mfu_mean"]
    df["power_mean"] = power_stats["power_mean"]
    df["power_peak"] = power_stats["power_peak"]

    # Quietly check for capacity calculations without warnings
    if "poisson_request_interval_generator_qps" in df.columns:
        df["capacity_per_dollar"] = df["poisson_request_interval_generator_qps"] / (
            df["num_gpus"] * df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS)
        )
        df["capacity_per_replica"] = (
            df["poisson_request_interval_generator_qps"] / df["cluster_config"].apply(lambda x: x["num_replicas"])
        )
    else:
        df["capacity_per_dollar"] = np.nan
        df["capacity_per_replica"] = np.nan

    df["gpu_hrs"] = df["runtime"] * df["num_gpus"] / 3600

    df["num_replica_gpus"] = (
        df["cluster_config"].apply(lambda x: x["replica_config"]["tensor_parallel_size"]) * df["cluster_config"].apply(lambda x: x["replica_config"]["num_pipeline_stages"])
    )
    df["hour_cost_per_replica"] = (
       df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS) * df["num_replica_gpus"]
    )
    region_config = REGIONAL_ENERGY_CONFIGS[region]
    pue = region_config.pue
    carbon_intensity = region_config.carbon_intensity

    power_values = get_gpu_power(sim_results_dir)
    process_mfu_energy(sim_results_dir, power_values)
    manufacturing_emissions = power_values.manufacturing_emissions

    # Get GPU type from config
    config_file = f"{sim_results_dir}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    gpu_type = config["cluster_config"]["replica_config"]["device"].lower()

    # Add gpu_type column to DataFrame before energy calculations
    df["gpu_type"] = gpu_type
    
    df["energy_gpu_kwh"] = df.apply(
        lambda row: calculate_energy_consumption(
            row["gpu_hrs"], power_values, row["mfu_mean"], row["gpu_type"]
        ), axis=1
    )

    df["total_energy_kwh"] = df["energy_gpu_kwh"] * pue

    df["carbon_emissions_gco2eq"] = df.apply(
        lambda row: calculate_carbon_emissions(
            row["total_energy_kwh"], carbon_intensity, manufacturing_emissions, row["gpu_hrs"]
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

    # Get number of requests from config
    config_file = f"{sim_results_dir}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    num_requests = config["request_generator_config"]["num_requests"]

    # Calculate energy per request
    energy_per_request = total_energy_kwh / num_requests if num_requests > 0 else 0

    simulation_stats = {
        "gpu_cost": gpu_cost,
        "sim_cpu_cost": cpu_cost,
        "total_gpu_hrs": total_gpu_hrs,
        "sim_time": sim_time,
        "total_runs": len(run_dirs),
        "valid_runs": len(all_results),
        "mfu_mean": mfu_mean,
        "average_power_watts": df["power_mean"].mean(),
        "peak_power_watts": df["power_peak"].max(),
        "total_energy_kwh": total_energy_kwh,
        "average_energy_per_request": energy_per_request,
        "total_carbon_emissions_gco2eq": total_carbon_emissions,
    }

    simulation_stats.update({
    "num_requests": num_requests,
    "prefill_tokens_per_request": prefill_tokens_per_request,
    "decode_tokens_per_request": decode_tokens_per_request,
    "total_tokens_processed": num_requests * total_tokens_processed,  # Sum over all requests
    })

    simulation_stats.update({
    "actual_batch_size_weighted_mean": df["actual_batch_size_weighted_mean"].mean(),
    "actual_batch_size_std": df["actual_batch_size_std"].mean(),
    })

    simulation_stats["tokens_per_second"] = (simulation_stats["total_tokens_processed"] / sim_time) if sim_time > 0 else np.nan


    # Convert all NumPy types to native Python types before JSON serialization
    simulation_stats = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in simulation_stats.items()}

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