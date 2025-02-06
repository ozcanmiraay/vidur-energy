import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
import logging

from vidur.config_optimizer.analyzer.constants import CPU_MACHINE_COST, GPU_COSTS
from vidur.logger import init_logger

# Set up logger with INFO level
logger = init_logger(__name__)
logger.setLevel(logging.INFO)

# Add a stream handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')  # Simplified format
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
    # Try both .yml and .json config files
    config_file = None
    for ext in ['.yml', '.json']:
        test_file = f"{run_dir}/config{ext}"
        if os.path.exists(test_file):
            config_file = test_file
            break
    
    if config_file is None:
        print(f"No config file found in {run_dir}")
        return None

    request_metrics_file = f"{run_dir}/request_metrics.csv"
    tbt_file = f"{run_dir}/plots/batch_execution_time.csv"
    ttft_file = f"{run_dir}/plots/prefill_e2e_time.csv"
    batch_size_file = f"{run_dir}/plots/batch_size.csv"
    batch_num_tokens_file = f"{run_dir}/plots/batch_num_tokens.csv"
    request_completion_time_series_file = (
        f"{run_dir}/plots/request_completion_time_series.csv"
    )

    try:
        with open(config_file, "r") as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)

        # Flatten the nested config structure
        flattened_config = {}
        flattened_config.update({
            "cluster_config_num_replicas": config["cluster_config"]["num_replicas"],
            "replica_config_tensor_parallel_size": config["cluster_config"]["replica_config"]["tensor_parallel_size"],
            "replica_config_num_pipeline_stages": config["cluster_config"]["replica_config"]["num_pipeline_stages"],
            "replica_config_device": config["cluster_config"]["replica_config"]["device"],
            "poisson_request_interval_generator_config_qps": (
                config["request_generator_config"]["interval_generator_config"]["qps"]
                if config["request_generator_config"]["interval_generator_config"]["name"] == "poisson"
                else None
            )
        })
        config.update(flattened_config)

        request_metrics_df = pd.read_csv(request_metrics_file)
        tbt_df = pd.read_csv(tbt_file)
        ttft_df = pd.read_csv(ttft_file)
        batch_size_df = pd.read_csv(batch_size_file)
        batch_num_tokens_df = pd.read_csv(batch_num_tokens_file)
        request_completion_time_series_df = pd.read_csv(
            request_completion_time_series_file
        )
    except FileNotFoundError as e:
        print(f"Could not find file: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing {run_dir}: {str(e)}")
        return None

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

    # Update the scheduler check to match new config structure
    if (
        config["cluster_config"]["replica_scheduler_config"]["name"] == "sarathi"
        and config["cluster_config"]["replica_scheduler_config"]["chunk_size"] == 4096
    ):
        config["replica_scheduler_provider"] = "orca+"
    else:
        config["replica_scheduler_provider"] = config["cluster_config"]["replica_scheduler_config"]["name"]

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


def get_sim_time(sim_results_dir: str):
    output_file = f"{sim_results_dir}/output.log"

    try:
        with open(output_file, "r") as f:
            lines = f.readlines()

        # search for Simulation took time: xxx
        for line in lines:
            if "Simulation took time" in line:
                return float(line.split(":")[-1].strip())
    except FileNotFoundError:
        logger.debug(f"Could not find {output_file}, using runtime from request completion time series")
        return None
    except Exception as e:
        logger.warning(f"Error reading simulation time from {output_file}: {str(e)}")
        return None


def process_trace(sim_results_dir: str):
    analysis_dir = f"{sim_results_dir}/analysis"

    # check if results already exist
    if os.path.exists(f"{analysis_dir}/stats.csv") and os.path.exists(
        f"{analysis_dir}/simulation_stats.json"  # Changed from .yml to .json
    ):
        logger.info(f"Analysis files already exist in {analysis_dir}")
        return

    os.makedirs(analysis_dir, exist_ok=True)

    # Update the glob pattern to match your directory structure
    run_dirs = [sim_results_dir]  # If config is directly in sim_results_dir

    num_cores = os.cpu_count() - 2

    with Pool(num_cores) as p:
        all_results = p.map(process_run, run_dirs)

    # filter out None values
    all_results = [r for r in all_results if r is not None]
    
    if not all_results:
        logger.error(f"No valid results found in {sim_results_dir}")
        return

    df = pd.DataFrame(all_results)
    
    # Remove debug prints
    # print("Available columns:", df.columns.tolist())
    # print("First row:", df.iloc[0].to_dict() if not df.empty else "DataFrame is empty")

    df["num_gpus"] = (
        df["cluster_config_num_replicas"]
        * df["replica_config_tensor_parallel_size"]
        * df["replica_config_num_pipeline_stages"]
    )
    df["cost"] = (
        df["runtime"] * df["num_gpus"] * df["replica_config_device"].map(GPU_COSTS) / 3600
    )
    df["capacity_per_dollar"] = df["poisson_request_interval_generator_config_qps"] / (
        df["num_gpus"] * df["replica_config_device"].map(GPU_COSTS)
    )
    df["gpu_hrs"] = df["runtime"] * df["num_gpus"] / 3600

    df["num_replica_gpus"] = (
        df["replica_config_tensor_parallel_size"]
        * df["replica_config_num_pipeline_stages"]
    )
    df["hour_cost_per_replica"] = (
        df["replica_config_device"].map(GPU_COSTS) * df["num_replica_gpus"]
    )
    df["capacity_per_replica"] = (
        df["poisson_request_interval_generator_config_qps"] / df["cluster_config_num_replicas"]
    )

    # store the df
    df.to_csv(f"{analysis_dir}/stats.csv", index=False)

    gpu_cost = df["cost"].sum()
    total_gpu_hrs = df["gpu_hrs"].sum()

    sim_time = get_sim_time(sim_results_dir)
    if sim_time is None:
        # Use the runtime from the DataFrame if we couldn't get it from output.log
        sim_time = df["runtime"].max()
    
    cpu_hrs = sim_time / 3600
    cpu_cost = cpu_hrs * CPU_MACHINE_COST

    simulation_stats = {
        "gpu_cost": gpu_cost,
        "sim_cpu_cost": cpu_cost,
        "total_gpu_hrs": total_gpu_hrs,
        "sim_time": sim_time,
        "total_runs": len(run_dirs),
        "valid_runs": len(all_results),
    }

    json.dump(
        simulation_stats, open(f"{analysis_dir}/simulation_stats.json", "w"), indent=4
    )
    
    # Add both logger and print
    sim_name = os.path.basename(os.path.normpath(sim_results_dir))
    msg = f"Successfully extracted stats from simulation '{sim_name}'"
    logger.info(msg)
    print(msg)  # Backup print statement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results-dir", type=str, required=True)
    args = parser.parse_args()

    process_trace(args.sim_results_dir)


if __name__ == "__main__":
    main()
