import os
import pandas as pd
import subprocess
import datetime
from typing import Dict, Any

from vidur.vidur_vessim_realtime.vidur_runner import run_vidur_chunk
from vidur.vidur_vessim_realtime.vidur_extractor import extract_vidur_data, aggregate_to_minute
from vidur.vidur_vessim_realtime.vessim_connector import run_vessim_for_chunk
from vidur.vidur_vessim_realtime.config_decider import decide_next_config

def realtime_simulation_loop(
    base_output_dir: str,
    simulation_start_time: datetime.datetime,
    simulation_end_time: datetime.datetime,
    initial_vidur_config: Dict[str, Any],
    initial_battery_soc: float = 0.8,
    chunk_size_requests: int = 1000,
    chunk_id_start: int = 0,
    global_agg_freq: str = "1min",
    location: str = "San Francisco",
    solar_scale_factor: float = 6000,
    battery_capacity: float = 5000,
    battery_min_soc: float = 0.2,
    vessim_step_size: int = 60
):
    """
    Iteratively runs Vidur + Vessim simulation chunk-by-chunk.
    """

    global_df = pd.DataFrame()
    current_time = simulation_start_time
    current_battery_soc = initial_battery_soc
    chunk_id = chunk_id_start
    last_chunk_solar = None
    last_chunk_usage = None
    current_config = initial_vidur_config.copy()

    while current_time < simulation_end_time:
        print(f"\n=== Starting chunk {chunk_id} at {current_time} ===")

        # Step 1: Run Vidur chunk
        chunk_output_dir = os.path.join(base_output_dir, f"chunk_{chunk_id}")
        os.makedirs(chunk_output_dir, exist_ok=True)

        run_vidur_chunk(
            output_dir=chunk_output_dir,
            config=current_config,
            num_requests=chunk_size_requests
        )

        # Step 2: Extract and aggregate Vidur usage data
        usage_df, chunk_duration = extract_vidur_data(
            vidur_output_dir=chunk_output_dir,
            agg_freq=global_agg_freq
        )

        usage_df_shifted = aggregate_to_minute(usage_df, current_time, chunk_duration)
        last_chunk_usage = usage_df_shifted["power_usage_watts"].abs().mean() if not usage_df_shifted.empty else 0.0

        # Step 3: Run Vessim simulation
        chunk_end_time = current_time + datetime.timedelta(seconds=chunk_duration)

        # 🔧 NEW: Define Vessim output file path
        vessim_output_path = os.path.join(chunk_output_dir, "vessim_output.csv")

        chunk_vessim_df, final_battery_soc, chunk_solar_avg = run_vessim_for_chunk(
            usage_df=usage_df_shifted,
            chunk_start_time=current_time,
            chunk_end_time=chunk_end_time,
            initial_soc=current_battery_soc,
            step_size=vessim_step_size,
            location=location,
            solar_scale_factor=solar_scale_factor,
            battery_capacity=battery_capacity,
            battery_min_soc=battery_min_soc,
            output_file=vessim_output_path  # ✅ Use proper path now
        )

        # Step 4: Append to global DataFrame
        if not chunk_vessim_df.empty:
            global_df = pd.concat([global_df, chunk_vessim_df])

        # Step 5: Decide next config
        last_chunk_solar = chunk_solar_avg
        current_battery_soc = final_battery_soc

        new_config = decide_next_config(
            current_config,
            current_battery_soc,
            last_chunk_usage,
            last_chunk_solar
        )

        print(f"Chunk {chunk_id} ended at {chunk_end_time}. SoC={final_battery_soc:.3f}, "
              f"solar_avg={chunk_solar_avg:.2f}, usage_avg={last_chunk_usage:.2f}")
        print(f"Next config: {new_config}")

        # Update loop state
        current_config = new_config
        current_time = chunk_end_time
        chunk_id += 1

        if current_time >= simulation_end_time:
            print(f"Reached or exceeded simulation_end_time at {current_time}")
            break

    print("\nSimulation complete. Saving global dataframe ...")
    global_df.sort_index(inplace=True)
    global_output_csv = os.path.join(base_output_dir, "global_sim_log.csv")
    global_df.to_csv(global_output_csv)
    print(f"Global simulation log saved at {global_output_csv}")