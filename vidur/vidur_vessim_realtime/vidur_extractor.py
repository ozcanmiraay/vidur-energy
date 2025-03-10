# vidur_extractor.py
import os
import subprocess
import pandas as pd
from datetime import timedelta

# extract_vidur_data runs the stats_extractor_energy script if needed, reads mfu_energy_power_data.csv, 
# resamples to 1min (or your chosen freq), and returns both a DataFrame and the chunk’s duration in seconds.
# aggregate_to_minute shifts the local timeline to global.


def extract_vidur_data(vidur_output_dir: str, agg_freq: str = "1min"):
    """
    Runs the existing stats_extractor_energy, then reads mfu_energy_power_data.csv
    Aggregates to chosen freq, returns (DataFrame, chunk_duration_seconds).
    """
    # Find the timestamped subdirectory inside the chunk folder
    subdirs = [os.path.join(vidur_output_dir, d) for d in os.listdir(vidur_output_dir)
               if os.path.isdir(os.path.join(vidur_output_dir, d)) and d[0].isdigit()]
    if not subdirs:
        print(f"No timestamped subdirectory found inside {vidur_output_dir}")
        return pd.DataFrame(), 0.0

    subdirs.sort()
    vidur_actual_output_dir = subdirs[-1]  # latest or only one

    analysis_dir = os.path.join(vidur_actual_output_dir, "analysis")
    csv_path = os.path.join(analysis_dir, "mfu_energy_power_data.csv")

    # 1) If not present, run the extractor
    if not os.path.exists(csv_path):
        try:
            subprocess.run([
                "python", "-m", "vidur.config_optimizer.analyzer.stats_extractor_energy",
                "--sim-results-dir", vidur_actual_output_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting Vidur stats: {e}")
            return pd.DataFrame(), 0.0

    if not os.path.exists(csv_path):
        print(f"Could not find extracted CSV at {csv_path}")
        return pd.DataFrame(), 0.0

    df = pd.read_csv(csv_path)

    if "time_extended" not in df.columns:
        print("Missing 'time_extended' in stats output CSV.")
        return pd.DataFrame(), 0.0

    df["time_extended"] = pd.to_datetime(df["time_extended"])
    df.set_index("time_extended", inplace=True)

    if "effective_power" in df.columns:
        df["power_usage_watts"] = df["effective_power"] * -1.0
    else:
        df["power_usage_watts"] = 0.0

    if len(df) < 2:
        print("Not enough data points to compute chunk duration.")
        return df, 0.0

    chunk_start = df.index[0]
    chunk_end = df.index[-1]
    chunk_duration_sec = (chunk_end - chunk_start).total_seconds()

    agg = df.resample(agg_freq).mean(numeric_only=True)

    return agg, chunk_duration_sec


def aggregate_to_minute(usage_df: pd.DataFrame, chunk_start_time, chunk_duration):
    """
    Shifts usage_df from local chunk-based timestamps (e.g. 0 -> 420s)
    to absolute times [chunk_start_time, chunk_start_time + chunk_duration].
    """
    if usage_df.empty:
        return usage_df

    # usage_df index is from chunk_start -> chunk_end in local reference.
    # We find the offset
    local_start = usage_df.index[0]
    offset_sec = (local_start - local_start).total_seconds()  # basically 0
    # We'll shift everything so that usage_df.index= usage_df.index - local_start + chunk_start_time
    # but if local_start is  "some datetime," we do it carefully:

    def shift_timestamp(ts):
        delta = (ts - local_start)
        return chunk_start_time + delta

    usage_df_shifted = usage_df.copy()
    usage_df_shifted.index = usage_df.index.map(shift_timestamp)

    # Ensure we only keep data up to chunk_start_time + chunk_duration
    chunk_end_abs = chunk_start_time + timedelta(seconds=chunk_duration)
    usage_df_shifted = usage_df_shifted.loc[usage_df_shifted.index < chunk_end_abs]

    return usage_df_shifted