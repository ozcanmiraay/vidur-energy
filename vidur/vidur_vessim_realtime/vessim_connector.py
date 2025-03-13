import pandas as pd
import vessim as vs
import datetime
from typing import Tuple

def run_vessim_for_chunk(
    usage_df: pd.DataFrame,
    chunk_start_time: datetime.datetime,
    chunk_end_time: datetime.datetime,
    initial_soc: float,
    step_size: int,
    location: str,
    solar_scale_factor: float,
    battery_capacity: float,
    battery_min_soc: float,
    output_file: str = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Runs Vessim from chunk_start_time to chunk_end_time using real Solcast and WattTime data.
    
    Parameters
    ----------
    usage_df : pd.DataFrame
        Minute-level power usage for this chunk, with 'power_usage_watts' column.
    chunk_start_time : datetime.datetime
        Start time (absolute).
    chunk_end_time : datetime.datetime
        End time (absolute).
    initial_soc : float
        Battery SoC at the start of this chunk.
    step_size : int
        Vessim step size in seconds (usually 60 for 1-minute).
    location : str
        The location key for solar data (e.g. "San Francisco").
    solar_scale_factor : float
        Scaling factor for solar dataset. For instance, 6000 for a 6kW system.
    battery_capacity : float
        Battery capacity in Wh for ClcBattery or SimpleBattery.
    battery_min_soc : float
        Minimum SoC (e.g., 0.2).
    output_file : str, optional
        If provided, Vessim’s Monitor can output a CSV to this file.
    
    Returns
    -------
    chunk_vessim_df : pd.DataFrame
        The step-level logs from Vessim for this chunk (time as index).
    final_battery_soc : float
        Battery SoC at the end of the chunk.
    avg_solar_power : float
        Average solar generation (W) during the chunk.
    """
    # If usage_df is empty or chunk duration is non-positive, skip
    if usage_df.empty:
        print("Usage DataFrame is empty; skipping Vessim run.")
        return pd.DataFrame(), initial_soc, 0.0
    
    duration_sec = int((chunk_end_time - chunk_start_time).total_seconds())
    if duration_sec <= 0:
        print("Non-positive chunk duration; skipping Vessim run.")
        return pd.DataFrame(), initial_soc, 0.0

    # Create signals from usage_df
    power_signal = vs.HistoricalSignal(usage_df[["power_usage_watts"]])

    # Real solar signal via solcast2022_global for given location
    # We pass the 'start_time' as chunk_start_time so it replays from that time onward
    solar_signal = vs.HistoricalSignal.load(
        "solcast2022_global",
        column=location,
        data_dir=None,  # or specify if using a custom data directory
        params={
            "scale": solar_scale_factor,
            "start_time": chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_forecast": False
        }
    )

    # Real carbon intensity signal via WattTime
    carbon_signal = vs.HistoricalSignal.load(
        "watttime2023_caiso-north",
        data_dir=None,  # or specify custom
        params={
            "start_time": chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_forecast": False
        }
    )

    # Create the battery (ClcBattery or SimpleBattery)
    battery = vs.ClcBattery(
        number_of_cells=int(battery_capacity / 3.63),
        initial_soc=initial_soc,
        min_soc=battery_min_soc
    )

    # Initialize environment
    env = vs.Environment(sim_start=chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Create a Monitor
    if not output_file:
        raise ValueError("output_file path must be provided for Vessim output.")
    monitor = vs.Monitor(outfile=output_file)


    # Add microgrid
    env.add_microgrid(
        actors=[
            vs.Actor(name="vidur_power_usage", signal=power_signal),
            vs.Actor(name="solar", signal=solar_signal),
            vs.Actor(name="carbon_intensity", signal=carbon_signal),
        ],
        controllers=[monitor],
        storage=battery,
        step_size=step_size  # 60s for 1-min steps, e.g.
    )

    # Run simulation
    env.run(until=duration_sec)

    # Convert logs to DataFrame
    chunk_vessim_df = pd.read_csv(output_file, parse_dates=["time"], index_col="time")
    if chunk_vessim_df.empty:
        print("No logs from Vessim monitor; returning empty DataFrame.")
        return pd.DataFrame(), battery.soc(), 0.0

    # Set time as index
    if "time" in chunk_vessim_df.columns:
        chunk_vessim_df.set_index("time", inplace=True)

    # Some stats: average solar power
    if "solar.p" in chunk_vessim_df.columns:
        avg_solar_power = chunk_vessim_df["solar.p"].mean()
    else:
        avg_solar_power = 0.0

    final_battery_soc = battery.soc()

    return chunk_vessim_df, final_battery_soc, avg_solar_power