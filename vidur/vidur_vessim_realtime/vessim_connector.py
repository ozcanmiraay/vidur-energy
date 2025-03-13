import pandas as pd
import vessim as vs
import datetime
from typing import Tuple, Optional
import os


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
    output_file: str = None,
    data_dir: Optional[str] = None
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
    data_dir : str, optional
        Path to directory containing the Solcast and WattTime data (defaults to ../data relative to script).
    
    Returns
    -------
    chunk_vessim_df : pd.DataFrame
        The step-level logs from Vessim for this chunk (time as index).
    final_battery_soc : float
        Battery SoC at the end of the chunk.
    avg_solar_power : float
        Average solar generation (W) during the chunk.
    """
    # Handle default data_dir
    if data_dir is None:
        # Automatically resolve the 'data' directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data"))

    # Early exit if no data
    if usage_df.empty:
        print("Usage DataFrame is empty; skipping Vessim run.")
        return pd.DataFrame(), initial_soc, 0.0

    duration_sec = int((chunk_end_time - chunk_start_time).total_seconds())
    if duration_sec <= 0:
        print("Non-positive chunk duration; skipping Vessim run.")
        return pd.DataFrame(), initial_soc, 0.0

    # --- Create Power Signal from Vidur Usage
    power_signal = vs.HistoricalSignal(usage_df[["power_usage_watts"]])

    # --- Solar Signal from Solcast Dataset
    solar_signal = vs.HistoricalSignal.load(
        "solcast2022_global",
        column=location,
        data_dir=data_dir,
        params={
            "scale": solar_scale_factor,
            "start_time": chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_forecast": False
        }
    )

    solar_signal.now(at=chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"), column=location)


    # --- Carbon Intensity Signal from WattTime Dataset
    carbon_signal = vs.HistoricalSignal.load(
        "watttime2023_caiso-north",
        data_dir=data_dir,
        params={
            "start_time": chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_forecast": False
        }
    )

    carbon_signal.now(at=chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # --- Battery Setup
    battery = vs.ClcBattery(
        number_of_cells=int(battery_capacity / 3.63),
        initial_soc=initial_soc,
        min_soc=battery_min_soc
    )

    # --- Initialize Vessim Environment
    env = vs.Environment(sim_start=chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # --- Attach Monitor
    if not output_file:
        raise ValueError("output_file path must be provided for Vessim output.")
    monitor = vs.Monitor(outfile=output_file)

    # --- Add Microgrid
    env.add_microgrid(
        actors=[
            vs.Actor(name="vidur_power_usage", signal=power_signal),
            vs.Actor(name="solar", signal=solar_signal),
            vs.Actor(name="carbon_intensity", signal=carbon_signal),
        ],
        controllers=[monitor],
        storage=battery,
        step_size=step_size
    )

    # --- Run Simulation
    env.run(until=duration_sec)

    # --- Read Monitor Output
    chunk_vessim_df = pd.read_csv(output_file, parse_dates=["time"], index_col="time")
    if chunk_vessim_df.empty:
        print("No logs from Vessim monitor; returning empty DataFrame.")
        return pd.DataFrame(), battery.soc(), 0.0

    # --- Compute Average Solar Power
    avg_solar_power = chunk_vessim_df.get("solar.p", pd.Series([0.0])).mean()
    final_battery_soc = battery.soc()

    return chunk_vessim_df, final_battery_soc, avg_solar_power