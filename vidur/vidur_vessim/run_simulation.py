import pandas as pd
import vessim as vs

def run_vessim_simulation(
    data, sim_start_time, sim_end_time, step_size=60, scale_factor=5000, output_file="vessim_output.csv", analysis_type="trend analysis", location="Berlin"
):
    """Runs the Vessim simulation with processed data."""
    
    # Ensure `data` is a DataFrame
    if isinstance(data, str):
        data = pd.read_csv(data, parse_dates=["time_extended"], index_col="time_extended")

    duration_seconds = int((sim_end_time - sim_start_time).total_seconds())

    environment = vs.Environment(sim_start=sim_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    power_signal = vs.HistoricalSignal(data[["power_usage_watts"]])

    solar_signal = vs.HistoricalSignal.load(
        "solcast2022_global",
        column=location,
        params={"scale": scale_factor, "start_time": sim_start_time}
    )

    environment.add_microgrid(
        actors=[
            vs.Actor(name="vidur_power_usage", signal=power_signal),
            vs.Actor(name="solar", signal=solar_signal),
        ],
        controllers=[vs.Monitor(outfile=output_file)],
        step_size=step_size,
    )

    environment.run(until=duration_seconds)