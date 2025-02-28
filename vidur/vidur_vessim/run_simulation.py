import pandas as pd
import vessim as vs

def run_vessim_simulation(
    data, sim_start_time, sim_end_time, step_size=60, scale_factor=5000, output_file="vessim_output.csv", 
    analysis_type="trend analysis", location="Berlin"
):
    """Runs the Vessim simulation with processed data, ensuring correct time translation."""
    
    # Ensure `data` is a DataFrame
    if isinstance(data, str):
        data = pd.read_csv(data, parse_dates=["time_extended"], index_col="time_extended")

    duration_seconds = int((sim_end_time - sim_start_time).total_seconds())

    # Correct timezone localization
    sim_start_time = sim_start_time.tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)
    sim_end_time = sim_end_time.tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)

    print(f"📅 Simulation Start Time (Localized): {sim_start_time}")
    print(f"📅 Simulation End Time (Localized): {sim_end_time}")

    sim_start_time_str = sim_start_time.strftime("%Y-%m-%d %H:%M:%S")
    environment = vs.Environment(sim_start=sim_start_time_str)

    # Create power demand signal
    power_signal = vs.HistoricalSignal(data[["power_usage_watts"]])

    # Create solar power generation signal
    solar_signal = vs.HistoricalSignal.load(
        "solcast2022_global",
        column=location,
        params={"scale": scale_factor, "start_time": sim_start_time}
    )

    # Add battery storage as in Colab
    battery = vs.SimpleBattery(capacity=5000, initial_soc=0.8, min_soc=0.2)

    # Define microgrid system with actors, controllers, and storage
    environment.add_microgrid(
        actors=[
            vs.Actor(name="vidur_power_usage", signal=power_signal),
            vs.Actor(name="solar", signal=solar_signal),
        ],
        controllers=[vs.Monitor(outfile=output_file)],
        storage=battery,  # ✅ Battery storage added
        step_size=step_size,
    )

    print("🚀 Running Vessim simulation...")
    environment.run(until=duration_seconds)

    print("✅ Vessim simulation completed successfully!")
