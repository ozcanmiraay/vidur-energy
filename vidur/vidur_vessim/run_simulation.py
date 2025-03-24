import pandas as pd
import numpy as np
import vessim as vs


def run_vessim_simulation(
    data,
    sim_start_time,
    sim_end_time,
    step_size,
    solar_scale,
    battery_capacity,
    battery_initial_soc,
    battery_min_soc,
    output_file,
    analysis_type,
    location,
    agg_freq
):
    duration_seconds = int((sim_end_time - sim_start_time).total_seconds())

    environment = vs.Environment(sim_start=sim_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Power signal (already aggregated in prepare_vessim_data)
    power_signal = vs.HistoricalSignal(data[["power_usage_watts"]])

    # Carbon intensity signal (always the same)
    carbon_intensity_signal = vs.HistoricalSignal.load(
        "watttime2023_caiso-north", params={"start_time": sim_start_time}
    )

    # --- Handle solar signal based on analysis type ---
    if analysis_type == "total power analysis":
        # Load base solar signal from Vessim dataset
        raw_solar_signal = vs.HistoricalSignal.load(
            "solcast2022_global",
            column=location,
            params={"scale": solar_scale, "start_time": sim_start_time},
        )

        # Build a DataFrame from the raw solar signal
        solar_times = pd.to_datetime(raw_solar_signal._actual[location][0])
        solar_values = raw_solar_signal._actual[location][1]
        solar_df = pd.DataFrame({"solar_power": solar_values}, index=solar_times)

        if "batch_stage_count" in data.columns:
            # Match indexes by aggregating batch_stage_count to match solar_df frequency
            batch_stage_count_aligned = data["batch_stage_count"].resample(agg_freq).sum()

            # Align batch_stage_count with solar_df index
            solar_df["batch_stage_count"] = batch_stage_count_aligned.reindex(
                solar_df.index, method="nearest"
            ).ffill().fillna(1)

            solar_df["adjusted_solar_power"] = (
                solar_df["solar_power"] * solar_df["batch_stage_count"]
            )
        else:
            # If batch_stage_count missing, fallback gracefully
            print("⚠️ 'batch_stage_count' column missing in data. Skipping adjustment.")
            solar_df["adjusted_solar_power"] = solar_df["solar_power"]

        solar_signal = vs.HistoricalSignal(solar_df["adjusted_solar_power"])

    else:  # "trend analysis"
        solar_signal = vs.HistoricalSignal.load(
            "solcast2022_global",
            column=location,
            params={"scale": solar_scale, "start_time": sim_start_time},
        )

    # --- Battery setup ---
    battery = vs.ClcBattery(
        number_of_cells=int(battery_capacity / 3.63),
        initial_soc=battery_initial_soc,
        min_soc=battery_min_soc,
    )

    # --- Add all components to the simulation environment ---
    environment.add_microgrid(
        actors=[
            vs.Actor(name="vidur_power_usage", signal=power_signal),
            vs.Actor(name="solar", signal=solar_signal),
            vs.Actor(name="carbon_intensity", signal=carbon_intensity_signal),
        ],
        controllers=[vs.Monitor(outfile=output_file)],
        storage=battery,
        step_size=step_size,
    )

    # --- Run the simulation ---
    environment.run(until=duration_seconds)
    print(f"✅ Vessim simulation complete: {duration_seconds} seconds.")