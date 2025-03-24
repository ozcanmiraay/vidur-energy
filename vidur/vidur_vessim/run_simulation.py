import pandas as pd
import vessim as vs

def interpolate_signal(signal, location, step_size, sim_start_time, sim_end_time):
    original_index = pd.to_datetime(signal._actual[location][0])
    values = signal._actual[location][1]
    df = pd.DataFrame({"value": values}, index=original_index)

    full_index = pd.date_range(start=sim_start_time, end=sim_end_time, freq=f"{step_size}s")
    df_interp = df.reindex(full_index)
    df_backup = df_interp.copy()
    df_interp = df_interp.interpolate(method="cubic")
    df_interp.loc[df_backup.notna().all(axis=1)] = df_backup.loc[df_backup.notna().all(axis=1)]

    signal = vs.HistoricalSignal(df_interp["value"])
    return signal, df_interp


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
    agg_freq,
    interpolate_signals=False 
):
    duration_seconds = int((sim_end_time - sim_start_time).total_seconds())
    environment = vs.Environment(sim_start=sim_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # --- Power signal (already resampled upstream) ---
    power_signal = vs.HistoricalSignal(data[["power_usage_watts"]])

    # --- Carbon Intensity Signal (conditionally interpolated) ---
    raw_carbon = vs.HistoricalSignal.load(
        "watttime2023_caiso-north", params={"start_time": sim_start_time}
    )

    if interpolate_signals:
        carbon_intensity_signal, _ = interpolate_signal(
            signal=raw_carbon,
            location="Caiso_North", 
            step_size=step_size,
            sim_start_time=sim_start_time,
            sim_end_time=sim_end_time,
        )
    else:
        carbon_intensity_signal = raw_carbon

    # --- Solar Signal (conditionally interpolated) ---
    raw_solar = vs.HistoricalSignal.load(
        "solcast2022_global",
        column=location,
        params={"scale": solar_scale, "start_time": sim_start_time},
    )

    if interpolate_signals:
        solar_signal_interpolated, solar_interp_df = interpolate_signal(
            signal=raw_solar,
            location=location,
            step_size=step_size,
            sim_start_time=sim_start_time,
            sim_end_time=sim_end_time,
        )
    else:
        solar_signal_interpolated = raw_solar

    # --- Apply batch_stage_count adjustment in total power analysis ---
    if analysis_type == "total power analysis":
        solar_df = pd.DataFrame(
            {"solar_power": solar_interp_df["value"]},
            index=solar_interp_df.index,
        )

        if "batch_stage_count" in data.columns:
            batch_stage_count = data["batch_stage_count"].resample(agg_freq).sum()
            solar_df["batch_stage_count"] = batch_stage_count.reindex(
                solar_df.index, method="nearest"
            ).ffill().fillna(1)
            solar_df["adjusted_solar_power"] = solar_df["solar_power"] * solar_df["batch_stage_count"]
        else:
            print("⚠️ 'batch_stage_count' column missing — skipping adjustment.")
            solar_df["adjusted_solar_power"] = solar_df["solar_power"]

        solar_signal = vs.HistoricalSignal(solar_df["adjusted_solar_power"])

    else:  # "trend analysis"
        solar_signal = solar_signal_interpolated

    # --- Battery Setup ---
    battery = vs.ClcBattery(
        number_of_cells=int(battery_capacity / 3.63),
        initial_soc=battery_initial_soc,
        min_soc=battery_min_soc,
    )

    # --- Add actors and run simulation ---
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

    environment.run(until=duration_seconds)
    print(f"✅ Vessim simulation complete: {duration_seconds} seconds.")