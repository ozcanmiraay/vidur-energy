import pandas as pd
import vessim as vs
import numpy as np


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
):
    duration_seconds = int((sim_end_time - sim_start_time).total_seconds())
    environment = vs.Environment(sim_start=sim_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    power_signal = vs.HistoricalSignal(data[["power_usage_watts"]])
    solar_signal = vs.HistoricalSignal.load(
        "solcast2022_global",
        column=location,
        params={"scale": solar_scale, "start_time": sim_start_time},
    )

    carbon_intensity_signal = vs.HistoricalSignal.load(
        "watttime2023_caiso-north", params={"start_time": sim_start_time}
    )

    battery = vs.ClcBattery(
        number_of_cells=int(battery_capacity / 3.63),
        initial_soc=battery_initial_soc,
        min_soc=battery_min_soc,
    )

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
