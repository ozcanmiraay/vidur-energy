# main.py
import datetime
from vidur.vidur_vessim_realtime.realtime_controller import realtime_simulation_loop

if __name__ == "__main__":
    # 1. Set your base output dir
    base_output_dir = "/Users/mirayozcan/Desktop/vidur_copy/vidur/simulator_output/vidur-vessim-realtime"

    # 2. Set your simulation start/end times
    simulation_start_time = datetime.datetime(2025, 2, 28, 21, 0, 0)
    simulation_end_time   = simulation_start_time + datetime.timedelta(hours=24)

    # 3. Define the initial Vidur config
    initial_config = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "batch_size": 16,
        "qps": 20
    }

    # 4. Additional Vessim parameters you want to pass
    location = "San Francisco"
    solar_scale_factor = 6000
    battery_capacity = 5000  # Wh
    battery_min_soc = 0.2
    step_size = 60

    realtime_simulation_loop(
        base_output_dir=base_output_dir,
        simulation_start_time=simulation_start_time,
        simulation_end_time=simulation_end_time,
        initial_vidur_config=initial_config,
        initial_battery_soc=0.8,
        chunk_size_requests=1000,
        chunk_id_start=0,
        global_agg_freq="1min",

        # Now also pass the new parameters for Vessim
        location=location,
        solar_scale_factor=solar_scale_factor,
        battery_capacity=battery_capacity,
        battery_min_soc=battery_min_soc,
        vessim_step_size=step_size
    )