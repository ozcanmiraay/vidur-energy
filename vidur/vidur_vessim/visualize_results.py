import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_vessim_results(output_file, step_size=60, save_dir="vessim_analysis"):
    df = pd.read_csv(output_file, parse_dates=["time"], index_col="time")
    df["grid_power"] = df["e_delta"] / step_size

    os.makedirs(save_dir, exist_ok=True)
    df.plot(y=["solar.p", "vidur_power_usage.p", "grid_power"], title="Power Usage", figsize=(12,6))
    plt.savefig(os.path.join(save_dir, "power_plot.png"))
    plt.close()