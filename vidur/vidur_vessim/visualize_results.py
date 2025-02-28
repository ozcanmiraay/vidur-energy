import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_vessim_results(output_file, step_size=60, save_dir="vessim_analysis"):
    """Plots Vessim results including power usage and battery SOC, with improved time formatting."""
    
    df = pd.read_csv(output_file, parse_dates=["time"], index_col="time")

    # Convert e_delta to power (W)
    df["grid_power"] = df["e_delta"] / step_size  

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    ## 🟢 **Plot 1: Power Usage & Generation**  
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df.index, df["solar.p"], color="gold", label="Solar Power (W)")
    ax1.plot(df.index, df["vidur_power_usage.p"], linestyle="dashed", color="red", label="Power Demand (W)")
    ax1.plot(df.index, df["grid_power"], linestyle="dotted", color="blue", label="Grid Power (W)")

    ax1.set_ylabel("Power (W)")
    ax1.set_xlabel("Time")
    ax1.set_title("Power Usage and Solar Generation Over Time")

    ax1.legend()
    ax1.grid()

    # 🔥 **Improved Time Formatting (Every 3 Hours)**
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Show every 3 hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Display only hours:minutes
    fig.autofmt_xdate()  # Auto-adjust layout to prevent overlap

    plt.savefig(os.path.join(save_dir, "power_plot.png"))
    plt.close()

    ## 🟢 **Plot 2: Battery State of Charge (SOC)**
    if "storage.soc" in df.columns:
        fig, ax2 = plt.subplots(figsize=(12, 4))

        ax2.plot(df.index, df["storage.soc"], color="green", label="Battery SOC (%)")

        ax2.set_ylabel("State of Charge (%)")
        ax2.set_xlabel("Time")
        ax2.set_title("Battery State of Charge Over Time")
        
        ax2.legend()
        ax2.grid()

        # 🔥 **Improved Time Formatting (Every 3 Hours)**
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Show every 3 hours
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Display only hours:minutes
        fig.autofmt_xdate()  # Auto-adjust layout

        plt.savefig(os.path.join(save_dir, "battery_soc_plot.png"))
        plt.close()

    print("✅ Plots saved successfully!")