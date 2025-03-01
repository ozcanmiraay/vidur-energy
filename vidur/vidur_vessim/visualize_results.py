import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_vessim_results(output_file, step_size=60, save_dir="vessim_analysis", location_tz=None, log_metrics=False):
    """Plots Vessim results, including power usage, battery SOC, and logs key metrics."""

    # Load data (assuming UTC)
    df = pd.read_csv(output_file, parse_dates=["time"], index_col="time")
    
    # Make sure index is timezone aware
    df.index = df.index.tz_localize('UTC')
    
    if location_tz:
        # Convert index to local time
        df.index = df.index.tz_convert(location_tz)
        print(f"Data range in {location_tz.zone}: {df.index[0]} to {df.index[-1]}")
    
    df["grid_power"] = df["e_delta"] / step_size  

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "simulation_metrics.txt")

    ## **Plot: Power Usage & Solar Generation**
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.fill_between(df.index, 0, df["solar.p"], color="gold", alpha=0.35, label="Solar Generation")
    ax1.plot(df.index, df["vidur_power_usage.p"], color="red", label="Power Demand (W)")
    ax1.plot(df.index, df["grid_power"], color="blue", label="Grid Power (W)")

    ax1.set_ylabel("Power (W)")
    ax1.set_xlabel(f"Time ({location_tz.zone if location_tz else 'UTC'})")
    ax1.set_title("Power Flow Analysis Over Time")

    ax1.legend()
    ax1.grid()

    # Use pandas built-in time formatting with timezone
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=location_tz))
    fig.autofmt_xdate()

    power_plot_path = os.path.join(save_dir, "power_plot.png")
    plt.savefig(power_plot_path)
    plt.close()

    ## **Plot: Battery State of Charge (SOC)**
    if "storage.soc" in df.columns:
        fig, ax2 = plt.subplots(figsize=(12, 4))

        ax2.plot(df.index, df["storage.soc"] * 100, color="green", label="Battery SOC (%)", linewidth=1.5)
        ax2.axhline(y=df["storage.min_soc"].iloc[0] * 100, color='r', linestyle='--', label="Min SoC", linewidth=1.2)

        ax2.set_ylabel("State of Charge (%)")
        ax2.set_xlabel(f"Time ({location_tz.zone if location_tz else 'UTC'})")
        ax2.set_title("Battery State of Charge Over Time")

        ax2.legend()
        ax2.grid()

        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))  
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  
        fig.autofmt_xdate()

        battery_plot_path = os.path.join(save_dir, "battery_soc_plot.png")
        plt.savefig(battery_plot_path)
        plt.close()

        # Add Battery Usage Distribution Plot
        charging_mask = df["storage.charge_level"].diff() > 0
        discharging_mask = df["storage.charge_level"].diff() < 0
        idle_mask = df["storage.charge_level"].diff() == 0

        charging_time = charging_mask.sum() * step_size / (len(df) * step_size) * 100
        discharging_time = discharging_mask.sum() * step_size / (len(df) * step_size) * 100
        idle_time = idle_mask.sum() * step_size / (len(df) * step_size) * 100

        # Only create pie chart if there's more than one state
        non_zero_states = sum(x > 1 for x in [charging_time, discharging_time, idle_time])
        if non_zero_states > 1:
            # Create regular pie chart for multiple states
            fig, ax3 = plt.subplots(figsize=(10, 8))
            colors = ['#3498db', '#e67e22', '#2ecc71']
            labels = ['Charging', 'Discharging', 'Idle']
            sizes = [charging_time, discharging_time, idle_time]
            wedges, texts, autotexts = ax3.pie(sizes, 
                                              labels=labels,
                                              colors=colors,
                                              autopct='%1.1f%%',
                                              startangle=90)
            ax3.axis('equal')
            plt.title("Battery Usage Distribution", pad=20, y=1.08)
            plt.setp(autotexts, size=10, weight="bold")
            plt.setp(texts, size=12)
        else:
            # Create simplified visualization for single state
            fig, ax3 = plt.subplots(figsize=(8, 4))
            
            # Determine the active state (using a lower threshold)
            if charging_time > 95:
                state, color = "Charging", '#3498db'
            elif discharging_time > 95:
                state, color = "Discharging", '#e67e22'
            elif idle_time > 95:
                state, color = "Idle", '#2ecc71'
            else:
                # If no single state dominates, fall back to pie chart
                fig, ax3 = plt.subplots(figsize=(10, 8))
                colors = ['#3498db', '#e67e22', '#2ecc71']
                labels = ['Charging', 'Discharging', 'Idle']
                sizes = [charging_time, discharging_time, idle_time]
                wedges, texts, autotexts = ax3.pie(sizes, 
                                                  labels=labels,
                                                  colors=colors,
                                                  autopct='%1.1f%%',
                                                  startangle=90)
                ax3.axis('equal')
                plt.title("Battery Usage Distribution", pad=20, y=1.08)
                plt.setp(autotexts, size=10, weight="bold")
                plt.setp(texts, size=12)
                return

            # Create the single-state visualization
            ax3.text(0.5, 0.5, f"{state}\n{max(charging_time, discharging_time, idle_time):.1f}%", 
                    ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    color=color)
            ax3.axis('off')
            plt.title("Battery Usage Distribution", pad=20)

        battery_usage_path = os.path.join(save_dir, "battery_usage_distribution.png")
        plt.savefig(battery_usage_path, bbox_inches='tight')
        plt.close()

        # Add these metrics to the log file if logging is enabled
        if log_metrics:
            with open(log_path, "a") as log_file:
                log_file.write("\n🔋 --Battery Usage Distribution--\n")
                log_file.write(f"⚡ Charging: {charging_time:.1f}%\n")
                log_file.write(f"📉 Discharging: {discharging_time:.1f}%\n")
                log_file.write(f"💤 Idle: {idle_time:.1f}%\n")

            print(f"\n🔋 Battery Usage Distribution:")
            print(f"⚡ Charging: {charging_time:.1f}%")
            print(f"📉 Discharging: {discharging_time:.1f}%")
            print(f"💤 Idle: {idle_time:.1f}%")

    ## **Log Key Metrics**
    if log_metrics:
        total_demand = abs(df["vidur_power_usage.p"].sum() * step_size / 3600000)
        total_solar = df["solar.p"].sum() * step_size / 3600000
        total_renewable_energy = total_solar
        total_grid = abs(df["grid_power"].sum() * step_size / 3600000)

        avg_soc = df["storage.soc"].mean() * 100
        min_soc_time = (df["storage.soc"] < 0.5).sum() * step_size / 3600  
        max_soc_time = (df["storage.soc"] > 0.8).sum() * step_size / 3600  

        battery_cycles = abs(df["storage.charge_level"].diff()).sum() / (2 * df["storage.capacity"].iloc[0])

        with open(log_path, "w") as log_file:
            log_file.write("⚡ --System Balance Analysis--\n")
            log_file.write(f"🔴 Total Energy Demand: {total_demand:.2f} kWh\n")
            log_file.write(f"🟡 Total Solar Generation: {total_solar:.2f} kWh\n")
            log_file.write(f"🌍 Total Renewable Energy: {total_renewable_energy:.2f} kWh\n")
            log_file.write(f"🔌 Total Grid Energy: {total_grid:.2f} kWh\n")
            log_file.write(f"✅ Renewable Penetration: {(total_renewable_energy/total_demand)*100:.1f}%\n")
            log_file.write(f"🚧 Grid Dependency: {(total_grid/total_demand)*100:.1f}%\n\n")

            log_file.write("🔋 --Battery Performance--\n")
            log_file.write(f"⚡ Average SoC: {avg_soc:.1f}%\n")
            log_file.write(f"⏳ Time Below 50% SoC: {min_soc_time:.1f} hours\n")
            log_file.write(f"⏫ Time Above 80% SoC: {max_soc_time:.1f} hours\n")
            log_file.write(f"🔄 Estimated Full Cycles: {battery_cycles:.1f}\n")

        print("\n📊 --Simulation Metrics Summary--")
        print(f"⚡ Total Energy Demand: {total_demand:.2f} kWh")
        print(f"🌞 Solar Energy: {total_solar:.2f} kWh")
        print(f"🌍 Total Renewable Energy: {total_renewable_energy:.2f} kWh")
        print(f"🔌 Grid Dependency: {(total_grid/total_demand)*100:.1f}%")
        print(f"🔋 Avg Battery SoC: {avg_soc:.1f}%")
        print(f"⏳ Time Below 50% SoC: {min_soc_time:.1f} hours")
        print(f"⏫ Time Above 80% SoC: {max_soc_time:.1f} hours")
        print(f"🔄 Estimated Battery Cycles: {battery_cycles:.1f}")
        print(f"📁 Results saved in {save_dir}")