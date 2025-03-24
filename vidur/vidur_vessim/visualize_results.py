import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta


def calculate_carbon_emissions(df, step_size):
    """
    Calculate carbon emissions and offsets from grid power usage and solar generation.

    Returns DataFrame with:
    - gross_emissions: Total emissions from power consumption
    - renewable_offset: Emissions avoided by using solar power
    - net_emissions: Actual carbon footprint after renewable offset
    """
    # Convert power from W to kW and time step to hours
    hour_fraction = step_size / 3600

    # Calculate total power consumption emissions
    power_consumption_kw = abs(df["vidur_power_usage.p"]) / 1000
    gross_emissions = power_consumption_kw * df["carbon_intensity.p"] * hour_fraction

    # Calculate emissions offset by solar generation
    solar_generation_kw = df["solar.p"] / 1000
    renewable_offset = solar_generation_kw * df["carbon_intensity.p"] * hour_fraction

    # Net emissions = what would have been emitted - what was offset by renewables
    net_emissions = gross_emissions - renewable_offset

    return pd.DataFrame(
        {
            "gross_emissions": gross_emissions,
            "renewable_offset": renewable_offset,
            "net_emissions": net_emissions,
        }
    )


def format_emissions(emissions_value):
    """Format emissions value in appropriate unit (kg or g)"""
    if abs(emissions_value) >= 1000:
        return f"{emissions_value/1000:.2f} kgCO₂"
    return f"{emissions_value:.1f} gCO₂"


def plot_vessim_results(
    output_file,
    step_size=60,
    save_dir="vessim_analysis",
    location_tz=None,
    log_metrics=False,
    analysis_type="trend analysis",
):
    """Plots Vessim results, including power usage, battery SOC, and carbon emissions."""

    # Load data (assuming UTC)
    df = pd.read_csv(output_file, parse_dates=["time"], index_col="time")

    # Make sure index is timezone aware
    df.index = df.index.tz_localize("UTC")

    if location_tz:
        # Convert index to local time
        df.index = df.index.tz_convert(location_tz)
        print(f"Data range in {location_tz.zone}: {df.index[0]} to {df.index[-1]}")

    df["grid_power"] = df["e_delta"] / step_size

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "simulation_metrics.txt")

    ## Initialize Log File
    if log_metrics:
        # Start with a fresh log file
        with open(log_path, "w") as log_file:
            log_file.write("📊 VESSIM SIMULATION METRICS\n")
            log_file.write("=" * 50 + "\n\n")

    def format_time_axis(ax, location_tz):
        """Helper function to consistently format time axes"""
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=location_tz))
        ax.tick_params(axis="both", labelsize=10)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
            
    ## Plot: Power Usage & Solar Generation
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("Power Flow Analysis", fontsize=16, y=0.95)

    # Automatically scale to MW if values are large
    if analysis_type == "total power analysis" and df["vidur_power_usage.p"].abs().max() > 1e5:
        scale_factor = 1e6
        y_unit = "MW"
    else:
        scale_factor = 1
        y_unit = "W"

    ax1.fill_between(
        df.index,
        0,
        df["solar.p"] / scale_factor,
        color="gold",
        alpha=0.35,
        label=f"Solar Generation ({y_unit})"
    )
    ax1.plot(
        df.index,
        df["vidur_power_usage.p"] / scale_factor,
        color="red",
        label=f"Power Demand ({y_unit})"
    )
    ax1.plot(
        df.index,
        df["grid_power"] / scale_factor,
        color="blue",
        label=f"Grid Power ({y_unit})"
    )

    ax1.set_ylabel(f"Power ({y_unit})", fontsize=12)
    ax1.set_xlabel(f"Time ({location_tz.zone if location_tz else 'UTC'})", fontsize=12)

    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    format_time_axis(ax1, location_tz)
    plt.tight_layout()

    power_plot_path = os.path.join(save_dir, "power_plot.png")
    plt.savefig(power_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    ## Plot: Battery State of Charge (SOC)
    if "storage.soc" in df.columns:
        fig, ax2 = plt.subplots(figsize=(12, 6))
        fig.suptitle("Battery State of Charge", fontsize=16, y=0.95)

        ax2.plot(
            df.index,
            df["storage.soc"] * 100,
            color="green",
            label="Battery SOC (%)",
            linewidth=2,
        )
        ax2.axhline(
            y=df["storage.min_soc"].iloc[0] * 100,
            color="r",
            linestyle="--",
            label="Min SoC",
            linewidth=1.5,
        )
        ax2.fill_between(df.index, df["storage.soc"] * 100, alpha=0.2, color="green")

        ax2.set_ylabel("State of Charge (%)", fontsize=12)
        ax2.set_xlabel(
            f"Time ({location_tz.zone if location_tz else 'UTC'})", fontsize=12
        )

        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        format_time_axis(ax2, location_tz)
        plt.tight_layout()

        battery_plot_path = os.path.join(save_dir, "battery_soc_plot.png")
        plt.savefig(battery_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Add Battery Usage Distribution Plot
        charging_mask = df["storage.charge_level"].diff() > 0
        discharging_mask = df["storage.charge_level"].diff() < 0
        idle_mask = df["storage.charge_level"].diff() == 0

        charging_time = charging_mask.sum() * step_size / (len(df) * step_size) * 100
        discharging_time = (
            discharging_mask.sum() * step_size / (len(df) * step_size) * 100
        )
        idle_time = idle_mask.sum() * step_size / (len(df) * step_size) * 100

        # Only create pie chart if there's more than one state
        non_zero_states = sum(
            x > 1 for x in [charging_time, discharging_time, idle_time]
        )
        if non_zero_states > 1:
            # Create regular pie chart for multiple states
            fig, ax3 = plt.subplots(figsize=(10, 8))
            colors = ["#3498db", "#e67e22", "#2ecc71"]
            labels = ["Charging", "Discharging", "Idle"]
            sizes = [charging_time, discharging_time, idle_time]
            wedges, texts, autotexts = ax3.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax3.axis("equal")
            plt.title("Battery Usage Distribution", pad=20, y=1.08)
            plt.setp(autotexts, size=10, weight="bold")
            plt.setp(texts, size=12)
        else:
            # Create simplified visualization for single state
            fig, ax3 = plt.subplots(figsize=(8, 4))

            # Determine the active state (using a lower threshold)
            if charging_time > 95:
                state, color = "Charging", "#3498db"
            elif discharging_time > 95:
                state, color = "Discharging", "#e67e22"
            elif idle_time > 95:
                state, color = "Idle", "#2ecc71"
            else:
                # If no single state dominates, fall back to pie chart
                fig, ax3 = plt.subplots(figsize=(10, 8))
                colors = ["#3498db", "#e67e22", "#2ecc71"]
                labels = ["Charging", "Discharging", "Idle"]
                sizes = [charging_time, discharging_time, idle_time]
                wedges, texts, autotexts = ax3.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax3.axis("equal")
                plt.title("Battery Usage Distribution", pad=20, y=1.08)
                plt.setp(autotexts, size=10, weight="bold")
                plt.setp(texts, size=12)
                return

            # Create the single-state visualization
            ax3.text(
                0.5,
                0.5,
                f"{state}\n{max(charging_time, discharging_time, idle_time):.1f}%",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=color,
            )
            ax3.axis("off")
            plt.title("Battery Usage Distribution", pad=20)

        battery_usage_path = os.path.join(save_dir, "battery_usage_distribution.png")
        plt.savefig(battery_usage_path, bbox_inches="tight")
        plt.close()

        # Add these metrics to the log file if logging is enabled
        if log_metrics:
            with open(log_path, "a") as log_file:
                log_file.write("\n🔋 BATTERY USAGE DISTRIBUTION\n")
                log_file.write("-" * 50 + "\n")
                log_file.write(f"⚡ Charging: {charging_time:.1f}%\n")
                log_file.write(f"📉 Discharging: {discharging_time:.1f}%\n")
                log_file.write(f"💤 Idle: {idle_time:.1f}%\n")
                log_file.write("-" * 50 + "\n")

            print(f"\n🔋 Battery Usage Distribution:")
            print(f"⚡ Charging: {charging_time:.1f}%")
            print(f"📉 Discharging: {discharging_time:.1f}%")
            print(f"💤 Idle: {idle_time:.1f}%")

    ## Plot: Carbon Emissions
    if "carbon_intensity.p" in df.columns:
        emissions_df = calculate_carbon_emissions(df, step_size)

        # Calculate totals for metrics
        total_gross = emissions_df["gross_emissions"].sum()
        total_offset = emissions_df["renewable_offset"].sum()
        total_net = emissions_df["net_emissions"].sum()

        # Calculate intensity metrics
        avg_intensity = df["carbon_intensity.p"].mean()
        peak_intensity = df["carbon_intensity.p"].max()
        min_intensity = df["carbon_intensity.p"].min()
        low_carbon_hours = (df["carbon_intensity.p"] < 100).sum() * step_size / 3600
        high_carbon_hours = (df["carbon_intensity.p"] > 200).sum() * step_size / 3600

        if log_metrics:
            with open(log_path, "a") as log_file:
                log_file.write("\n🌍 CARBON EMISSIONS ANALYSIS\n")
                log_file.write("=" * 50 + "\n")
                log_file.write("\n📊 Emissions Summary:\n")
                log_file.write(
                    f"• Total Emissions from Power Usage: {format_emissions(total_gross)}\n"
                )
                log_file.write(
                    f"• Emissions Offset by Solar: {format_emissions(total_offset)}\n"
                )
                log_file.write(
                    f"• Final Carbon Footprint: {format_emissions(total_net)}\n"
                )
                log_file.write(
                    f"• Percentage Offset by Renewables: {(total_offset/total_gross)*100:.1f}%\n"
                )

                log_file.write("\n📈 Carbon Intensity Metrics:\n")
                log_file.write(f"• Average: {avg_intensity:.1f} gCO₂/kWh\n")
                log_file.write(f"• Peak: {peak_intensity:.1f} gCO₂/kWh\n")
                log_file.write(f"• Minimum: {min_intensity:.1f} gCO₂/kWh\n")

                log_file.write("\n⏱️ Time Analysis:\n")
                log_file.write(
                    f"• Low Carbon Hours (<100 gCO₂/kWh): {low_carbon_hours:.1f} hours\n"
                )
                log_file.write(
                    f"• High Carbon Hours (>200 gCO₂/kWh): {high_carbon_hours:.1f} hours\n"
                )
                log_file.write("=" * 50 + "\n")

        # Calculate totals and determine units
        max_emission = max(
            abs(emissions_df["gross_emissions"].cumsum().max()),
            abs(emissions_df["renewable_offset"].cumsum().max()),
        )
        y_scale = 1000 if max_emission >= 1000 else 1
        y_unit = "kg" if max_emission >= 1000 else "g"

        # Create plot with more space between subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        fig.suptitle("Carbon Footprint Analysis", fontsize=16, y=0.95)

        # Top plot - Emissions
        ax1.plot(
            df.index,
            emissions_df["gross_emissions"].cumsum() / y_scale,
            color="#FF6B6B",
            label="Total Emissions",
            linewidth=2,
        )
        ax1.plot(
            df.index,
            emissions_df["renewable_offset"].cumsum() / y_scale,
            color="#4ECB71",
            label="Solar Offset",
            linewidth=2,
        )
        ax1.plot(
            df.index,
            emissions_df["net_emissions"].cumsum() / y_scale,
            color="#9B59B6",
            label="Net Footprint",
            linewidth=2.5,
        )

        # Simplified fill
        ax1.fill_between(
            df.index,
            0,
            emissions_df["net_emissions"].cumsum() / y_scale,
            color="#9B59B6",
            alpha=0.1,
        )

        ax1.set_ylabel(f"Cumulative CO₂ ({y_unit})", fontsize=12)
        ax1.legend(fontsize=10, loc="upper left", framealpha=0.9)
        ax1.grid(True, alpha=0.2)

        # Bottom plot - Carbon Intensity
        ax2.plot(
            df.index,
            df["carbon_intensity.p"],
            color="#E74C3C",
            label="Grid Carbon Intensity",
            linewidth=2,
        )
        ax2.axhline(
            y=100,
            color="#27AE60",
            linestyle="--",
            alpha=0.5,
            label="Low Carbon Threshold",
        )

        ax2.set_ylabel("Grid Carbon Intensity\n(gCO₂/kWh)", fontsize=12)
        ax2.set_xlabel(
            f"Time ({location_tz.zone if location_tz else 'UTC'})", fontsize=12
        )
        ax2.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax2.grid(True, alpha=0.2)

        # Format axes
        format_time_axis(ax1, location_tz)
        format_time_axis(ax2, location_tz)

        # Adjust layout
        plt.subplots_adjust(hspace=0.3)  # Increase space between subplots

        emissions_plot_path = os.path.join(save_dir, "carbon_emissions_plot.png")
        plt.savefig(emissions_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    ## Final System Metrics
    if log_metrics:
        if 'analysis_type' not in locals():
            analysis_type = 'trend analysis'  # fallback default

        if analysis_type == "trend analysis":
            total_demand = abs(df["vidur_power_usage.p"].sum() * step_size / 3600000)
            total_solar = df["solar.p"].sum() * step_size / 3600000
            total_grid = abs(df["grid_power"].sum() * step_size / 3600000)
        else:  # total power analysis
            total_demand = abs(df["vidur_power_usage.p"].sum() / 3600000)
            total_solar = df["solar.p"].sum() / 3600000
            total_grid = abs(df["grid_power"].sum() / 3600000)

        total_renewable_energy = total_solar

        avg_soc = df["storage.soc"].mean() * 100 if "storage.soc" in df.columns else 0
        min_soc_time = (
            (df["storage.soc"] < 0.5).sum() * step_size / 3600
            if "storage.soc" in df.columns
            else 0
        )
        max_soc_time = (
            (df["storage.soc"] > 0.8).sum() * step_size / 3600
            if "storage.soc" in df.columns
            else 0
        )
        battery_cycles = (
            abs(df["storage.charge_level"].diff()).sum()
            / (2 * df["storage.capacity"].iloc[0])
            if "storage.charge_level" in df.columns
            else 0
        )

        with open(log_path, "a") as log_file:
            log_file.write("\n⚡ SYSTEM BALANCE ANALYSIS\n")
            log_file.write("-" * 50 + "\n")
            log_file.write(f"🔴 Total Energy Demand: {total_demand:.2f} kWh\n")
            log_file.write(f"🟡 Total Solar Generation: {total_solar:.2f} kWh\n")
            log_file.write(
                f"🌍 Total Renewable Energy: {total_renewable_energy:.2f} kWh\n"
            )
            log_file.write(f"🔌 Total Grid Energy: {total_grid:.2f} kWh\n")
            log_file.write(
                f"✅ Renewable Penetration: {(total_renewable_energy/total_demand)*100:.1f}%\n"
            )
            log_file.write(
                f"🚧 Grid Dependency: {(total_grid/total_demand)*100:.1f}%\n"
            )

            if "storage.soc" in df.columns:
                log_file.write("\n🔋 Battery Performance:\n")
                log_file.write(f"⚡ Average SoC: {avg_soc:.1f}%\n")
                log_file.write(f"⏳ Time Below 50% SoC: {min_soc_time:.1f} hours\n")
                log_file.write(f"⏫ Time Above 80% SoC: {max_soc_time:.1f} hours\n")
                log_file.write(f"🔄 Estimated Full Cycles: {battery_cycles:.1f}\n")
            log_file.write("-" * 50 + "\n")

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
