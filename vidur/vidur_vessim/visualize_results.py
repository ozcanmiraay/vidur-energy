import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")  # Optional: for headless environments
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,  # optional: better minus signs
})


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
        return f"{emissions_value/1000:.2f} kgCO2"
    return f"{emissions_value:.1f} gCO2"


def plot_vessim_results(
    output_file,
    step_size=60,
    save_dir="vessim_analysis",
    location_tz=None,
    log_metrics=False,
    carbon_analysis=False,
    analysis_type="trend analysis",
    low_carbon_threshold=100,
    high_carbon_threshold=200,
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


    # If carbon intensity is stored in separate CSV, merge it in
    if carbon_analysis and "carbon_intensity.p" not in df.columns:
        carbon_file = output_file.replace(".csv", "_carbon.csv")
        if os.path.exists(carbon_file):
            # Read carbon data and make timezone consistent
            carbon_df = pd.read_csv(carbon_file, parse_dates=["time"], index_col="time")
            
            # Convert carbon_df index to match df's timezone
            if df.index.tz is not None:
                carbon_df.index = carbon_df.index.tz_localize("UTC").tz_convert(df.index.tz)
            else:
                # If main df is tz-naive, make carbon df naive too
                carbon_df.index = carbon_df.index.tz_localize(None)
                
            df = df.join(carbon_df.rename(columns={"carbon_intensity": "carbon_intensity.p"}), how="left")
            print("🟢 Carbon intensity data merged from separate file.")
        else:
            print(f"⚠️ Expected carbon file not found: {carbon_file}")

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
    fig, ax1 = plt.subplots(figsize=(5.5, 2.8))  # Reduced size to increase font scale

    # Scale to MW if appropriate
    if analysis_type == "total power analysis" and df["vidur_power_usage.p"].abs().max() > 1e5:
        scale_factor = 1e6
        y_unit = "MW"
    else:
        scale_factor = 1
        y_unit = "W"

    # Solar Generation Area
    ax1.fill_between(
        df.index,
        0,
        df["solar.p"] / scale_factor,
        color="gold",
        alpha=0.35,
        label=f"Solar Generation ({y_unit})"
    )

    # Power Demand Line
    ax1.plot(
        df.index,
        df["vidur_power_usage.p"] / scale_factor,
        color="red",
        linewidth=2,
        label=f"Power Demand ({y_unit})"
    )

    # Grid Power Line
    ax1.plot(
        df.index,
        df["grid_power"] / scale_factor,
        color="blue",
        linewidth=2,
        label=f"Grid Power ({y_unit})"
    )

    # Axis Labels
    ax1.set_ylabel(f"Power ({y_unit})", fontsize=12)
    ax1.set_xlabel(f"Time ({location_tz.zone if location_tz else 'UTC'})", fontsize=12)

    # Ticks and Grid
    ax1.tick_params(axis='both', labelsize=11)
    ax1.grid(True, alpha=0.3)

    # Legend
    ax1.legend(fontsize=9, loc="upper left")

    # Time formatting
    format_time_axis(ax1, location_tz)

    plt.tight_layout()
    power_plot_path = os.path.join(save_dir, "power_plot.png")
    plt.savefig(power_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    ## Plot: Battery State of Charge (SOC)
    if "storage.soc" in df.columns:

        # Compute battery state and percent values
        df["battery_state"] = df["storage.charge_level"].diff().apply(
            lambda x: "charging" if x > 0 else ("discharging" if x < 0 else "idle")
        )
        df["hour"] = df.index.hour
        df["soc_percent"] = df["storage.soc"] * 100

        # ✅ Battery usage stats (needed for logging even if we don't plot)
        battery_state_counts = df["battery_state"].value_counts(normalize=True) * 100
        charging_time = battery_state_counts.get("charging", 0.0)
        discharging_time = battery_state_counts.get("discharging", 0.0)
        idle_time = battery_state_counts.get("idle", 0.0)

        # Setup improved aspect ratio (wider and taller)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8.5, 6.5),  # Increased height
            gridspec_kw={"height_ratios": [2, 2]}
        )

        # ----------------------------------------
        # 1. Battery SOC Over Time (Top Plot)
        # ----------------------------------------
        ax_soc = axes[0]
        ax_soc.plot(
            df.index,
            df["soc_percent"],
            color="green",
            label="Battery SOC (%)",
            linewidth=2,
        )
        ax_soc.axhline(
            y=df["storage.min_soc"].iloc[0] * 100,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Min SoC"
        )
        ax_soc.fill_between(df.index, df["soc_percent"], alpha=0.2, color="green")

        ax_soc.set_ylabel("State of Charge (%)", fontsize=16)
        ax_soc.set_xlabel(f"Time ({location_tz.zone if location_tz else 'UTC'})", fontsize=16)
        ax_soc.legend(fontsize=16, loc="best")
        ax_soc.grid(True, alpha=0.3)
        format_time_axis(ax_soc, location_tz)
        ax_soc.tick_params(axis="both", labelsize=16)  # Bigger ticks

        # ----------------------------------------
        # 2. Battery SOC Violin Plot by Hour (Bottom Plot)
        # ----------------------------------------
        sns.set_theme(style="whitegrid", palette="colorblind")
        ax_violin = axes[1]
        sns.violinplot(
            data=df,
            x="hour",
            y="soc_percent",
            inner="box",
            density_norm="width",
            linewidth=1.2,
            color="#81c784",
            ax=ax_violin,
        )
        ax_violin.set_xlabel("Hour of Day", fontsize=16)
        ax_violin.set_ylabel("State of Charge (%)", fontsize=16)
        ax_violin.tick_params(axis="x", labelsize=14)
        ax_violin.tick_params(axis="y", labelsize=18)
        ax_violin.grid(True, linestyle="--", alpha=0.3)

        # Final layout and export
        plt.tight_layout()
        battery_plot_path = os.path.join(save_dir, "battery_soc_plot.png")
        plt.savefig(battery_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

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
    if carbon_analysis and "carbon_intensity.p" in df.columns:
        emissions_df = calculate_carbon_emissions(df, step_size)

        # Calculate totals for metrics
        total_gross = emissions_df["gross_emissions"].sum()
        total_offset = emissions_df["renewable_offset"].sum()
        total_net = emissions_df["net_emissions"].sum()

        # Calculate intensity metrics
        avg_intensity = df["carbon_intensity.p"].mean()
        peak_intensity = df["carbon_intensity.p"].max()
        min_intensity = df["carbon_intensity.p"].min()
        low_carbon_hours = (df["carbon_intensity.p"] < low_carbon_threshold).sum() * step_size / 3600
        high_carbon_hours = (df["carbon_intensity.p"] > high_carbon_threshold).sum() * step_size / 3600

        # Optional logging
        if log_metrics:
            with open(log_path, "a") as log_file:
                log_file.write("\n🌍 CARBON EMISSIONS ANALYSIS\n")
                log_file.write("=" * 50 + "\n")
                log_file.write("\n📊 Emissions Summary:\n")
                log_file.write(f"• Total Emissions from Power Usage: {format_emissions(total_gross)}\n")
                log_file.write(f"• Emissions Offset by Solar: {format_emissions(total_offset)}\n")
                log_file.write(f"• Final Carbon Footprint: {format_emissions(total_net)}\n")
                log_file.write(f"• Percentage Offset by Renewables: {(total_offset/total_gross)*100:.1f}%\n")
                log_file.write("\n📈 Carbon Intensity Metrics:\n")
                log_file.write(f"• Average: {avg_intensity:.1f} gCO2/kWh\n")
                log_file.write(f"• Peak: {peak_intensity:.1f} gCO2/kWh\n")
                log_file.write(f"• Minimum: {min_intensity:.1f} gCO2/kWh\n")
                log_file.write("\n⏱️ Time Analysis:\n")
                log_file.write(f"• Low Carbon Hours (<{low_carbon_threshold} gCO2/kWh): {low_carbon_hours:.1f} hours\n")
                log_file.write(f"• High Carbon Hours (>{high_carbon_threshold} gCO2/kWh): {high_carbon_hours:.1f} hours\n")
                log_file.write("=" * 50 + "\n")

        # Determine scaling
        max_emission = max(
            abs(emissions_df["gross_emissions"].cumsum().max()),
            abs(emissions_df["renewable_offset"].cumsum().max()),
        )
        y_scale = 1000 if max_emission >= 1000 else 1
        y_unit = "kg" if max_emission >= 1000 else "g"

        # Create compact plot for improved font scaling
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), height_ratios=[2, 1])

        # -----------------------
        # Top: Cumulative Emissions
        # -----------------------
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
        ax1.fill_between(
            df.index,
            0,
            emissions_df["net_emissions"].cumsum() / y_scale,
            color="#9B59B6",
            alpha=0.1,
        )

        ax1.set_ylabel(f"Cumulative CO2 ({y_unit})", fontsize=17)
        ax1.legend(fontsize=16, loc="upper left", framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # ⏰ Apply time formatting first
        format_time_axis(ax1, location_tz)

        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(15)

        # -----------------------
        # Bottom: Carbon Intensity Over Time
        # -----------------------
        ax2.plot(
            df.index,
            df["carbon_intensity.p"],
            color="#E74C3C",
            label="Grid Carbon Intensity",
            linewidth=2,
        )
        ax2.axhline(
            y=low_carbon_threshold,
            color="#27AE60",
            linestyle="--",
            alpha=0.6,
            label=f"Low Carbon Threshold ({low_carbon_threshold} gCO2/kWh)",
        )
        ax2.axhline(
            y=high_carbon_threshold,
            color="#C0392B",
            linestyle="--",
            alpha=0.6,
            label=f"High Carbon Threshold ({high_carbon_threshold} gCO2/kWh)",
        )

        ax2.set_ylabel("Carbon Intensity\n(gCO2/kWh)", fontsize=18)
        ax2.set_xlabel(f"Time ({location_tz.zone if location_tz else 'UTC'})", fontsize=18)
        ax2.legend(fontsize=12, loc="lower left", framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        # ⏰ Apply time formatting first
        format_time_axis(ax2, location_tz)

        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(15)

        # Final layout and export
        plt.subplots_adjust(hspace=0.4)
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
