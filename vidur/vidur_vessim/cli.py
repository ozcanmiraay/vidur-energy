import pytz
import argparse
import os
import pandas as pd
from datetime import datetime
from vidur.vidur_vessim.extract_vidur_stats import extract_vidur_stats
from vidur.vidur_vessim.prepare_data import prepare_vessim_data
from vidur.vidur_vessim.run_simulation import run_vessim_simulation
from vidur.vidur_vessim.visualize_results import plot_vessim_results

# Mapping locations to their timezones
LOCATION_TIMEZONE_MAP = {
    "Berlin": "Europe/Berlin",
    "Cape Town": "Africa/Johannesburg",
    "Hong Kong": "Asia/Hong_Kong",
    "Lagos": "Africa/Lagos",
    "Mexico City": "America/Mexico_City",
    "Mumbai": "Asia/Kolkata",
    "San Francisco": "America/Los_Angeles",
    "Stockholm": "Europe/Stockholm",
    "Sydney": "Australia/Sydney",
    "São Paulo": "America/Sao_Paulo",
}

def main():
    parser = argparse.ArgumentParser(description="Vidur to Vessim Co-Simulation")

    parser.add_argument("--vidur-sim-dir", required=True, help="Path to an already-ran Vidur simulation directory")
    parser.add_argument("--location", required=True, choices=LOCATION_TIMEZONE_MAP.keys(), help="Simulation location")
    parser.add_argument("--agg-freq", default="1min", help="Aggregation frequency (e.g., 1s, 1min, 5min)")
    parser.add_argument("--analysis-type", choices=["trend analysis", "total power analysis"], default="trend analysis")
    parser.add_argument("--step-size", type=int, default=60, help="Simulation step size in seconds")
    parser.add_argument("--solar-scale-factor", type=int, default=5000, help="Solar scaling factor")
    parser.add_argument("--battery-capacity", type=int, default=5000, help="Battery capacity in Wh")
    parser.add_argument("--battery-initial-soc", type=float, default=0.4, help="Initial battery state of charge (0-1)")
    parser.add_argument("--battery-min-soc", type=float, default=0.3, help="Minimum battery state of charge (0-1)")
    parser.add_argument("--log-metrics", action="store_true", help="Enable detailed energy logging")
    parser.add_argument("--carbon-analysis", action="store_true", 
                       help="Enable carbon emissions analysis")
    parser.add_argument("--low-carbon-threshold", type=float, default=100,
                       help="Threshold for low carbon intensity in gCO2/kWh")

    args = parser.parse_args()

    location_timezone = pytz.timezone(LOCATION_TIMEZONE_MAP[args.location])
    vessim_output_dir = os.path.join(args.vidur_sim_dir, "vessim_analysis")
    os.makedirs(vessim_output_dir, exist_ok=True)

    csv_file = extract_vidur_stats(args.vidur_sim_dir)
    if csv_file is None:
        print("❌ Could not extract or locate mfu_energy_power_data.csv. Exiting.")
        return

    processed_file = os.path.join(vessim_output_dir, "vessim_ready_data.csv")
    vessim_ready_data = prepare_vessim_data(csv_file, args.agg_freq, args.analysis_type, processed_file)

    # Get simulation times in UTC
    sim_start_time = vessim_ready_data.index[0]
    sim_end_time = vessim_ready_data.index[-1]

    # For debugging
    print(f"Raw simulation time: {sim_start_time}")
    print(f"Location timezone: {location_timezone}")
    print(f"Location offset from UTC: {location_timezone.utcoffset(sim_start_time)}")

    # Convert to UTC if not already
    if sim_start_time.tzinfo is None:
        sim_start_time = sim_start_time.tz_localize('UTC')
        sim_end_time = sim_end_time.tz_localize('UTC')

    print(f"Final simulation time (UTC): {sim_start_time}")

    sim_output_file = os.path.join(vessim_output_dir, "vessim_output.csv")
    run_vessim_simulation(
        vessim_ready_data, 
        sim_start_time.tz_localize(None),  # Vessim needs naive UTC
        sim_end_time.tz_localize(None),    # Vessim needs naive UTC
        args.step_size, 
        args.solar_scale_factor,
        args.battery_capacity, 
        args.battery_initial_soc, 
        args.battery_min_soc,
        sim_output_file, 
        args.analysis_type, 
        args.location
    )

    # Only pass timezone for display purposes
    plot_vessim_results(
        output_file=sim_output_file, 
        step_size=args.step_size, 
        save_dir=vessim_output_dir,
        location_tz=location_timezone,
        log_metrics=args.log_metrics or args.carbon_analysis  # Enable logging if either flag is set
    )

if __name__ == "__main__":
    main()