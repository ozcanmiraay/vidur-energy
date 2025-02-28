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
    parser.add_argument("--scale-factor", type=int, default=5000)
    parser.add_argument("--capacity", type=int, default=50000, help="Battery capacity in Wh")
    parser.add_argument("--interpolate", type=bool, default=True, help="Enable interpolation")

    args = parser.parse_args()

    # Get the corresponding timezone
    location_timezone = pytz.timezone(LOCATION_TIMEZONE_MAP[args.location])

    # Define the Vessim output directory inside the Vidur simulation result directory
    vessim_output_dir = os.path.join(args.vidur_sim_dir, "vessim_analysis")
    os.makedirs(vessim_output_dir, exist_ok=True)

    # Step 1: Extract stats and locate CSV
    csv_file = extract_vidur_stats(args.vidur_sim_dir)
    if csv_file is None:
        print("❌ Could not extract or locate mfu_energy_power_data.csv. Exiting.")
        return

    # Step 2: Convert to Vessim-ready format
    processed_file = os.path.join(vessim_output_dir, "vessim_ready_data.csv")
    print("🔄 Processing data for Vessim...")
    vessim_ready_data = prepare_vessim_data(csv_file, args.agg_freq, args.analysis_type, args.interpolate, processed_file)

    # Step 3: Apply Proper Timezone Conversion (EXACTLY LIKE YOUR ORIGINAL LOGIC)
    sim_start_time = vessim_ready_data.index[0]
    sim_end_time = vessim_ready_data.index[-1]

    # Original logic preserved:
    local_time = sim_start_time.tz_localize("UTC")  # Assume it's initially UTC
    converted_time = local_time.tz_convert(location_timezone)  # Convert to local timezone
    utc_naive_time = converted_time.tz_localize(None)  # Remove timezone info but keep UTC time

    # Same for end time
    local_end_time = sim_end_time.tz_localize("UTC")
    converted_end_time = local_end_time.tz_convert(location_timezone)
    utc_naive_end_time = converted_end_time.tz_localize(None)

    print(f"⏳ Simulation Start (Local {args.location}): {converted_time}")
    print(f"🌍 Converted UTC Time for Solcast: {utc_naive_time}")

    # Step 4: Run Vessim simulation
    sim_output_file = os.path.join(vessim_output_dir, "vessim_output.csv")
    print("🚀 Running Vessim simulation...")
    run_vessim_simulation(
        vessim_ready_data, utc_naive_time, utc_naive_end_time, args.step_size, args.scale_factor, sim_output_file, args.analysis_type
    )

    # Step 5: Generate and save visualization inside the same folder
    print("📈 Saving plots to", vessim_output_dir)
    plot_vessim_results(sim_output_file, args.step_size, save_dir=vessim_output_dir)

if __name__ == "__main__":
    main()