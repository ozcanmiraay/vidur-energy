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

    parser.add_argument(
        "--vidur-sim-dir",
        required=True,
        help="Path to an already-ran Vidur simulation directory",
    )
    parser.add_argument(
        "--location",
        default="San Francisco",
        choices=LOCATION_TIMEZONE_MAP.keys(),
        help="Simulation location",
    )
    parser.add_argument(
        "--agg-freq",
        default="1min",
        help="Aggregation frequency (e.g., 1s, 1min, 5min)",
    )
    parser.add_argument(
        "--analysis-type",
        choices=["trend analysis", "total power analysis"],
        default="trend analysis",
    )
    parser.add_argument(
        "--step-size", type=int, default=60, help="Simulation step size in seconds"
    )
    parser.add_argument(
        "--solar-scale-factor", type=int, default=5000, help="Solar scaling factor"
    )
    parser.add_argument(
        "--battery-capacity", type=int, default=5000, help="Battery capacity in Wh"
    )
    parser.add_argument(
        "--battery-initial-soc",
        type=float,
        default=0.4,
        help="Initial battery state of charge (0-1)",
    )
    parser.add_argument(
        "--battery-min-soc",
        type=float,
        default=0.3,
        help="Minimum battery state of charge (0-1)",
    )
    parser.add_argument(
        "--log-metrics", action="store_true", help="Enable detailed energy logging"
    )
    parser.add_argument(
        "--carbon-analysis",
        action="store_true",
        help="Enable carbon emissions analysis",
    )
    parser.add_argument(
        "--low-carbon-threshold",
        type=float,
        default=100,
        help="Threshold for low carbon intensity in gCO2/kWh",
    )

    args = parser.parse_args()

    location_timezone = pytz.timezone(LOCATION_TIMEZONE_MAP[args.location])
    vessim_output_dir = os.path.join(args.vidur_sim_dir, "vessim_analysis")
    os.makedirs(vessim_output_dir, exist_ok=True)

    # 1) Extract the per-request power usage from Vidur
    csv_file = extract_vidur_stats(args.vidur_sim_dir)
    if csv_file is None:
        print("❌ Could not extract or locate mfu_energy_power_data.csv. Exiting.")
        return

    # 2) Prepare data for Vessim
    processed_file = os.path.join(vessim_output_dir, "vessim_ready_data.csv")
    vessim_ready_data = prepare_vessim_data(
        csv_file,
        agg_freq=args.agg_freq,
        analysis_type=args.analysis_type,
        output_file=processed_file,
    )

    # 3) Use the first and last timestamps of the aggregated data as our sim_start & sim_end
    sim_start_time = vessim_ready_data.index[0]
    sim_end_time = vessim_ready_data.index[-1]

    print(f"Raw simulation start_time in data: {sim_start_time}")
    print(f"Raw simulation end_time in data: {sim_end_time}")

    # print(
    #     f"Interpreting these times as local time in {args.location} => {location_timezone.zone}"
    # )

    # # If the timestamps are naive, treat them as local times
    # if sim_start_time.tzinfo is None:
    #     # Step A: localize them to the location
    #     local_start_aware = location_timezone.localize(sim_start_time)
    #     local_end_aware = location_timezone.localize(sim_end_time)

    #     # Step B: Convert to UTC
    #     start_utc = local_start_aware.astimezone(pytz.utc)
    #     end_utc = local_end_aware.astimezone(pytz.utc)

    #     # Step C: Remove the timezone entirely
    #     sim_start_time_utc_naive = start_utc.replace(tzinfo=None)
    #     sim_end_time_utc_naive = end_utc.replace(tzinfo=None)

    #     print("Converted sim_start_time => UTC naive:", sim_start_time_utc_naive)
    #     print("Converted sim_end_time   => UTC naive:", sim_end_time_utc_naive)

    #     # Overwrite the times in code
    #     sim_start_time = sim_start_time_utc_naive
    #     sim_end_time = sim_end_time_utc_naive
    # else:
    #     # If they already have tzinfo, ensure we do the same steps
    #     # (but this scenario is less common in your setup)
    #     sim_start_time = sim_start_time.astimezone(pytz.utc).replace(tzinfo=None)
    #     sim_end_time = sim_end_time.astimezone(pytz.utc).replace(tzinfo=None)

    # print(
    #     f"Final simulation time used by Vessim (UTC naive): {sim_start_time} -> {sim_end_time}"
    # )

    # 4) Now run the Vessim simulation with these naive-UTC times
    sim_output_file = os.path.join(vessim_output_dir, "vessim_output.csv")
    run_vessim_simulation(
        data=vessim_ready_data,
        sim_start_time=sim_start_time,
        sim_end_time=sim_end_time,
        step_size=args.step_size,
        solar_scale=args.solar_scale_factor,
        battery_capacity=args.battery_capacity,
        battery_initial_soc=args.battery_initial_soc,
        battery_min_soc=args.battery_min_soc,
        output_file=sim_output_file,
        analysis_type=args.analysis_type,
        location=args.location,
    )

    # 5) Post-process and visualize results
    plot_vessim_results(
        output_file=sim_output_file,
        step_size=args.step_size,
        save_dir=vessim_output_dir,
        location_tz=location_timezone,  # for converting back to local time in plots
        log_metrics=args.log_metrics or args.carbon_analysis,
    )


if __name__ == "__main__":
    main()
