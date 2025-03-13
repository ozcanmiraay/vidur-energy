import pandas as pd


def prepare_vessim_data(
    stats_file,
    agg_freq="1s",
    analysis_type="trend analysis",
    output_file="vessim_ready_data.csv",
):
    """
    Processes Vidur simulation output into Vessim-compatible format and saves it.
    """
    vidur_data = pd.read_csv(stats_file)
    vidur_data["time_extended"] = pd.to_datetime(vidur_data["time_extended"])
    vidur_data.set_index("time_extended", inplace=True)

    vessim_ready_data = vidur_data.rename(
        columns={
            "effective_power": "power_usage_watts",
            "energy": "energy_usage_joules",
            "gpu_hrs": "gpu_hours",
            "mfu": "model_flop_utilization",
        }
    )

    vessim_ready_data["power_usage_watts"] *= -1  # Represent power consumption

    def weighted_avg(data, weight_col):
        return (data["model_flop_utilization"] * data[weight_col]).sum() / data[
            weight_col
        ].sum()

    if analysis_type == "trend analysis":
        aggregated = vessim_ready_data.resample(agg_freq).agg(
            {
                "power_usage_watts": "mean",
                "energy_usage_joules": "mean",
                "gpu_hours": "mean",
                "model_flop_utilization": "mean",
            }
        )
    elif analysis_type == "total power analysis":
        aggregated = vessim_ready_data.resample(agg_freq).agg(
            {
                "power_usage_watts": "sum",
                "energy_usage_joules": "sum",
                "gpu_hours": "sum",
                "model_flop_utilization": "mean",
            }
        )
        aggregated["batch_stage_count"] = vessim_ready_data.resample(agg_freq)[
            "power_usage_watts"
        ].count()

    aggregated["model_flop_utilization"] = vessim_ready_data.resample(agg_freq).apply(
        lambda x: weighted_avg(x, weight_col="power_usage_watts")
    )

    # Save the processed data to CSV
    aggregated.to_csv(output_file)

    # Return the actual DataFrame, not just the file path
    return aggregated
