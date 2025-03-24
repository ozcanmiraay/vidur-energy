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

    # Represent power demand as negative
    vessim_ready_data["power_usage_watts"] *= -1

    def weighted_avg(group, value_col, weight_col):
        weights = group[weight_col].fillna(0)
        values = group[value_col].fillna(0)
        total_weight = weights.sum()
        if total_weight == 0:
            return 0
        return (values * weights).sum() / total_weight

    if analysis_type == "trend analysis":
        # Mean aggregation for trend analysis
        aggregated = vessim_ready_data.resample(agg_freq).agg(
            {
                "power_usage_watts": "mean",
                "energy_usage_joules": "mean",
                "gpu_hours": "mean",
            }
        )
        # Weighted MFU
        aggregated["model_flop_utilization"] = vessim_ready_data.resample(agg_freq).apply(
            lambda x: weighted_avg(x, "model_flop_utilization", "power_usage_watts")
        )

    elif analysis_type == "total power analysis":
        # Sum-based aggregation for total power analysis
        aggregated = vessim_ready_data.resample(agg_freq).agg(
            {
                "power_usage_watts": "sum",
                "energy_usage_joules": "sum",
                "gpu_hours": "sum",
            }
        )
        # Weighted MFU
        aggregated["model_flop_utilization"] = vessim_ready_data.resample(agg_freq).apply(
            lambda x: weighted_avg(x, "model_flop_utilization", "power_usage_watts")
        )
        # Add batch stage count
        aggregated["batch_stage_count"] = vessim_ready_data.resample(agg_freq)[
            "power_usage_watts"
        ].count()

    # Save to CSV and return
    aggregated.to_csv(output_file)
    return aggregated