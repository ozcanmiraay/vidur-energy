import os
from typing import Dict, Any, Tuple
import logging
import pandas as pd
from .data.processor import DataProcessor
from .data.calculator import EnergyMetricsCalculator
from .visualization.plotter import EnergyVisualizer
from .report.generator import ReportGenerator
from .config.gpu_configs import GPU_POWER_CONFIGS
from .config.region_configs import REGIONAL_ENERGY_CONFIGS

logger = logging.getLogger(__name__)


class SimulationEnergyAnalyzer:
    def __init__(self, sim_results_dir: str, region: str):
        self.sim_results_dir = sim_results_dir
        self.region = region.lower()
        self.data_processor = DataProcessor(sim_results_dir)
        self.visualizer = EnergyVisualizer()
        self.report_generator = ReportGenerator(
            os.path.join(sim_results_dir, "analysis")
        )

    def get_configs(self) -> Tuple[str, int]:
        """Get GPU configuration from simulation config."""
        config = self.data_processor.load_config()

        gpu_type = config["cluster_config"]["replica_config"]["device"].lower()
        num_gpus = (
            config["cluster_config"]["num_replicas"]
            * config["cluster_config"]["replica_config"]["num_pipeline_stages"]
        )

        if gpu_type not in GPU_POWER_CONFIGS:
            raise ValueError(
                f"Unsupported GPU type: {gpu_type}. Supported types: {list(GPU_POWER_CONFIGS.keys())}"
            )

        return gpu_type, num_gpus

    def analyze_energy_consumption(self) -> Dict[str, Any]:
        """Perform comprehensive energy analysis."""
        try:
            # Get configurations
            gpu_type, num_gpus = self.get_configs()
            gpu_config = GPU_POWER_CONFIGS[gpu_type]
            region_config = REGIONAL_ENERGY_CONFIGS[self.region]
            calculator = EnergyMetricsCalculator(gpu_config, region_config)

            # Process MFU data
            mfu_df = self.data_processor.process_mfu_data()
            logger.info(f"Processed {len(mfu_df)} MFU data points")

            # Calculate energy metrics for each time point
            energy_data = []
            for _, row in mfu_df.iterrows():
                execution_time = row.get("model_execution_time", row["execution_time"])
                metrics = calculator.calculate_energy(
                    gpu_hours=execution_time / 3600,  # Convert to hours
                    mfu=row["mfu"] / 100.0,
                )
                energy_data.append(
                    {
                        "time": row["time"],
                        "mfu": row["mfu"],
                        "replica": row["replica"],
                        "stage": row["stage"],
                        "batch_id": row["batch_id"],
                        "num_tokens": row["num_tokens"],
                        "batch_size": row["batch_size"],
                        "energy": metrics["energy_kwh"],
                        "energy_efficiency": metrics["energy_efficiency"],
                        "effective_power": metrics["effective_power"],
                    }
                )

            energy_df = pd.DataFrame(energy_data)

            # Calculate regional comparisons
            regional_metrics = {}
            for region_name, region_cfg in REGIONAL_ENERGY_CONFIGS.items():
                temp_calculator = EnergyMetricsCalculator(gpu_config, region_cfg)
                total_metrics = {
                    "energy_kwh": 0,
                    "energy_cost": 0,
                    "carbon_emissions": 0,
                }

                for _, row in mfu_df.iterrows():
                    metrics = temp_calculator.calculate_energy(
                        gpu_hours=row["execution_time"] / 3600, mfu=row["mfu"]
                    )
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]

                regional_metrics[region_name] = total_metrics

            # Generate visualizations
            plots = {
                "energy_time_series": self.visualizer.create_time_series_plot(
                    energy_df
                ),
                "efficiency_scatter": self.visualizer.create_efficiency_plot(energy_df),
                "regional_impact": self.visualizer.create_regional_impact_plot(
                    regional_metrics
                ),
            }

            # Calculate total metrics
            total_metrics = {
                "energy_kwh": energy_df["energy"].sum(),
                "energy_cost": energy_df["energy"].sum()
                * region_config.electricity_cost,
                "carbon_emissions": (
                    energy_df["energy"].sum() * region_config.carbon_intensity
                    + gpu_config.manufacturing_emissions
                    * (energy_df["time"].max() / 3600)
                ),
                "energy_efficiency": energy_df["energy_efficiency"].mean(),
            }

            # Generate report
            config_summary = {
                "gpu_type": gpu_type,
                "num_gpus": num_gpus,
                "region": self.region,
            }

            self.report_generator.generate_summary_report(
                total_metrics, config_summary, regional_metrics, plots
            )

            logger.info("Energy analysis completed successfully")
            return {
                "energy_df": energy_df,
                "total_metrics": total_metrics,
                "regional_metrics": regional_metrics,
                "config": config_summary,
            }

        except Exception as e:
            logger.error(f"Energy analysis failed: {str(e)}")
            raise
