import pandas as pd
import json
import os
import glob
from typing import Dict, List, Optional
import yaml


class DataProcessor:
    def __init__(self, sim_results_dir: str):
        self.sim_results_dir = sim_results_dir

    def process_mfu_data(self) -> pd.DataFrame:
        """Process MFU data from simulation results."""
        mfu_files = glob.glob(os.path.join(self.sim_results_dir, "plots/*_mfu.json"))

        all_data = []
        for file_path in mfu_files:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract replica and stage from filename
            filename = os.path.basename(file_path)
            replica_stage = filename.replace("_mfu.json", "").split("_")
            replica = int(replica_stage[1])
            stage = int(replica_stage[3])

            # Get the MFU distribution data
            mfu_distribution = data[f"replica_{replica}_stage_{stage}_mfu_distribution"]

            # Process each data point in the distribution
            for data_entry in mfu_distribution:
                # Only process entries with batch data (non-zero MFU entries)
                if "batch_id" in data_entry:
                    batch_data = {
                        "time": data_entry["time"],
                        "mfu": data_entry["mfu"],
                        "execution_time": data_entry["execution_time"],
                        "model_execution_time": data_entry.get(
                            "model_execution_time", data_entry["execution_time"]
                        ),
                        "batch_size": data_entry["batch_size"],
                        "num_tokens": data_entry["num_tokens"],
                        "replica": replica,
                        "stage": stage,
                        "batch_id": data_entry["batch_id"],
                    }
                    all_data.append(batch_data)

        return pd.DataFrame(all_data)

    def load_config(self) -> Dict:
        """Load simulation configuration."""
        config_file = None
        for ext in [".json", ".yml", ".yaml"]:
            test_file = os.path.join(self.sim_results_dir, f"config{ext}")
            if os.path.exists(test_file):
                config_file = test_file
                break

        if not config_file:
            raise FileNotFoundError("No config file found in simulation directory")

        with open(config_file, "r") as f:
            if config_file.endswith(".json"):
                return json.load(f)
            return yaml.safe_load(f)
