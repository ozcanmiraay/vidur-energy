import os
import subprocess


def extract_vidur_stats(vidur_sim_dir):
    """
    Runs stats extraction and fetches the generated energy CSV file.

    Args:
        vidur_sim_dir (str): Path to the already-ran Vidur simulation directory.

    Returns:
        str: Path to the extracted mfu_energy_power_data.csv file.
    """
    analysis_dir = os.path.join(vidur_sim_dir, "analysis")
    csv_path = os.path.join(analysis_dir, "mfu_energy_power_data.csv")

    # Ensure the stats extractor has not already been run
    if not os.path.exists(csv_path):
        print(f"📊 Running stats extraction for {vidur_sim_dir}...")

        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "vidur.config_optimizer.analyzer.stats_extractor_energy",
                    "--sim-results-dir",
                    vidur_sim_dir,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running stats extraction: {e}")
            return None

    else:
        print(f"✅ Stats already extracted. Using existing {csv_path}")

    return csv_path if os.path.exists(csv_path) else None
