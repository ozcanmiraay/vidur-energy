import argparse
from vidur.config_optimizer.analyzer.stats_extractor_energy_advanced.analyzer import SimulationEnergyAnalyzer

def main():
    parser = argparse.ArgumentParser(
        description='Advanced Energy Consumption Analysis'
    )
    parser.add_argument(
        '--sim-results-dir',
        required=True,
        help='Directory containing simulation results'
    )
    parser.add_argument(
        '--region',
        required=True,
        choices=['california', 'washington'],
        help='Region for energy analysis'
    )
    
    args = parser.parse_args()
    
    analyzer = SimulationEnergyAnalyzer(
        args.sim_results_dir,
        args.region
    )
    analyzer.analyze_energy_consumption()

if __name__ == "__main__":
    main() 