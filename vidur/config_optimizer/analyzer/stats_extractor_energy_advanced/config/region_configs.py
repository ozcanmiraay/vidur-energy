from dataclasses import dataclass

@dataclass
class RegionalEnergyConfig:
    """Configuration for regional energy parameters."""
    region: str
    pue: float
    carbon_intensity: float  # gCO2eq/kWh
    electricity_cost: float  # $/kWh

# Define regional energy configurations
REGIONAL_ENERGY_CONFIGS = {
    "california": RegionalEnergyConfig(
        region="California",
        pue=1.2,
        carbon_intensity=350.861,
        electricity_cost=0.18
    ),
    "washington": RegionalEnergyConfig(
        region="Washington",
        pue=1.15,
        carbon_intensity=260.0,
        electricity_cost=0.10
    ),
} 