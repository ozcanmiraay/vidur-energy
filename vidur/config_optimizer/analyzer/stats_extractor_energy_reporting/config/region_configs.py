from dataclasses import dataclass


@dataclass
class RegionalEnergyConfig:
    """Configuration for regional energy parameters."""

    region: str
    pue: float  # Power Usage Effectiveness
    carbon_intensity: float  # gCO2eq/kWh
    electricity_cost: float  # $/kWh


# Define regional energy configurations
REGIONAL_ENERGY_CONFIGS = {
    # United States
    "california": RegionalEnergyConfig(
        region="California",
        pue=1.2,  # Google datacenter average
        carbon_intensity=350.861,  # EPA eGRID 2021
        electricity_cost=0.18,  # EIA 2023
    ),
    "washington": RegionalEnergyConfig(
        region="Washington",
        pue=1.15,  # Microsoft datacenter
        carbon_intensity=260.0,  # EPA eGRID 2021
        electricity_cost=0.10,  # EIA 2023
    ),
    "virginia": RegionalEnergyConfig(
        region="Virginia",
        pue=1.25,
        carbon_intensity=385.2,  # EPA eGRID 2021
        electricity_cost=0.13,  # EIA 2023
    ),
    "texas": RegionalEnergyConfig(
        region="Texas",
        pue=1.3,
        carbon_intensity=425.3,  # EPA eGRID 2021
        electricity_cost=0.12,  # EIA 2023
    ),
    # Europe
    "ireland": RegionalEnergyConfig(
        region="Ireland",
        pue=1.17,  # Microsoft Dublin
        carbon_intensity=296.0,  # IEA 2022
        electricity_cost=0.25,  # Eurostat 2023
    ),
    "germany": RegionalEnergyConfig(
        region="Germany",
        pue=1.2,
        carbon_intensity=350.0,  # IEA 2022
        electricity_cost=0.31,  # Eurostat 2023
    ),
    "france": RegionalEnergyConfig(
        region="France",
        pue=1.18,
        carbon_intensity=51.0,  # IEA 2022 (low due to nuclear)
        electricity_cost=0.23,  # Eurostat 2023
    ),
    "netherlands": RegionalEnergyConfig(
        region="Netherlands",
        pue=1.16,  # Google Eemshaven
        carbon_intensity=315.0,  # IEA 2022
        electricity_cost=0.22,  # Eurostat 2023
    ),
    # Asia Pacific
    "singapore": RegionalEnergyConfig(
        region="Singapore",
        pue=1.35,  # Tropical climate
        carbon_intensity=408.0,  # IEA 2022
        electricity_cost=0.19,  # SP Group 2023
    ),
    "japan": RegionalEnergyConfig(
        region="Japan",
        pue=1.22,
        carbon_intensity=435.0,  # IEA 2022
        electricity_cost=0.21,  # TEPCO 2023
    ),
    "south_korea": RegionalEnergyConfig(
        region="South Korea",
        pue=1.28,
        carbon_intensity=415.0,  # IEA 2022
        electricity_cost=0.14,  # KEPCO 2023
    ),
    "australia": RegionalEnergyConfig(
        region="Australia",
        pue=1.3,
        carbon_intensity=478.0,  # IEA 2022
        electricity_cost=0.22,  # Australian Energy Regulator 2023
    ),
    # Other Notable Regions
    "canada": RegionalEnergyConfig(
        region="Canada",
        pue=1.18,
        carbon_intensity=120.0,  # IEA 2022 (low due to hydro)
        electricity_cost=0.11,  # Canadian Average 2023
    ),
    "norway": RegionalEnergyConfig(
        region="Norway",
        pue=1.15,  # Cold climate advantage
        carbon_intensity=28.0,  # IEA 2022 (very low due to hydro)
        electricity_cost=0.15,  # Nordic average 2023
    ),
    "india": RegionalEnergyConfig(
        region="India",
        pue=1.45,  # Higher due to climate
        carbon_intensity=643.0,  # IEA 2022
        electricity_cost=0.11,  # Industrial rate 2023
    ),
}
