from typing import Dict
import numpy as np
from ..config.gpu_configs import GPUPowerConfig
from ..config.region_configs import RegionalEnergyConfig

class EnergyMetricsCalculator:
    def __init__(self, gpu_config: GPUPowerConfig, region_config: RegionalEnergyConfig):
        self.gpu_config = gpu_config
        self.region_config = region_config

    def interpolate_power(self, mfu: float) -> float:
        """
        Interpolate power consumption based on MFU using linear interpolation
        between idle and max utilization states.
        
        Args:
            mfu: Model FLOPs Utilization (0.0 to 1.0)
        Returns:
            Interpolated power consumption in Watts
        """
        # Linear interpolation: P = P_idle + (P_max - P_idle) * mfu
        power = self.gpu_config.idle + (self.gpu_config.max_util - self.gpu_config.idle) * mfu
        return float(power)

    def calculate_energy(self, gpu_hours: float, mfu: float) -> Dict[str, float]:
        """Calculate energy metrics based on GPU usage."""
        # Calculate power consumption
        effective_power = self.interpolate_power(mfu)
        
        # Calculate energy consumption in kWh
        energy_kwh = (effective_power * gpu_hours * self.region_config.pue) / 1000
        
        # Calculate energy cost
        energy_cost = energy_kwh * self.region_config.electricity_cost
        
        # Calculate carbon emissions (including manufacturing emissions)
        carbon_emissions = (
            energy_kwh * self.region_config.carbon_intensity +
            self.gpu_config.manufacturing_emissions * gpu_hours
        )
        
        # Calculate energy efficiency (FLOPS/W)
        theoretical_flops = self.gpu_config.memory_bandwidth * 1e9  # Convert GB/s to B/s
        actual_flops = theoretical_flops * mfu
        energy_efficiency = actual_flops / effective_power if effective_power > 0 else 0
        
        return {
            'energy_kwh': energy_kwh,
            'energy_cost': energy_cost,
            'carbon_emissions': carbon_emissions,
            'energy_efficiency': energy_efficiency,
            'effective_power': effective_power
        } 