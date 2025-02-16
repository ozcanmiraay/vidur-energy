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
        Power interpolation considering that max GPU utilization (and thus max power draw)
        can occur even at lower MFU values.
        
        Args:
            mfu: Model FLOP Utilization (in percentage, 0-100)
        Returns:
            Interpolated power consumption in Watts
        
        Notes:
            - 100% GPU utilization (max power draw) can occur at MFU values as low as 20-45%
            - Power should never exceed max_util (100% GPU utilization power)
            - We use a more aggressive scaling for lower MFU values to reflect this
        """
        # Convert MFU to decimal if it's in percentage
        mfu_decimal = mfu / 100
        
        # Define the MFU threshold where we might hit max GPU utilization
        TYPICAL_HIGH_MFU = 0.45  # Based on typical high MFU values for inference
        
        # More aggressive power scaling for lower MFU values
        # This reflects that we can hit high GPU utilization even at lower MFU
        utilization_factor = min(1.0, (mfu_decimal / TYPICAL_HIGH_MFU) ** 0.7)
        
        # Calculate power, capped at max_util
        power = self.gpu_config.idle + (self.gpu_config.max_util - self.gpu_config.idle) * utilization_factor
        return float(power)

    def calculate_energy(self, gpu_hours: float, mfu: float) -> Dict[str, float]:
        """Calculate energy consumption and related metrics."""
        # Get effective power based on MFU
        effective_power = self.interpolate_power(mfu)
        
        # Calculate energy in kWh
        energy_kwh = (effective_power * gpu_hours) / 1000  # Convert W*h to kWh
        
        # Calculate energy cost
        energy_cost = energy_kwh * self.region_config.electricity_cost
        
        # Calculate carbon emissions
        carbon_emissions = (
            energy_kwh * self.region_config.carbon_intensity +
            self.gpu_config.manufacturing_emissions * gpu_hours
        )
        
        return {
            'energy_kwh': energy_kwh,
            'energy_cost': energy_cost,
            'carbon_emissions': carbon_emissions,
            'effective_power': effective_power
        } 