from dataclasses import dataclass

@dataclass
class GPUPowerConfig:
    """Configuration for GPU power characteristics."""
    idle: float  # Watts
    low_util: float  # 10% utilization (Watts)
    med_util: float  # 50% utilization (Watts)
    high_util: float  # 100% utilization (Watts)
    tdp: float  # Thermal Design Power (Watts)
    manufacturing_emissions: float  # gCO2eq for manufacturing (Scope 3)
    memory_bandwidth: float  # GB/s
    memory_capacity: float  # GB

# Enhanced GPU power configurations
GPU_POWER_CONFIGS = {
    "a100": GPUPowerConfig(
        idle=744.8,
        low_util=1687.5,
        med_util=3781.8,
        high_util=5245.0,
        tdp=400,
        manufacturing_emissions=135.3,
        memory_bandwidth=1935,
        memory_capacity=80
    ),
    # Add other GPU configs...
} 