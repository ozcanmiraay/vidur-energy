from dataclasses import dataclass

@dataclass
class GPUPowerConfig:
    """Configuration for GPU power characteristics."""
    idle: float  # Watts at idle
    max_util: float  # Watts at 100% utilization (typical FP16/BF16 inference)
    tdp: float  # Thermal Design Power (Watts)
    manufacturing_emissions: float  # gCO2eq for manufacturing (Scope 3)
    memory_bandwidth: float  # GB/s
    memory_capacity: float  # GB
    peak_flops_fp16: float  # Peak TFLOPS for FP16/BF16
    peak_flops_fp32: float  # Peak TFLOPS for FP32

# Updated GPU power configurations (per physical GPU)
GPU_POWER_CONFIGS = {
    "a100": GPUPowerConfig(
        idle=100.0,  # Approx. idle draw per SXM4 A100
        max_util=400.0,  # Full utilization power draw
        tdp=400.0,
        manufacturing_emissions=150_000.0,  # gCO2e = 150 kgCO2e
        memory_bandwidth=2039.0,
        memory_capacity=80.0,
        peak_flops_fp16=312.0,
        peak_flops_fp32=19.5  # Raw FP32 (not tensor TF32)
    ),
    "h100": GPUPowerConfig(
        idle=60.0,  # Estimated idle draw per SXM5 H100
        max_util=700.0,  # Full power under load
        tdp=700.0,
        manufacturing_emissions=180_000.0,  # gCO2e = 180 kgCO2e
        memory_bandwidth=3350.0,
        memory_capacity=80.0,
        peak_flops_fp16=990.0,
        peak_flops_fp32=67.0
    ),
    "a40": GPUPowerConfig(
        idle=30.0,
        max_util=300.0,
        tdp=300.0,
        manufacturing_emissions=100_000.0,  # gCO2e = 100 kgCO2e
        memory_bandwidth=696.0,
        memory_capacity=48.0,
        peak_flops_fp16=150.0,
        peak_flops_fp32=37.4
    )
}