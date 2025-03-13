from dataclasses import dataclass


@dataclass
class GPUPowerConfig:
    """Configuration for GPU power characteristics."""

    idle: float  # Watts at idle
    max_util: float  # Watts at 100% utilization
    tdp: float  # Thermal Design Power (Watts)
    manufacturing_emissions: float  # gCO2eq for manufacturing (Scope 3)
    memory_bandwidth: float  # GB/s
    memory_capacity: float  # GB
    peak_flops_fp16: float  # Peak TFLOPS for FP16/BF16
    peak_flops_fp32: float  # Peak TFLOPS for FP32


# Enhanced GPU power configurations
GPU_POWER_CONFIGS = {
    "a100": GPUPowerConfig(
        idle=744.8,
        max_util=5245.0,
        tdp=400,
        manufacturing_emissions=135.3,
        memory_bandwidth=1935,
        memory_capacity=80,
        peak_flops_fp16=312.0,  # 312 TFLOPS for FP16/BF16
        peak_flops_fp32=156.0,  # 156 TFLOPS for FP32
    ),
    "h100": GPUPowerConfig(
        idle=900.0,  # Higher idle due to larger chip
        max_util=6500.0,  # Higher max power
        tdp=700,  # Actual TDP
        manufacturing_emissions=150.0,  # Higher due to larger die
        memory_bandwidth=3350,  # Actual spec
        memory_capacity=80,  # Actual spec
        peak_flops_fp16=312.0,  # 312 TFLOPS for FP16/BF16
        peak_flops_fp32=156.0,  # 156 TFLOPS for FP32
    ),
    "a40": GPUPowerConfig(
        idle=400.0,  # Lower idle than A100
        max_util=3000.0,  # Lower max than A100
        tdp=300,  # Actual TDP
        manufacturing_emissions=100.0,  # Lower due to smaller die
        memory_bandwidth=696,  # Actual spec
        memory_capacity=48,  # Actual spec
        peak_flops_fp16=312.0,  # 312 TFLOPS for FP16/BF16
        peak_flops_fp32=156.0,  # 156 TFLOPS for FP32
    ),
}
