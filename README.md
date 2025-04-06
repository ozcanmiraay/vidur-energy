# 📗 **Vidur-Energy: Extending Vidur for Power and Energy Tracking in LLM Inference**

📄 _This repository accompanies the paper:_  

**“Towards Quantifying the Energy Consumption and Carbon Emissions of LLM Inference: A Simulation-Based Approach”**  
_Miray Özcan, Minerva University, 2025_  

Vidur-Energy is an **enhanced version** of [Vidur](https://github.com/microsoft/vidur), a high-fidelity **LLM inference system simulator**, with **additional energy tracking capabilities**. This extension introduces:

⚡ **Power draw monitoring**  
🔋 **Energy consumption analysis**  
🌍 **Carbon emission estimation**

These features enable **more sustainable AI inference optimizations**. 🌱🌎

---

## 🌟 **Features**  

Vidur-Energy retains all core functionalities of Vidur while adding **new energy-aware insights**, such as:  

✅ **Power Tracking**: Extracts **GPU power draw** at different utilization levels.  
✅ **Energy Consumption Estimation**: Tracks **energy usage** across inference workloads.  
✅ **Preliminary Carbon Footprint Estimation**: Uses **grid carbon intensity data** to estimate **inference-related emissions**.  
✅ **Modular Energy Tracking**: A more **configurable** and **extensible** approach to tracking energy metrics.  
✅ **Full Compatibility with Vidur**: All existing **simulation capabilities** remain **unchanged**.  

---

## 🔧 **1. Setup**  

### 🐍 **Using `mamba` (Recommended)**  
```sh
mamba env create -p ./env -f ./environment.yml
mamba env update -f environment-dev.yml
```

### 🐍 **Using `venv`**  
```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 🐍 **Using `conda` (Least Recommended)**  
```sh
conda env create -p ./env -f ./environment.yml
conda env update -f environment-dev.yml
```

---

## 🚀 **2. Running the Simulator**

You can run Vidur in standalone mode for LLM inference simulations with or without energy tracking.

> ℹ️ **Need help with parameters?**  
> Run the following to view all configurable flags and their descriptions:  
> ```sh
> python -m vidur.main -h
> ```

---

### 🧪 **Running a Standard Simulation**

To execute a standard simulation (**without energy tracking**):  
```sh
python -m vidur.main
```

---

### 🔁 **Example: Synthetic Request Generator**
```sh
python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Llama-2-7b-hf \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 400000 \
--length_generator_config_type zipf \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 20 \
--replica_scheduler_config_type vllm \
--vllm_scheduler_config_batch_size_cap 128 \
--vllm_scheduler_config_max_tokens_in_batch 4096 \
--metrics_config_store_utilization_metrics \
--execution_time_predictor_config_type random_forrest
```

---

### 📂 **Example: Trace-Based Simulation**
```sh
python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 512  \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
```

---

## 🌿 **3. Energy Tracking in `energy-tracking` Branch**  

The `energy-tracking` branch introduces **power and energy-related analytics** within Vidur. This branch features:  

📊 **Energy-aware statistics extraction**  
📉 **A reporting module for power consumption, energy efficiency, and carbon footprint**  

### 🔄 **Changes in This Branch**  
- **New Scripts**:  
  - `stats_extractor_energy.py`: Extracts **power, energy, and carbon footprint** metrics.  
  - `stats_extractor_energy_reporting/`: Directory for **configurations and reporting tools**.  

- **Configuration Additions**:  
  - `config/gpu_configs.py`: Defines **power profiles** for different GPU models.  
  - `config/region_configs.py`: Provides **grid parameters** like carbon intensity, PUE, and electricity cost.  

### 🌱 **Running Energy Tracking & Analysis**  

#### 📊 **Extracting Energy Metrics from a Simulation**  
Let's assume the name of our simulation result subdirectory is 'vidur-results-demo.'
```sh
python -m vidur.config_optimizer.analyzer.stats_extractor_energy \
--sim-results-dir simulator_output/vidur-results-demo
```
📁 This generates an **`analysis/` subdirectory** containing energy usage statistics.  

#### 📄 **Generating Energy Reports**  
```sh
python -m vidur.config_optimizer.analyzer.stats_extractor_energy_reporting \
--sim-results-dir simulator_output/vidur-results-demo  \
--region california
```
This generates **visual reports** on:  
✅ **Power and energy usage over time**  
✅ **Carbon emissions impact per region**  
✅ **Model parallelism efficiency vs energy consumption**  
✅ **Comparative energy costs across regions**  

---

### 📊 Energy Consumption Reports

<div align="center">

<table>
  <tr>
    <td align="center" style="padding: 10px;">
      <strong>🌍 Sustainable AI Performance Metrics</strong><br>
      <img src="./assets/energy-report-1.png" width="250" height="170">
    </td>
    <td align="center" style="padding: 10px;">
      <strong>📉 Energy Over Time & Efficiency</strong><br>
      <img src="./assets/energy-report-2.png" width="320" height="170">
    </td>
    <td align="center" style="padding: 10px;">
      <strong>🌍 Regional Emissions & Cost</strong><br>
      <img src="./assets/energy-report-3.png" width="240" height="170">
    </td>
  </tr>
</table>

</div>

---

## 🧪 **4. Replicating the Experiments in `experiments` Branch**  

This project includes a full set of scripted experiments that were used to generate the results in the accompanying paper:

📄 _“Towards Quantifying the Energy Consumption and Carbon Emissions of LLM Inference: A Simulation-Based Approach”_

These experiments evaluate:
- ⚡ **Power consumption** under different inference workloads
- ⏳ **Execution time** across varying request patterns and system configs
- 🌱 **Energy efficiency** tradeoffs for different models and deployment settings

---

### 📂 **Directory Structure**

All relevant files are in the `experiments-final/` directory, organized as follows:

- `experiments-final/scripts/`  
  Contains **fully executable Python scripts** for each experiment.  
  Example:  
  - `exp1-numrequests-power-energy.py`: Evaluates how increasing the number of requests impacts power usage and energy draw.  
  - `exp3-prefill-decode-ratio.py`: Analyzes tradeoffs across prefill/decode token ratio variations.  

- `experiments-final/analysis/`  
  Contains **Jupyter Notebooks** for post-processing, plotting, and deeper analysis.  
  Each notebook corresponds to an experiment script and renders visuals similar to those in the paper.  
  Example:  
  - `exp1-numrequests-power-energy.ipynb`: Plots power and efficiency results from the script above.

---

### 🔬 **How to Run the Experiments**

> 🛠️ Before running, ensure you have completed setup and are in the correct environment.

#### Step 1: Run a predefined experiment script  
Each script generates raw simulation outputs + energy tracking metadata.

```sh
python experiments-final/scripts/exp1-numrequests-power-energy.py
```

You can repeat this with other scripts in the same folder.

#### Step 2: Open the corresponding Jupyter notebook for analysis

```sh
jupyter notebook experiments-final/analysis/exp1-numrequests-power-energy.ipynb
```

This notebook:
- Loads the output directory automatically
- Plots power draw, GPU utilization, energy efficiency, and emissions metrics
- Compares across variations (e.g., model type, request volume, QPS)

#### Step 3: Adjust experiment parameters (optional)

Each script uses predefined simulation configurations, but you can modify:
- Model type (e.g., LLaMA-70B vs LLaMA-8B)
- Request generator parameters (e.g., QPS, length distribution)
- Parallelism settings
- Region for emissions comparison

Modify directly in the experiment script or use YAML configs in `configs/`.

---

### 📊 Visual Examples from the Paper’s Experiments

<div align="center">

<table>
  <tr>
    <td align="center" style="padding: 10px;">
      <strong>⚡ Power Consumption vs. Number of Requests</strong><br>
      <img src="./assets/experiments-1.png" width="300" height="200">
    </td>
    <td align="center" style="padding: 10px;">
      <strong>⏳ Execution Time vs. Number of Requests</strong><br>
      <img src="./assets/experiments-2.png" width="300" height="200">
    </td>
    <td align="center" style="padding: 10px;">
      <strong>🌱 Comparative Energy Efficiency Across Models</strong><br>
      <img src="./assets/experiments-3.png" width="300" height="200">
    </td>
  </tr>
</table>

</div>

---

## 🔋 **5. Vidur–Vessim Co-Simulation (`vidur-vessim-basic` Branch)**

The `vidur-vessim-basic` branch enables **carbon-aware inference simulations** by integrating the **Vidur LLM inference simulator** with **Vessim**, a testbed for modeling solar generation, battery dynamics, and regional carbon emissions.

This co-simulation pipeline allows you to:
- 🔆 Simulate solar energy generation across locations
- 🔋 Track battery charging/discharging cycles and storage capacity
- 🌍 Evaluate the **carbon footprint** of your LLM inference workload under varying energy profiles and grid intensities

---

### ⚙️ Environment Setup

Due to dependency mismatches (notably `numpy`), we recommend creating a **separate virtual environment** for running Vessim integration.

```bash
# Create and activate a new environment
python3 -m venv .venv-vessim
source .venv-vessim/bin/activate

# Install all required packages for this branch
pip install -r requirements.txt
```

Run your **Vidur simulation first** (as shown in previous sections), then use the path to the result folder in the next step.

---

### 🌞 Running Vidur–Vessim Co-Simulation

Example command:

```bash
python -m vidur.vidur_vessim.cli \
--vidur-sim-dir simulator_output/vidur-vessim-example-capstone \
--location "San Francisco" \
--agg-freq 1min \
--analysis-type "trend analysis" \
--step-size 60 \
--solar-scale-factor 600 \
--battery-capacity 100 \
--battery-initial-soc 0.8 \
--battery-min-soc 0.2 \
--log-metrics \
--carbon-analysis \
--low-carbon-threshold 100 \
--high-carbon-threshold 200 \
--interpolate-datasets
```

---

### 📘 CLI Argument Reference

| Argument | Description |
|----------|-------------|
| `--vidur-sim-dir` | Path to Vidur’s simulation output directory (must contain power data). |
| `--location` | Location used for time zone conversion and solar irradiance estimation. |
| `--agg-freq` | Aggregation frequency for simulation data (e.g., `1min`). |
| `--analysis-type` | Type of analysis: `"trend analysis"` for time series or `"total power analysis"`. |
| `--step-size` | Simulation time step in seconds (e.g., `60` for 1-minute resolution). |
| `--solar-scale-factor` | Size of the solar panel system in watts (e.g., `600` for a 600W array). |
| `--battery-capacity` | Battery storage capacity in watt-hours (e.g., `100` = 0.1 kWh). |
| `--battery-initial-soc` | Initial battery state of charge (0.0 to 1.0). |
| `--battery-min-soc` | Minimum state of charge to preserve (battery cannot discharge below this). |
| `--log-metrics` | Enables writing a detailed `simulation_metrics.txt` file summarizing power, emissions, and battery usage. |
| `--carbon-analysis` | Enables carbon footprint calculations using dynamic carbon intensity data. |
| `--low-carbon-threshold` | Threshold (in gCO₂/kWh) below which energy is considered "green". |
| `--high-carbon-threshold` | Threshold above which energy is considered "dirty". |
| `--interpolate-datasets` | Enables interpolation to align power, solar, and carbon intensity data on common timestamps. |

---

### 📊 Outputs & Visualizations

After execution, results are saved under a directory called `vessim_analysis/`. You’ll find:

### 📊 Vessim Co-Simulation Visualizations

<div align="center">

<table>
  <tr>
    <td align="center" style="padding: 10px; vertical-align: top;">
      <strong>🔄 Power Flow Analysis</strong><br>
      <img src="./assets/power_flow_analysis.png" width="320"><br>
      <p style="margin-top: 5px;">Visualizes how solar, grid, and model power usage intersect.</p>
    </td>
    <td align="center" style="padding: 10px; vertical-align: top;">
      <strong>🔋 Battery SOC Over Time</strong><br>
      <img src="./assets/battery_soc_plot.png" width="320"><br>
      <p style="margin-top: 5px;">Tracks battery charge levels and threshold efficiency.</p>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding: 10px; vertical-align: top;">
      <strong>🌍 Carbon Emissions Breakdown</strong><br>
      <img src="./assets/carbon_emissions_plot.png" width="320"><br>
      <p style="margin-top: 5px;">Gross emissions, renewable offset, and net footprint.</p>
    </td>
    <td align="center" style="padding: 10px; vertical-align: top;">
      <strong>📉 Battery Usage Distribution</strong><br>
      <img src="./assets/battery_usage_distribution.png" width="320"><br>
      <p style="margin-top: 5px;">Time spent charging, discharging, or idle.</p>
    </td>
  </tr>
</table>

</div>

- **Simulation Metrics Summary**  
  All key statistics are logged to `vessim_analysis/simulation_metrics.txt`, including:
  - Total energy demand
  - Grid vs. solar energy share
  - Carbon intensity over time
  - Battery SoC distribution and cycling behavior

---

## 🔄 **6. Formatting Code**  

To format the code:  
```sh
make format
```

---

## 🤝 **7. Contributing**  

We welcome contributions to extend **Vidur-Energy**! 🎉  

🔹 **Fork the repository**  
🔹 **Create a new branch**  
🔹 **Submit a pull request**  

---

## 🙌 **8. Acknowledgments**  

### 🌿 **Built on Microsoft Research’s Vidur**  
This project builds on **Vidur**, originally developed by Microsoft Research. Our modifications enhance its sustainability features while preserving **high-fidelity inference simulation capabilities**.  

For more details, visit the **[Vidur paper (MLSys'24)](https://arxiv.org/abs/2405.05465).**  

### ⚡ **Powered by Vessim**  
Vidur-Energy **integrates Vessim**, a testbed for **carbon-aware applications and systems**, developed by TU Berlin’s **Distributed and Operating Systems Group**.  

If you use Vessim in your research, please cite:  
📝 **Wiesner et al. (2024). Vessim: A Testbed for Carbon-Aware Applications and Systems.**  

📄 **[Read more about Vessim](https://github.com/dos-group/vessim)**  

---

## 📜 **9. License**  

Vidur-Energy follows the **original Vidur license**. Please review [`LICENSE`](./LICENSE) for details.  

🚀 **Happy Sustainable AI Computing!** 🌱
