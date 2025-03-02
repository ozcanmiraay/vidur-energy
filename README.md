# 🌱⚡ Vidur-Energy: Extending Vidur for Power and Energy Tracking in LLM Inference

Vidur-Energy is an **enhanced version** of [Vidur](https://github.com/microsoft/vidur), a high-fidelity LLM inference system simulator, with **additional energy tracking capabilities**. This extension introduces **power draw monitoring**, **energy consumption analysis**, and **carbon emission estimation**, enabling more **sustainable AI inference optimizations**. 🌍💡

---

## 🌟 Features

Vidur-Energy retains all core functionalities of Vidur while adding **new energy-aware insights**, such as:

- ⚡ **Power Tracking**: Extracts **GPU power draw** at different utilization levels.
- 🔋 **Energy Consumption Estimation**: Tracks energy usage **across inference workloads**.
- 🌿 **Preliminary Carbon Footprint Estimation**: Uses **grid carbon intensity data** to estimate **inference-related emissions**.
- 🛠️ **Modular Energy Tracking**: A more **configurable** and **extensible** approach to tracking energy metrics.
- ✅ **Full Compatibility with Vidur**: All existing **simulation capabilities** remain **unchanged**.

---

## ⚡ 4. Vidur-Vessim Co-Simulation (`vidur-vessim-basic` Branch)

The `vidur-vessim-basic` branch integrates **Vidur** with **[Vessim](https://github.com/dos-group/vessim)**, enabling co-simulation for **power-aware AI inference analysis**. 🔄 This branch allows tracking **energy demand**, **solar energy generation**, **battery storage behavior**, and **carbon emissions** during inference. ☀️🔋

### 🚀 Features:
- 📊 **Vidur Simulation Integration** → Uses `vidur` simulation results as input for `vessim`.
- 🌎 **Time-Zone Aware Simulations** → Supports **global locations**.
- ☀️🔋 **Solar Energy & Battery Modeling** → Estimates power flow.
- 🌿 **Carbon Emissions Analysis** → Calculates emissions and **renewable offsets**.
- 📈 **Custom Aggregation & Reporting** → Offers **trend analysis** and **total power analysis**.

---

### 🏆 7. Acknowledgments

This project builds on **Vidur**, originally developed by **Microsoft Research**, and integrates **[Vessim](https://github.com/dos-group/vessim)**, developed by the **Distributed and Operating Systems group at TU Berlin**. 🏛️💚  

### 🔬 About Vessim
[Vessim](https://github.com/dos-group/vessim) is a **versatile co-simulation testbed for carbon-aware applications and systems**. It connects domain-specific simulators for renewable power generation and energy storage with real computing infrastructure, making it an essential tool for **carbon-aware AI research**. 🌍⚡  

If you use **Vessim** in your research, please cite:
> **Philipp Wiesner, Ilja Behnke, Paul Kilian, Marvin Steinke, and Odej Kao**.  
> *"Vessim: A Testbed for Carbon-Aware Applications and Systems."*  
> 3rd Workshop on Sustainable Computer Systems (**HotCarbon 2024**).  
> [Read More](https://arxiv.org/abs/2405.05465)

For **software-in-the-loop** methodology, refer to:
> **Philipp Wiesner, Marvin Steinke, Henrik Nickel, Yazan Kitana, and Odej Kao**.  
> *"Software-in-the-Loop Simulation for Developing and Testing Carbon-Aware Applications."*  
> Software: Practice and Experience, 53 (12). **2023**.  
> [Read More](https://doi.org/10.1002/spe.3275)

💡 **Check out the official** [Vessim documentation](https://github.com/dos-group/vessim) **to explore its capabilities.** 🚀♻️  

For more details on **Vidur**, visit the **[Vidur paper (MLSys'24)](https://arxiv.org/abs/2405.05465)**.

---

## 📜 8. License

Vidur-Energy follows the **original Vidur license**, and Vessim is licensed under **MIT**.  
Please review [`LICENSE`](./LICENSE) for details. 📄

---

🌱 **Sustainable AI starts here!** 🌎⚡ Let’s make LLM inference **greener and smarter** together! 🚀♻️
