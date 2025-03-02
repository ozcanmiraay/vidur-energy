\section{Conclusion}
As the energy footprint of large-scale AI models continues to grow, optimizing LLM inference for efficiency and sustainability is imperative. In this work, we proposed a simulation-based framework that integrates Vidur, a high-fidelity LLM inference simulator, with Vessim, a carbon-aware energy co-simulation tool. Our approach enables detailed power profiling of inference workloads and introduces carbon-aware scheduling strategies to reduce emissions without compromising performance.

\medskip

Through controlled experiments, we demonstrated how inference parameters—such as batch size, query rate, and model parallelism—affect power consumption and energy efficiency. Our findings reveal that GPU utilization metrics alone do not capture computational efficiency, and that Model FLOPs Utilization (MFU) serves as a more reliable predictor of power consumption. Furthermore, we show that integrating renewable energy into inference workloads can offset emissions by up to 58.6\%, underscoring the importance of carbon-aware scheduling.

\medskip

By bridging AI inference modeling with energy grid dynamics, our work lays the foundation for sustainable AI deployment strategies. Future research should focus on refining empirical power models, expanding real-time scheduling optimizations, and generalizing our approach to broader AI workloads. As LLMs continue to scale, incorporating environmental impact considerations into inference-serving architectures will be critical in aligning AI advancements with global sustainability goals.
