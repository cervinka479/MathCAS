# MathCAS: Modular Neural Network Training Framework

A robust, highly modular PyTorch-based framework tailored for training, evaluating, and profiling neural networks on complex tabular datasets.

Originally developed for research in **compressible fluid flow analysis**, this framework is designed to bridge the gap between rapid academic experimentation and clean, production-ready engineering practices.

![Model Performance](figures/model_performance_ResStrain_MSE.png)
_(Note: Example visualization comparing inference time vs. MSE across different model configurations. Data and specific physical parameters are anonymized for confidentiality)._

## ✨ Key Features

This repository is built with a strong focus on reproducibility, performance, and clear separation of concerns:

- **⚙️ YAML-Driven Architecture:** Define network layers, learning rates, data splits, and early stopping rules entirely via configuration files (no hardcoding).
- **🚀 NVTX Profiling:** Integrated NVIDIA NVTX markers (`nvtx_range`) in the dataloaders and training loops for deep performance analysis and bottleneck identification.
- **🧠 Advanced Diagnostics (Dying ReLU):** Custom PyTorch forward hooks designed to detect and report "dying" activation neurons across batches during inference.
- **🔄 Full Reproducibility:** Automated experiment directory creation, config hashing, and strict random seed freezing.

## 📂 Project Structure

```text
MathCAS/
├── config/             # YAML schema definitions and config loaders
├── datasets/           # Directory for tabular data (gitignored)
├── experiments/        # Auto-generated experiment tracking & model checkpoints
├── figures/            # Exported visualizations and plots
├── notebooks/          # Jupyter notebooks for EDA, visualization, and tooling
├── src/                # Core ML logic (architecture, dataloaders, evaluation)
├── templates/          # Base YAML templates for regression/classification
├── utils/              # Helpers: logger, NVTX profiler, metric computations
└── main.py             # Main entry point for the training pipeline
```
