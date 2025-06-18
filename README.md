# MathCAS: Modular Neural Network Training Framework

A minimal, modular PyTorch-based framework for training neural networks on tabular data, with YAML-configurable architecture, logging, and experiment management.

> **Note:** This project is a work in progress. Features and documentation may change as development continues.

## Features

-   Modular architecture and data loading
-   YAML-based experiment configuration
-   Logging to console and file (configurable)
-   Early stopping, reproducibility, and output management

## Quickstart

1. **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

2. **Prepare your dataset:**  
   Place your CSV in `datasets/` and update `templates/regression.yaml` as needed.

3. **Run training:**

    ```
    python main.py
    ```

4. **Check outputs:**  
   Models, logs, and metrics are saved in the `outputs/` directory.

## Configuration

Edit `templates/regression.yaml` or create custom templates to change model, training, or data settings.

## License

MIT License. See [LICENSE](LICENSE).
