# Restormer: PyTorch Edition

This repository provides a streamlined and easy-to-use PyTorch implementation of the paper **[Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)**.

The project is managed by a central `launch.py` script that automates the entire workflow, including environment setup, dependency installation, data downloading, training, and evaluation.

## Quick Start

Getting started is designed to be as simple as possible. For detailed installation steps, see [INSTALL.md](./INSTALL.md).

### Run the Full Pipeline
To set up the environment, download data, train the model, and run evaluation, simply execute the launcher script:
```bash
python launch.py
```

### Running Only Training or Testing
You can easily customize the workflow using command-line flags:

- **Train Only**:
  ```bash
  python launch.py --skip-test
  ```
- **Test Only** (using a pre-existing trained model):
  ```bash
  python launch.py --skip-train
  ```
- **Skip Installation**: If your environment and datasets are already set up, you can skip the installation steps:
  ```bash
  python launch.py --skip-install
  ```

For a full list of options, use the help flag:
```bash
python launch.py --help
```

## Using Pre-trained Models

To evaluate a pre-trained model:
1. Download the model weights (e.g., from the [official repository's releases](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u?usp=sharing)).
2. Place the model file (e.g., `net_g_latest.pth`) inside the `experiments/Restormer/models/` directory. You may need to create these folders if they don't exist.
3. Run the launcher with the `--skip-train` flag:
   ```bash
   python launch.py --skip-train
   ```

## Configuration

The main configuration for the project, including model architecture, training parameters, and dataset paths, is located in:
```
Options/Restormer.yml
```
You can modify this file to change hyperparameters or adjust experiment settings.

## Project Workflow

The `launch.py` script follows the workflow illustrated below, automating each step from setup to execution.

![Project Workflow](./assets/workflow.png)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
