# Installation

This repository is built and tested on PyTorch >=2.0, Python >=3.8, and CUDA 11.8+.

## 1. Clone the repository

```
git clone https://github.com/swz30/Restormer.git
cd Restormer
```

## 2. Run the automated setup script

Simply run the `launch.py` script. It will automatically handle:
- Creating a virtual environment.
- Installing all dependencies (including the correct PyTorch version).
- Downloading the required datasets.
- Setting up the project.

```
python launch.py
```

This single command will prepare the environment, train the model, and run the tests. 
You can use flags like `--skip-train` or `--skip-test` to customize the execution. See `python launch.py --help` for more options.
