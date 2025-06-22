import os
import sys
import subprocess
import argparse
import yaml
import tempfile
import torch

def run_command(command):
    """Runs a command, streams its output, and checks for errors."""
    print(f"--- Running command: {' '.join(command)} ---")
    try:
        # Use Popen with direct output passthrough for tqdm compatibility
        process = subprocess.Popen(
            command,
            stdout=None,  # Direct passthrough to parent's stdout
            stderr=None,  # Direct passthrough to parent's stderr
            text=True,
            encoding='utf-8'
        )
        
        # Wait for completion
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        print("--- Command finished successfully ---\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n--- Command failed with exit code {e.returncode}: {' '.join(command)} ---")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n--- Command not found: {command[0]}. Is it in your PATH? ---")
        sys.exit(1)

def install():
    """Install dependencies and run setup.py."""
    print("--- Step 1: Installing dependencies ---")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("--- Step 2: Setting up project with setup.py ---")

    setup_command = [sys.executable, "setup.py", "develop"]

    # The setup script requires CUDA Toolkit for compilation, not just a GPU.
    # We check for both a GPU and the CUDA_HOME env var. If either is missing,
    # we skip compiling the extensions, which is what the original setup.py supports.
    cuda_available = torch.cuda.is_available()
    cuda_home_set = os.getenv('CUDA_HOME') is not None
    
    if not (cuda_available and cuda_home_set):
        if not cuda_available:
            print("--- INFO: PyTorch CUDA runtime is not available. ---")
        if not cuda_home_set:
            print("--- INFO: CUDA_HOME environment variable is not set. ---")
        
        print("--- Passing --no_cuda_ext to setup.py to skip CUDA extension compilation. ---")
        setup_command.append("--no_cuda_ext")

    run_command(setup_command)
    print("--- Installation complete ---")

def train(args):
    """Run the training process."""
    print("--- Step 3: Starting model training ---")
    train_script = "basicsr/train.py"
    master_port = "4321"

    try:
        with open(args.opt, 'r') as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Options file not found at {args.opt}")
        sys.exit(1)

    # The paths in the yml file are like './Datasets/...'.
    # We replace the './Datasets' part with the provided data root to make them correct.
    if args.dataroot:
        print(f"--- Updating dataset paths with root: {args.dataroot} ---")
        for phase, dataset in opt['datasets'].items():
            for key in ['dataroot_gt', 'dataroot_lq']:
                if key in dataset and dataset.get(key) and './Datasets' in dataset[key]:
                    original_path = dataset[key]
                    new_path = original_path.replace('./Datasets', args.dataroot)
                    dataset[key] = new_path
                    print(f"  - Updated {phase} '{key}' to: {new_path}")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml', encoding='utf-8') as tmp_file:
        yaml.dump(opt, tmp_file, sort_keys=False)
        temp_opt_file = tmp_file.name
    
    os.environ.update({
        "MASTER_ADDR": "localhost", "MASTER_PORT": master_port,
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "USE_LIBUV": "0"
    })

    command = [sys.executable, train_script, "-opt", temp_opt_file, "--launcher", "pytorch"]
    run_command(command)
    os.remove(temp_opt_file)
    print("--- Training complete ---")

def test(args):
    """Run the testing process to generate results and metrics."""
    print("--- Step 4: Starting model testing and evaluation ---")

    try:
        with open(args.opt, 'r') as f:
            opt = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Options file not found at {args.opt}")
        sys.exit(1)

    experiment_name = opt.get('name')
    if not experiment_name:
        print(f"Error: Could not find 'name' in options file {args.opt}")
        sys.exit(1)

    # Path to the model trained in the previous step
    model_path = f"experiments/{experiment_name}/models/net_g_latest.pth"
    
    # Check if the model file actually exists before trying to test
    if not os.path.exists(model_path):
        print(f"--- ERROR: Trained model not found at {model_path}. Skipping test. ---")
        return

    command = [
        sys.executable, 
        "test.py",
        "-opt",
        args.opt,
        f"--model_path={model_path}",
    ]
    run_command(command)
    print("--- Testing complete ---")

def main():
    parser = argparse.ArgumentParser(
        description="Restormer all-in-one script for setup, training, and testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--skip-install', action='store_true', 
        help='Skip dependency installation and setup.py.'
    )
    parser.add_argument(
        '--skip-train', action='store_true',
        help='Skip the training step (requires a pre-existing trained model for testing).'
    )
    parser.add_argument(
        '--skip-test', action='store_true',
        help='Skip the final testing and evaluation step.'
    )
    parser.add_argument(
        "--dataroot", type=str, default="Datasets",
        help="Root directory for the datasets for training."
    )
    parser.add_argument(
        "--opt", type=str, default="Options/Restormer.yml",
        help="Path to the options YAML file for training."
    )
    args = parser.parse_args()

    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print("="*60)
        print("ERROR: You must run this script inside a virtual environment.")
        print("Please create and activate one first. For example:")
        print("\n  python -m venv venv")
        print("  .\\venv\\Scripts\\activate   (on Windows)")
        print("  source venv/bin/activate  (on Linux/macOS)")
        print("\nThen, run 'python launch.py' again.")
        print("="*60)
        sys.exit(1)

    if not args.skip_install:
        install()
    
    if not args.skip_train:
        train(args)
    
    if not args.skip_test:
        test(args)

    print("\n--- All steps complete! ---")

if __name__ == '__main__':
    main() 