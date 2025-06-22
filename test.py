## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import sys
import argparse
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Import the model architecture
from basicsr.models.archs.restormer_arch import Restormer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='Options/Deraining_Restormer.yml', help='Path to option YAML file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--dataset', type=str, help='Name of the test dataset to evaluate (e.g., Rain100H). If not provided, all test datasets are evaluated.')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Read Configuration and Set Up
    # -------------------------------------------------------------------------
    if not os.path.exists(args.model_path):
        print(f"--- ERROR: Model file not found at '{args.model_path}' ---")
        sys.exit(1)

    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    project_root = Path(__file__).parent
    DATASETS_ROOT = project_root / 'Datasets' / 'test'
    exp_name = opt.get('name', 'Restormer_experiment')
    RESULTS_ROOT = project_root / 'experiments' / exp_name / 'visualization'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Load the Trained Model
    # -------------------------------------------------------------------------
    network_g_opt = opt['network_g']
    model = Restormer(
        inp_channels=network_g_opt.get('inp_channels', 3),
        out_channels=network_g_opt.get('out_channels', 3),
        dim=network_g_opt.get('dim', 48),
        num_blocks=network_g_opt.get('num_blocks', [4, 6, 6, 8]),
        heads=network_g_opt.get('heads', [1, 2, 4, 8]),
        ffn_expansion_factor=network_g_opt.get('ffn_expansion_factor', 2.66),
        bias=network_g_opt.get('bias', False),
        LayerNorm_type=network_g_opt.get('LayerNorm_type', 'WithBias'),
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint.get('params_ema', checkpoint.get('params', checkpoint)))
    model.eval()
    print(f"Successfully loaded model from {args.model_path}")

    # -------------------------------------------------------------------------
    # 3. Define Datasets and Run Inference/Evaluation
    # -------------------------------------------------------------------------
    dataset_folders = []
    if args.dataset:
        dataset_path = DATASETS_ROOT / args.dataset
        if not dataset_path.is_dir():
            print(f"--- ERROR: Specified dataset '{args.dataset}' not found in '{DATASETS_ROOT}' ---")
            sys.exit(1)
        dataset_folders.append(dataset_path)
    else:
        if not DATASETS_ROOT.is_dir():
            print(f"--- ERROR: Test data directory not found at '{DATASETS_ROOT}' ---")
            sys.exit(1)
        dataset_folders = sorted([p for p in DATASETS_ROOT.iterdir() if p.is_dir()])

    if not dataset_folders:
        print(f"--- No datasets found to evaluate in '{DATASETS_ROOT}' ---")
        sys.exit(1)

    overall_psnr = []
    overall_ssim = []
    print("Starting evaluation on test datasets...")

    for dataset_path in dataset_folders:
        dataset_name = dataset_path.name
        gt_root = dataset_path / 'target'
        lq_root = dataset_path / 'input'
        output_root = RESULTS_ROOT / dataset_name
        output_root.mkdir(parents=True, exist_ok=True)

        if not gt_root.is_dir() or not any(gt_root.iterdir()):
            print(f"Warning: Ground truth directory for {dataset_name} is missing or empty. Skipping.")
            continue

        gt_paths = sorted(p for p in gt_root.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg'])

        for gt_path in tqdm(gt_paths, desc=f"Processing {dataset_name}"):
            lq_path = lq_root / gt_path.name
            if not lq_path.exists():
                continue

            img_lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_lq)
            
            restored_img = torch.clamp(output, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()
            restored_img = (restored_img * 255.0).round().astype(np.uint8)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_root / gt_path.name), restored_img)

        result_paths = {p.stem: p for p in output_root.iterdir()}
        gt_paths_map = {p.stem: p for p in gt_paths}
        common_files = sorted(list(result_paths.keys() & gt_paths_map.keys()))

        if not common_files:
            print(f"Warning: No common files for evaluation in {dataset_name}. Skipping metrics.")
            continue

        dataset_psnr, dataset_ssim = [], []
        for img_stem in common_files:
            restored_array = np.array(Image.open(result_paths[img_stem]))
            gt_array = np.array(Image.open(gt_paths_map[img_stem]))
            
            psnr = compute_psnr(gt_array, restored_array, data_range=255)
            ssim = compute_ssim(gt_array, restored_array, data_range=255, channel_axis=-1, win_size=7)
            dataset_psnr.append(psnr)
            dataset_ssim.append(ssim)

        avg_psnr = np.mean(dataset_psnr) if dataset_psnr else 0
        avg_ssim = np.mean(dataset_ssim) if dataset_ssim else 0
        print(f"{dataset_name}: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}")
        overall_psnr.extend(dataset_psnr)
        overall_ssim.extend(dataset_ssim)

    if overall_psnr:
        final_psnr = np.mean(overall_psnr)
        final_ssim = np.mean(overall_ssim)
        print("\n" + "="*50 + f"\nOverall Average: PSNR = {final_psnr:.4f}, SSIM = {final_ssim:.4f}\n" + "="*50)

if __name__ == '__main__':
    main()