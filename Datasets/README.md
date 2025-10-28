# Datasets

This directory contains the training and testing datasets for Restormer image restoration.

## Expected Structure

```
Datasets/
├── train/
│   └── Train13k/
│       ├── input/     # Low-quality (rainy) images
│       └── target/    # Ground truth (clean) images
└── test/
    ├── Rain100H/
    │   ├── input/     # Test images with heavy rain
    │   └── target/    # Ground truth clean images
    ├── Rain100L/
    │   ├── input/     # Test images with light rain
    │   └── target/    # Ground truth clean images
    ├── Test100/
    │   ├── input/     # Additional test images
    │   └── target/    # Ground truth clean images
    ├── Test1200/
    │   ├── input/     # Large test set
    │   └── target/    # Ground truth clean images
    └── Test2800/
        ├── input/     # Extended test set
        └── target/    # Ground truth clean images
```

## Dataset Information

- **Training Dataset**: Rain13K - A large-scale dataset with ~13,000 synthetic rainy/clean image pairs
- **Testing Datasets**: Multiple test sets with different rain densities and image counts

## Download

Datasets are automatically downloaded when you run:
```bash
python launch.py
```

Or manually download using:
```bash
python download_data.py --data train-test
```

## Manual Download Links

If automatic download fails (Google Drive rate limits):

- **Training Data**: [Rain13K Training Set](https://drive.google.com/file/d/14BidJeG4nSNuFNFDf99K-7eErCq4i47t/view)
- **Testing Data**: [Testing Datasets](https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view)

Extract the downloaded zip files directly into this `Datasets/` directory.

## Usage

The dataset paths are configured in `Options/Restormer.yml`. The default configuration expects:

- Training data: `./Datasets/test/Test1200/` (used for training in the current config)
- Validation data: `./Datasets/test/Rain100H/` (used for validation)

## Notes

- Total training dataset size: ~1.5-2 GB
- Total testing dataset size: ~100-200 MB
- Image format: PNG
- Resolution: Various (will be cropped/resized during training)

For troubleshooting dataset issues, see the main project README or run:
```bash
python manual_download_helper.py
```
