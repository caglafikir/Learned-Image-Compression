# Learned Image Compression

This project is a notebook-based PyTorch experiment for learned image compression. It explores a compact autoencoder-style compression model that learns an image latent representation and uses entropy modeling to estimate bit rate under different training settings.

The notebook focuses on:

- quantization ablation: `noise`, `ste`, and `round`
- entropy model comparison: factorized prior vs. scale hyperprior
- loss ablation: MSE, MS-SSIM, and MSE + LPIPS
- real-rate evaluation using actual compressed bitstreams
- Kodak image evaluation with PSNR and MS-SSIM metrics
- **a from-scratch arithmetic coder** (Witten-Neal-Cleary), written and validated by me, that encodes and decodes real latents end to end

## Project overview

The implementation is centered around a small convolutional codec (`SimpleCodec`) and the CompressAI `ScaleHyperprior` model. Training follows the rate-distortion objective:

L = R + λ · D

where:

- `R` is the modeled bitrate estimate
- `D` is the distortion term selected from MSE or MS-SSIM-based losses
- `λ` controls the rate-distortion tradeoff

## Repository structure

- `learned_image_compression.ipynb` — main experiment notebook
- `train_images/` — training dataset directory, typically DIV2K images
- `kodak/` — Kodak validation/test images
- `ckpt/` — model checkpoints
- `results.json` — evaluation results output

## Requirements

The notebook expects a Python environment with:

- Python 3.9+
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy
- compressai
- pytorch-msssim

Recommended:

- CUDA-capable GPU for faster training
- at least 16 GB RAM for dataset caching and training

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install torch torchvision pillow matplotlib numpy
pip install compressai pytorch-msssim
```

3. Open the notebook in Jupyter or VS Code and run the cells in order.

## Data preparation

The notebook automatically downloads and prepares data when run:

- DIV2K training images are downloaded into `./train_images`
- Kodak test images are downloaded into `./kodak`

The notebook sets these defaults:

```python
TRAIN_DIR = "./train_images"
KODAK_DIR = "./kodak"
CKPT_DIR = "./ckpt"
RESULTS_JSON = "./results.json"
```

## Usage

Run all cells sequentially in `learned_image_compression.ipynb`:

1. Environment setup and imports
2. Dataset download and preparation
3. Model definition and loss functions
4. Training loop and checkpoints
5. Evaluation of PSNR / MS-SSIM / bitrate

Key training parameters defined in the notebook include:

```python
ITERS = 10_000
BATCH = 16
PATCH = 256
SEED = 42
```

These can be adjusted for a quick experiment or a longer, more meaningful run.

## Metrics

The notebook reports real compression performance using bits per pixel (bpp) and image quality metrics:

- PSNR
- MS-SSIM
- actual encoded bitstream size from the entropy coder

This is implemented via a codec round-trip that compresses and decompresses images and measures the resulting bitstream length.

## Custom arithmetic coder (implemented from scratch)

Beyond using CompressAI's entropy coder as a black box, I implemented my own arithmetic coder (section 12 of the notebook) and validated it against CompressAI:

- **Coder:** a standard Witten-Neal-Cleary arithmetic coder with 32-bit integer precision, bit-level renormalization with pending-bit handling, and 16-bit probability tables (`pmf_to_cdf`, `ac_encode`, `ac_decode`). No external entropy-coding library is used for it.
- **Probability model:** per-channel PMFs are extracted from the trained factorized prior (`_logits_cumulative` of CompressAI's `EntropyBottleneck`) and fed into my coder.
- **Validation:**
  - the encode → decode round-trip is checked to be lossless (asserted for every channel)
  - the actual bit count is compared against the theoretical bound Σ -log2 p
  - the actual bit count is compared against CompressAI's real `compress()` output
- **End-to-end demo:** all latent channels are encoded and decoded with my coder, then passed through the decoder (`g_s`). The reconstruction is compared with CompressAI's `decompress()` output, with bpp, PSNR and max pixel difference reported.

Limitation: values outside the table range (`-K..K`) are clipped to the edge bin instead of using an escape symbol, and the coder is pure Python, so it is much slower than CompressAI's C++ implementation. It is intended for validation and understanding, not speed.

## Notes

- GPU use is strongly recommended for practical training speed.
- The default `ITERS` value is for a lightweight trial.
- For more meaningful comparisons, consider increasing training iterations to around 50k-200k depending on hardware.
- The notebook is designed for experimentation, so many hyperparameters are intentionally configurable.
