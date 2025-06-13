# TensorRT Tiling Upscaler

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)

A high-performance, GPU-accelerated batch image upscaler using NVIDIA TensorRT. This script employs a sophisticated tiling strategy with blending to process extremely high-resolution images that wouldn't otherwise fit into GPU memory.

It's designed for speed and efficiency, making it ideal for upscaling large collections of images, including professional RAW formats.

## Key Features

-   **🚀 Blazing Fast Performance**: Leverages NVIDIA TensorRT for optimized inference on NVIDIA GPUs.
-   **🖼️ Tiling for Large Images**: Intelligently splits large images into smaller tiles, processes them, and seamlessly stitches them back together.
-   **✨ Seamless Blending**: Uses a Gaussian weighting overlap to eliminate visible seams or artifacts between tiles.
-   **🗂️ Batch Processing**: Upscale entire folders of images with a single command.
-   **📸 RAW Format Support**: Natively handles professional camera RAW formats (e.g., `.arw`, `.cr2`, `.nef`) thanks to `rawpy`.
-   **🔧 Highly Configurable**: Fine-tune the process with command-line arguments for tile size, overlap, output format, and more.
-   **⚡ Parallel Processing**: Utilizes multiple workers for efficient I/O and processing pipelines (though GPU inference is sequential per instance).

---

## ⚠️ Important: Bring Your Own Model (BYOM)

This repository provides the powerful processing script (the "engine"), but **it does not include a pre-trained AI model.**

You must provide your own TensorRT engine file (`.trt`).

### Why?
Models are often subject to different licenses, and this script is designed to be a general-purpose tool compatible with various upscaling architectures (like SWINIR, Real-ESRGAN, etc.).

### How to get a `.trt` model?
You need to convert a pre-trained model (e.g., from `.pth`, `.onnx`, or `.pb` format) into a TensorRT engine. The most common way is to:
1.  Export your model to ONNX format.
2.  Use the NVIDIA TensorRT `trtexec` command-line tool or the TensorRT Python API to convert the ONNX file into a `.trt` engine.

For more information, please refer to the [official NVIDIA TensorRT documentation](https://developer.nvidia.com/tensorrt).

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:
-   **Python 3.8** or newer.
-   An **NVIDIA GPU** with a recent driver.
-   The **NVIDIA CUDA Toolkit** (version should be compatible with your PyCUDA and TensorRT installation).
-   **NVIDIA TensorRT 8.x** or newer.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kyrazzx/TensorRT-Tiling-Upscaler.git
    cd TensorRT-Tiling-Upscaler
    ```

2.  **(Recommended)** Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

For the script to work correctly, organize your files as follows:

```
TensorRT-Tiling-Upscaler/
├── main.py              # The main script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Directory for your TRT models
│   └── your_model.trt      # <--- Place your engine file here
├── input/                  # Default input directory for your images
│   ├── image1.jpg
│   └── photo.cr2
└── output/                 # Default output directory for upscaled results
```

## Usage

Run the script from your terminal. All parameters are configurable via command-line arguments.

### Basic Command

```bash
python main.py --engine_path path/to/your_model.trt --input_dir input --output_dir output
```

### Command-Line Arguments

| Argument          | Description                                                               | Default       |
| ----------------- | ------------------------------------------------------------------------- | ------------- |
| `--input_dir`     | Path to the directory containing images to upscale.                       | `input`       |
| `--output_dir`    | Path to the directory where upscaled images will be saved.                | `output`      |
| `--engine_path`   | **(Required)** Path to your compiled TensorRT (`.trt`) engine file.         | `None`        |
| `--scale_factor`  | The upscaling factor (e.g., 2 for 2x, 4 for 4x).                          | `1`           |
| `--tile_size`     | The size of the square tiles (in pixels). Should match your model's input size. | `4600`        |
| `--overlap`       | The overlap between tiles in pixels to prevent seams.                     | `2000`        |
| `--output_format` | The output image format. Choices: `png`, `jpg`, `jpeg`, `bmp`.            | `jpg`         |
| `--jpg_quality`   | The quality for JPEG output (1-100).                                      | `100`         |
| `--dtype`         | Data type for input tensor preparation. Choices: `float16`, `float32`.    | `float32`     |
| `--workers`       | Number of worker threads for processing. Use 0 for auto (CPU count).      | `1`           |
| `--force`         | Force reprocessing of images even if an output file already exists.       | `False`       |

### Example

This command upscales all images in the `photos_to_process` folder by 4x using a model named `RealESRGAN_x4.trt`. It uses a tile size of 256x256 with an overlap of 64 pixels and saves the results as high-quality PNGs.

```bash
python main.py \
    --input_dir "photos_to_process" \
    --output_dir "processed_photos" \
    --engine_path "models/RealESRGAN_x4.trt" \
    --scale_factor 4 \
    --tile_size 256 \
    --overlap 64 \
    --output_format png
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/somethingcool`).
3.  Commit your changes (`git commit -m 'Add something cool'`).
4.  Push to the branch (`git push origin feature/SomethingCool`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This script relies on several fantastic open-source libraries:
-   [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
-   [PyCUDA](https://mathema.tician.de/software/pycuda/)
-   [OpenCV](https://opencv.org/)
-   [Pillow](https://python-pillow.org/)
-   [RawPy](https://github.com/letmaik/rawpy)
-   [tqdm](https://github.com/tqdm/tqdm)
