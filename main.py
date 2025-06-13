import numpy as np
import tensorrt_rtx as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image, ImageFile
import cv2
import os
import time
from pathlib import Path
import rawpy
import concurrent.futures
import argparse
from tqdm.auto import tqdm
import sys
import traceback

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(image_path: Path):
    ext = image_path.suffix.lower()
    try:
        if ext in ['.arw', '.cr2', '.nef', '.dng', '.raf', '.raw']:
            with rawpy.imread(str(image_path)) as raw:
                return raw.postprocess(use_camera_wb=True, output_bps=8)
        else:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    except FileNotFoundError:
        print(f"Error: File not found - {image_path}")
        return None
    except (rawpy.LibRawError, IOError, SyntaxError) as e:
        print(f"Error loading {image_path.name} ({type(e).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {image_path.name}: {e}")
        traceback.print_exc()
        return None

def process_image(
    image_path: Path,
    context: trt.IExecutionContext,
    bindings: list,
    d_input: cuda.DeviceAllocation,
    d_output: cuda.DeviceAllocation,
    stream: cuda.Stream,
    input_shape: tuple,
    output_shape: tuple,
    output_np_dtype: np.dtype,
    output_dir: Path,
    scale_factor: int,
    tile_size: int,
    overlap: int,
    output_format: str,
    jpg_quality: int,
    numpy_dtype: np.dtype
):
    start_time = time.time()
    img = load_image(image_path)
    if img is None:
        return None
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        print(f"Error: Image {image_path.name} is not in the expected RGB format.")
        return None

    h, w = img.shape[:2]
    output_h, output_w = h * scale_factor, w * scale_factor
    output_img = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight_map = np.zeros((output_h, output_w, 3), dtype=np.float32)

    y_grid, x_grid = np.mgrid[0:tile_size, 0:tile_size].astype(np.float64)
    center = (tile_size - 1) / 2.0
    sigma = tile_size / 4.0
    weight_template = np.exp(-((x_grid - center)**2 + (y_grid - center)**2) / (2 * sigma**2))
    weight_template = np.repeat(weight_template[:, :, np.newaxis], 3, axis=2).astype(np.float32)
    scaled_tile_size = tile_size * scale_factor
    scaled_weight_template = cv2.resize(weight_template, (scaled_tile_size, scaled_tile_size), interpolation=cv2.INTER_LINEAR)

    stride = tile_size - overlap
    x_tiles = (w + stride - 1) // stride
    y_tiles = (h + stride - 1) // stride
    total_tiles = max(1, x_tiles * y_tiles)
    tile_times = []

    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=numpy_dtype)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=output_np_dtype)

    with tqdm(total=total_tiles, desc=f"Tiles ({image_path.name})", unit="tile",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
              leave=False) as pbar:
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                tile_start_time = time.time()
                y_start, y_end = y, min(y + tile_size, h)
                x_start, x_end = x, min(x + tile_size, w)
                tile_orig = img[y_start:y_end, x_start:x_end]
                tile_h, tile_w = tile_orig.shape[:2]

                if tile_h < tile_size or tile_w < tile_size:
                    pad_h, pad_w = tile_size - tile_h, tile_size - tile_w
                    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
                    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
                    tile = cv2.copyMakeBorder(tile_orig, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
                else:
                    tile = tile_orig

                try:
                    tile_input = tile.transpose(2, 0, 1).astype(numpy_dtype) / 255.0
                    tile_input = np.expand_dims(tile_input, axis=0)

                    if tile_input.shape != tuple(input_shape):
                        print(f"\nError: Tile shape {tile_input.shape} != expected {input_shape} for {image_path.name}")
                        pbar.update(1)
                        continue

                    np.copyto(h_input, tile_input.ravel())
                    cuda.memcpy_htod_async(d_input, h_input, stream)
                    context.execute_async_v3(stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(h_output, d_output, stream)
                    stream.synchronize()

                    tile_output_full = h_output.reshape(output_shape)
                    if tile_output_full.shape[0] == 1:
                        tile_output_full = tile_output_full[0]

                    if tile_output_full.dtype == np.float16:
                        tile_output_full = tile_output_full.astype(np.float32)

                    tile_output_full = tile_output_full.transpose(1, 2, 0)

                    out_y_start, out_x_start = y_start * scale_factor, x_start * scale_factor
                    out_h_orig, out_w_orig = tile_h * scale_factor, tile_w * scale_factor

                    if tile_h < tile_size or tile_w < tile_size:
                        pad_top_scaled, pad_left_scaled = pad_top * scale_factor, pad_left * scale_factor
                        end_row, end_col = pad_top_scaled + out_h_orig, pad_left_scaled + out_w_orig
                        if end_row > tile_output_full.shape[0] or end_col > tile_output_full.shape[1]:
                            print(f"\nWarning: Invalid cropping boundaries for padding {image_path.name} ({y},{x}).")
                            tile_output = tile_output_full
                        else:
                            tile_output = tile_output_full[pad_top_scaled:end_row, pad_left_scaled:end_col, :]
                    else:
                        tile_output = tile_output_full

                    out_y_end = out_y_start + tile_output.shape[0]
                    out_x_end = out_x_start + tile_output.shape[1]

                    if out_y_end > output_img.shape[0] or out_x_end > output_img.shape[1]:
                        out_y_end = min(out_y_end, output_img.shape[0])
                        out_x_end = min(out_x_end, output_img.shape[1])
                        tile_output = tile_output[:out_y_end-out_y_start, :out_x_end-out_x_start, :]

                    current_weight = scaled_weight_template[:tile_output.shape[0], :tile_output.shape[1], :]

                    if tile_output.shape == current_weight.shape:
                        output_img[out_y_start:out_y_end, out_x_start:out_x_end] += tile_output * current_weight
                        weight_map[out_y_start:out_y_end, out_x_start:out_x_end] += current_weight
                    else:
                        print(f"\nWarning: Mismatch in tile/weight shapes for {image_path.name} ({y},{x}).")

                except Exception as e:
                    print(f"\nError during TRT processing of tile {image_path.name} ({y},{x}): {e}")
                    traceback.print_exc()

                tile_time = time.time() - tile_start_time
                tile_times.append(tile_time)
                if tile_times:
                    avg_tile_time = sum(tile_times) / len(tile_times)
                    remaining_tiles = total_tiles - pbar.n - 1
                    if remaining_tiles > 0:
                        estimated_remaining_time = remaining_tiles * avg_tile_time
                        pbar.set_postfix_str(f"Avg: {avg_tile_time:.3f}s/tile, Remaining: {estimated_remaining_time:.1f}s")
                    else:
                        pbar.set_postfix_str(f"Avg: {avg_tile_time:.3f}s/tile")
                pbar.update(1)

    mask = weight_map > 1e-6
    output_img[mask] /= weight_map[mask]
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)

    output_filename = f"{image_path.stem}_upscaled_SWINIR-L-GAN.{output_format.lower()}"
    output_path = output_dir / output_filename
    try:
        output_image = Image.fromarray(output_img)
        save_params = {}
        fmt = output_format.lower()
        if fmt in ['jpg', 'jpeg']:
            save_params['quality'] = jpg_quality
            save_params['subsampling'] = 0
        elif fmt == 'png':
            save_params['compress_level'] = 4
        output_image.save(output_path, **save_params)
        elapsed_time = time.time() - start_time
        avg_tile_t = sum(tile_times) / len(tile_times) if tile_times else 0
        tqdm.write(f"Finished: {image_path.name} -> {output_path.name} (time: {elapsed_time:.2f}s, avg tile: {avg_tile_t:.3f}s)")
        return output_path
    except Exception as e:
        print(f"\nError saving {output_path.name}: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='TensorRT Batch Upscaling.')
    parser.add_argument('--input_dir', type=str, default='input')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--engine_path', type=str, default='scunet_color_real_gan_4600fp32rtx.trt')
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--tile_size', type=int, default=4600)
    parser.add_argument('--overlap', type=int, default=2000)
    parser.add_argument('--output_format', type=str, default='jpg', choices=['png', 'jpg', 'jpeg', 'bmp'])
    parser.add_argument('--jpg_quality', type=int, default=100)
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'])
    args = parser.parse_args()

    if args.workers == 0:
        args.workers = os.cpu_count() or 1
    print(f"Using {args.workers} workers.")
    if args.workers < 0:
        print("Error: Negative number of workers.")
        sys.exit(1)

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    engine_path = Path(args.engine_path)
    if not input_dir.is_dir():
        print(f"Error: Input dir '{input_dir}' not found.")
        sys.exit(1)
    if not engine_path.is_file():
        print(f"Error: Engine file '{engine_path}' not found.")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    numpy_dtype = np.float16 if args.dtype == 'float16' else np.float32
    print(f"Using {args.dtype} ({numpy_dtype.__name__}) for input preparation.")

    print(f"--- DEBUG: TensorRT Python version: {trt.__version__} ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    print(f"Loading TensorRT engine: {engine_path}")
    engine, context = None, None
    try:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f"TensorRT engine loaded successfully.")
        context = engine.create_execution_context()
        if not context:
            raise RuntimeError("Cannot create execution context.")
        print("TensorRT execution context created.")
    except Exception as e:
        print(f"Error loading engine/creating context: {e}")
        traceback.print_exc()
        sys.exit(1)

    input_binding_idx, output_binding_idx = -1, -1
    input_shape, output_shape = None, None
    input_name, output_name = "", ""
    input_dtype_trt, output_dtype_trt = None, None
    output_np_dtype = np.float32

    print("--- Engine Tensor Information (TRT 10.x API) ---")
    num_io_tensors = engine.num_io_tensors
    tensor_indices = list(range(num_io_tensors))

    for i in tensor_indices:
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)

        print(f"Tensor {i}: Name='{name}', Shape={shape}, Dtype={dtype}, Mode={mode}")

        if mode == trt.TensorIOMode.INPUT:
            if input_binding_idx != -1:
                print("Warning: Multiple input tensors found.")
            else:
                input_binding_idx = i
                input_shape = tuple(shape)
                input_name = name
                input_dtype_trt = dtype
                if len(input_shape) == 4 and (input_shape[2] != args.tile_size or input_shape[3] != args.tile_size):
                    print(f"WARNING: Engine tile size ({input_shape[2]}x{input_shape[3]}) != --tile_size ({args.tile_size}).")
                elif len(input_shape) != 4:
                    print(f"WARNING: Input shape {input_shape} is not 4D.")
        elif mode == trt.TensorIOMode.OUTPUT:
            if output_binding_idx != -1:
                print("Warning: Multiple output tensors found.")
            else:
                output_binding_idx = i
                output_shape = tuple(shape)
                output_name = name
                output_dtype_trt = dtype
                if output_dtype_trt == trt.float16:
                    output_np_dtype = np.float16
                elif output_dtype_trt == trt.int32:
                    output_np_dtype = np.int32

    if not input_name or not output_name:
        print("Error: Input/output tensor not identified by name.")
        sys.exit(1)

    print(f"Identified input: Index={input_binding_idx}, Name='{input_name}', Shape={input_shape}, Type={input_dtype_trt}")
    print(f"Identified output: Index={output_binding_idx}, Name='{output_name}', Shape={output_shape}, Type={output_dtype_trt} (NumPy: {output_np_dtype.__name__})")
    print("------------------------------------")

    if len(input_shape) == 4 and (args.tile_size != input_shape[2] or args.tile_size != input_shape[3]):
        print(f"Correcting tile_size to {input_shape[2]}x{input_shape[3]} according to the engine.")
        args.tile_size = input_shape[2]
    elif len(input_shape) != 4:
        print(f"Cannot correct tile_size, input is not 4D.")

    d_input, d_output, stream, bindings = None, None, None, None
    try:
        if numpy_dtype == np.float16 and input_dtype_trt != trt.float16:
            print(f"WARNING: Input dtype {numpy_dtype} != expected TRT dtype {input_dtype_trt}")
        elif numpy_dtype == np.float32 and input_dtype_trt == trt.float16:
            print(f"WARNING: Input dtype {numpy_dtype} != expected TRT dtype {input_dtype_trt}")

        d_input_size = trt.volume(input_shape) * np.dtype(numpy_dtype).itemsize
        d_input = cuda.mem_alloc(d_input_size)
        d_output_size = trt.volume(output_shape) * np.dtype(output_np_dtype).itemsize
        d_output = cuda.mem_alloc(d_output_size)
        stream = cuda.Stream()

        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        bindings = None

        print("Device (GPU) memory allocated.")
        print(f"  Input buffer: {d_input_size / (1024**2):.2f} MiB")
        print(f"  Output buffer: {d_output_size / (1024**2):.2f} MiB (type: {output_np_dtype.__name__})")

    except cuda.MemoryError:
        print(f"Error: Insufficient GPU memory!")
        sys.exit(1)
    except Exception as e:
        print(f"Error during GPU memory allocation: {e}")
        traceback.print_exc()
        sys.exit(1)

    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.arw', '.cr2', '.nef', '.dng', '.raf', '.raw']
    all_image_paths = []
    print(f"Searching in: {input_dir}")
    for ext in supported_extensions:
        all_image_paths.extend(input_dir.glob(f'*{ext}'))
        all_image_paths.extend(input_dir.glob(f'*{ext.upper()}'))

    image_paths_to_process = []
    processed_stems = set()
    output_suffix = f"_ESC-XL-REAL-GAN.{args.output_format.lower()}"
    existing_outputs_stems = set()
    
    if not args.force:
        for f in output_dir.glob(f'*{output_suffix}'):
            existing_outputs_stems.add(f.stem.replace('_ESC-XL-REAL-GAN', ''))
        if existing_outputs_stems:
            print(f"Found {len(existing_outputs_stems)} existing outputs.")

    for img_path in all_image_paths:
        stem = img_path.stem
        if stem in processed_stems:
            continue
        if not args.force and stem in existing_outputs_stems:
            continue
        processed_stems.add(stem)
        image_paths_to_process.append(img_path)

    if not image_paths_to_process:
        print(f"No new images to process.")
        return
    
    print(f"Found {len(image_paths_to_process)} images to process.")

    results = []
    start_overall_time = time.time()
    
    if args.workers > 1:
        print(f"Starting processing with {args.workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_image, img_path, context, bindings, d_input, d_output, stream,
                    input_shape, output_shape, output_np_dtype, output_dir, args.scale_factor,
                    args.tile_size, args.overlap, args.output_format, args.jpg_quality, numpy_dtype
                ): img_path for img_path in image_paths_to_process
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                              desc="Overall progress", unit="image"):
                img_path_completed = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    tqdm.write(f"\nERROR in thread for {img_path_completed.name}: {e}")
    else:
        print("Starting sequential processing...")
        for image_path in tqdm(image_paths_to_process, desc="Overall progress", unit="image"):
            try:
                result = process_image(
                    image_path, context, bindings, d_input, d_output, stream,
                    input_shape, output_shape, output_np_dtype, output_dir, args.scale_factor,
                    args.tile_size, args.overlap, args.output_format, args.jpg_quality, numpy_dtype
                )
                if result:
                    results.append(result)
            except Exception as e:
                tqdm.write(f"\nERROR processing {image_path.name}: {e}")
                traceback.print_exc()

    end_overall_time = time.time()
    total_time = end_overall_time - start_overall_time
    num_processed = len(results)
    avg_time_per_image = total_time / num_processed if num_processed > 0 else 0
    
    print("-" * 30)
    print(f"Processing finished in {total_time:.2f} s.")
    print(f"Successfully saved {num_processed} images.")
    if num_processed > 0:
        print(f"Average time per image: {avg_time_per_image:.2f} s.")
    
    failed_count = len(image_paths_to_process) - num_processed
    if failed_count > 0:
        print(f"Failed/skipped images: {failed_count}")
    
    print(f"Output files in: {output_dir}")

if __name__ == "__main__":
    main()
