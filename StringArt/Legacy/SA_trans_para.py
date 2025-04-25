# === Imports ===
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
from collections import deque
import os
import time
# === Add multiprocessing imports ===
import multiprocessing

# === Constants (remain the same) ===
IMG_SIZE = 500
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 4000
MIN_DISTANCE = 20
MIN_LOOP = 20
DEFAULT_LINE_WEIGHT = 10
SCALE = 15
DEFAULT_ERROR_POWER = 1.5
DEFAULT_LINE_BRIGHTNESS = 0.08


# === Helper Functions (load_and_preprocess_image, calculate_pin_coords, get_line_pixels remain the same) ===
def load_and_preprocess_image(image_path, target_size, apply_equalization=True):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA' or img.mode == 'P': img = img.convert('RGB')
        img = img.convert('L')
        if apply_equalization:
            # print("Applying histogram equalization...") # Reduce noise
            img = ImageOps.equalize(img)
    except FileNotFoundError: return None
    except Exception: return None
    width, height = img.size
    short_side = min(width, height)
    left, top = (width - short_side) / 2, (height - short_side) / 2
    right, bottom = (width + short_side) / 2, (height + short_side) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    mask = Image.new('L', (target_size, target_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((1, 1, target_size-1, target_size-1), fill=255)
    img_array = np.array(img, dtype=np.float32)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    img_array = img_array * mask_array + 255.0 * (1.0 - mask_array)
    img_array = np.clip(img_array, 0, 255)
    return img_array.astype(np.float32)

def calculate_pin_coords(num_pins, size):
    center = size / 2.0; radius = size / 2.0 - 1
    coords = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = int(round(center + radius * math.cos(angle)))
        y = int(round(center + radius * math.sin(angle)))
        x = max(0, min(size - 1, x)); y = max(0, min(size - 1, y))
        coords.append((x, y))
    return coords

def get_line_pixels(p1, p2, img_size):
    x0, y0 = p1; x1, y1 = p2
    dist = math.ceil(math.sqrt((x1 - x0)**2 + (y1 - y0)**2))
    n_steps = int(dist)
    if n_steps <= 0: return np.array([y0]), np.array([x0])
    x_coords = np.linspace(x0, x1, n_steps + 1)
    y_coords = np.linspace(y0, y1, n_steps + 1)
    rows = np.clip(np.round(y_coords).astype(int), 0, img_size - 1)
    cols = np.clip(np.round(x_coords).astype(int), 0, img_size - 1)
    indices = np.lexsort((cols, rows))
    coords = np.vstack((rows[indices], cols[indices])).T
    unique_mask = np.concatenate(([True], np.any(coords[1:] != coords[:-1], axis=1)))
    unique_coords = coords[unique_mask]
    return unique_coords[:, 0], unique_coords[:, 1]

# --- Parallel Pre-calculation ---

# Global variables for worker processes (initialized once per worker)
worker_pin_coords = None
worker_img_size = None

def init_worker(pin_coords_data, img_size_data):
    """Initializer function for each worker process."""
    global worker_pin_coords, worker_img_size
    worker_pin_coords = pin_coords_data
    worker_img_size = img_size_data
    # print(f"Worker {os.getpid()} initialized.") # Optional debug print

def calculate_line_task(pin_indices):
    """The function executed by each worker process."""
    global worker_pin_coords, worker_img_size
    i, j = pin_indices
    try:
        p1 = worker_pin_coords[i]
        p2 = worker_pin_coords[j]
        rows, cols = get_line_pixels(p1, p2, worker_img_size)
        # Return the original indices along with the results
        return i, j, rows, cols
    except Exception as e:
        print(f"Error in worker {os.getpid()} processing pins {i},{j}: {e}")
        return i, j, None, None # Return None on error

def precalculate_lines_parallel(pin_coords, num_pins, img_size, min_distance):
    """Pre-calculates pixel coordinates in parallel using multiprocessing."""
    print("Pre-calculating lines (parallel)...")
    start_time = time.time()

    # 1. Generate list of tasks (pairs of pin indices)
    tasks = []
    for i in range(num_pins):
        for j in range(i + 1, num_pins):
            dist = min((j - i) % num_pins, (i - j) % num_pins)
            if dist < min_distance:
                continue
            tasks.append((i, j))

    print(f"  Generated {len(tasks)} line calculation tasks.")
    if not tasks:
        print("  No tasks to run.")
        return {}

    line_cache = {}
    # Determine number of processes (use all available cores by default)
    num_workers = os.cpu_count()
    print(f"  Starting parallel calculation with {num_workers} workers...")

    try:
        # 2. Create a Pool of worker processes
        # Use initializer to pass read-only data efficiently
        with multiprocessing.Pool(processes=num_workers,
                                initializer=init_worker,
                                initargs=(pin_coords, img_size)) as pool:

            # 3. Map tasks to the pool (starmap unpacks arguments)
            # pool.map will block until all results are ready
            results = pool.map(calculate_line_task, tasks)

        # 4. Process results and populate the cache
        print("  Parallel calculation finished. Processing results...")
        calculation_count = 0
        for i, j, rows, cols in results:
            if rows is not None and cols is not None:
                line_cache[(i, j)] = (rows, cols)
                line_cache[(j, i)] = (rows, cols) # Cache reverse direction too
                calculation_count += 1
            else:
                 print(f"  Warning: Failed to calculate line for pins ({i}, {j}).")


        end_time = time.time()
        print(f"Line pre-calculation finished ({calculation_count}/{len(tasks)} lines cached) in {end_time - start_time:.2f} seconds.")
        return line_cache

    except Exception as e:
        print(f"An error occurred during parallel pre-calculation: {e}")
        # Fallback or re-raise? For simplicity, return empty cache
        return {}


# === Main Algorithm (generate_string_art remains the same) ===
def generate_string_art(image_array, num_pins, max_lines, min_distance, min_loop,
                        line_weight, error_power, line_brightness,
                        pin_coords, line_cache):
    print(f"Starting string art generation (error_power={error_power}, line_brightness={line_brightness})...")
    start_time = time.time()
    img_size = image_array.shape[0]
    error_image = 255.0 - image_array
    output_res = img_size * SCALE
    output_accumulator = np.zeros((output_res, output_res), dtype=np.float32)
    line_sequence = [0]
    current_pin = 0
    last_pins = deque(maxlen=min_loop)

    for line_num in range(max_lines):
        best_pin = -1
        max_weighted_error = -np.inf
        for offset in range(min_distance, num_pins - min_distance):
            test_pin = (current_pin + offset) % num_pins
            if test_pin in last_pins: continue
            if (current_pin, test_pin) not in line_cache: continue
            rows, cols = line_cache[(current_pin, test_pin)]
            current_line_error_pixels = error_image[rows, cols]
            positive_error_mask = current_line_error_pixels > 0
            if np.any(positive_error_mask):
                 weighted_error = np.sum(current_line_error_pixels[positive_error_mask] ** error_power)
            else: weighted_error = 0
            if weighted_error > max_weighted_error:
                max_weighted_error = weighted_error; best_pin = test_pin

        if best_pin == -1: break
        line_sequence.append(best_pin)
        rows, cols = line_cache[(current_pin, best_pin)]
        error_image[rows, cols] -= line_weight
        np.clip(error_image, 0, None, out=error_image)
        p1_scaled = (pin_coords[current_pin][0] * SCALE, pin_coords[current_pin][1] * SCALE)
        p2_scaled = (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)
        out_rows, out_cols = get_line_pixels(p1_scaled, p2_scaled, output_res)
        output_accumulator[out_rows, out_cols] += line_brightness
        last_pins.append(current_pin)
        current_pin = best_pin
        if (line_num + 1) % 200 == 0: print(f"  Generated line {line_num + 1}/{max_lines}") # Reduce noise

    # print("Normalizing accumulator...") # Reduce noise
    non_zero_accumulator = output_accumulator[output_accumulator > 0]
    if non_zero_accumulator.size > 0:
        clip_value = np.percentile(non_zero_accumulator, 99.0)
        clip_value = max(clip_value, line_brightness * 2)
        clipped_accumulator = np.clip(output_accumulator, 0, clip_value)
        normalized_output = (clipped_accumulator / clip_value) * 255.0
    else: normalized_output = np.zeros_like(output_accumulator)
    final_image_array = np.clip(normalized_output, 0, 255).astype(np.uint8)
    output_image_pil = Image.fromarray(final_image_array, mode='L')
    end_time = time.time()
    print(f"String art generation finished ({len(line_sequence)-1} lines drawn) in {end_time - start_time:.2f} seconds.")
    return line_sequence, output_image_pil

# === Main Execution ===
if __name__ == "__main__":
    # --- Argument Parsing (remains the same) ---
    parser = argparse.ArgumentParser(description="Generate String Art (Semi-Transparent White Lines on Black BG).")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("-o", "--output_seq", default="string_art_sequence.txt", help="Output file for the pin sequence.")
    parser.add_argument("-p", "--output_png", default="string_art_preview_accum.png", help="Output file for the preview image.")
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins (default: {DEFAULT_N_PINS})")
    parser.add_argument("--lines", type=int, default=DEFAULT_MAX_LINES, help=f"Maximum number of lines (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Weight subtracted from error map per line (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing image size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Minimum pin distance (default: {MIN_DISTANCE})")
    parser.add_argument("--minloop", type=int, default=MIN_LOOP, help=f"Recent pins to avoid reusing (default: {MIN_LOOP})")
    parser.add_argument("--equalize", action='store_true', default=True, help="Apply histogram equalization to input image.")
    parser.add_argument("--no-equalize", action='store_false', dest='equalize', help="Do NOT apply histogram equalization.")
    parser.add_argument("--error_power", type=float, default=DEFAULT_ERROR_POWER, help=f"Exponent for error calculation (default: {DEFAULT_ERROR_POWER})")
    parser.add_argument("--brightness", type=float, default=DEFAULT_LINE_BRIGHTNESS, help=f"Brightness added per line to output (default: {DEFAULT_LINE_BRIGHTNESS})")
    parser.add_argument("--previewpins", action='store_true', help="Draw markers for pins on the preview image.")
    parser.add_argument("--negative", action='store_true', help="Color negation.")
    args = parser.parse_args()

    # --- Main Steps ---
    # 1. Load and preprocess
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array = load_and_preprocess_image(args.input_image, args.size, args.equalize)
    if processed_image_array is None: print("Error loading image."); exit(1)

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # 3. ***** Call the PARALLEL pre-calculation function *****
    line_pixel_cache = precalculate_lines_parallel(pin_coordinates, args.pins, args.size, args.mindist)
    if not line_pixel_cache: print("Error: Line pre-calculation failed."); exit(1)

    # 4. Generate String Art Sequence & Image (uses the cache)
    sequence, preview_image = generate_string_art(
        processed_image_array, args.pins, args.lines, args.mindist, args.minloop,
        args.weight, args.error_power, args.brightness,
        pin_coordinates, line_pixel_cache
    )

    # 5. Save Sequence
    try:
        with open(args.output_seq, 'w') as f: f.write(','.join(map(str, sequence)))
        print(f"Pin sequence saved to '{args.output_seq}'")
    except IOError as e: print(f"Error saving sequence file: {e}")

    # 6. Save Preview Image
    try:
        if args.previewpins:
            draw_preview = ImageDraw.Draw(preview_image)
            pin_marker_radius = max(1, SCALE * 1)
            for x, y in pin_coordinates:
                scaled_x = x * SCALE; scaled_y = y * SCALE
                draw_preview.ellipse(
                    (int(scaled_x-pin_marker_radius), int(scaled_y-pin_marker_radius),
                     int(scaled_x+pin_marker_radius), int(scaled_y+pin_marker_radius)),
                    fill=255)
        if args.negative:
          preview_image.save(args.output_png)
        else:
          preview_image = preview_image.point(lambda _: 255-_)
          preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e: print(f"Error saving preview image: {e}")
    except Exception as e: print(f"An unexpected error occurred saving preview: {e}")

    print("Done.")