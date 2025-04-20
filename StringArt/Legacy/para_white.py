import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
from collections import deque
import os
import time
import multiprocessing # Import the multiprocessing library

# --- Constants (Defaults, can be overridden by command-line args) ---
# ... (Keep constants the same) ...
IMG_SIZE = 500
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 4000
MIN_DISTANCE = 20
MIN_LOOP = 20
DEFAULT_LINE_WEIGHT = 10
SCALE = 15


# --- Helper Functions ---
# ... (load_and_preprocess_image, calculate_pin_coords, get_line_pixels remain the same) ...
def load_and_preprocess_image(image_path, target_size):
    """Loads, crops, resizes, grayscales, and circle-crops the image."""
    try:
        img = Image.open(image_path)
        # Ensure image is in a mode that can be converted to 'L' (grayscale)
        if img.mode == 'RGBA' or img.mode == 'P':
             img = img.convert('RGB')
        img = img.convert('L') # Convert to grayscale
    except FileNotFoundError:
        print(f"Error: Input image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening or processing image: {e}")
        return None

    # Square crop from center
    width, height = img.size
    short_side = min(width, height)
    left = (width - short_side) / 2
    top = (height - short_side) / 2
    right = (width + short_side) / 2
    bottom = (height + short_side) / 2
    img = img.crop((left, top, right, bottom))

    # Resize
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Create circular mask
    mask = Image.new('L', (target_size, target_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    # Draw ellipse slightly smaller to avoid harsh edges right at the boundary
    mask_draw.ellipse((1, 1, target_size-1, target_size-1), fill=255)

    # Convert image and mask to numpy arrays for processing
    img_array = np.array(img, dtype=np.float32)
    mask_array = np.array(mask, dtype=np.float32) / 255.0 # Normalize mask to 0.0-1.0

    # Apply mask: Keep pixels inside circle, make outside pixels white (255)
    # Where mask is 1, use img_array pixel. Where mask is 0, use 255.
    img_array = img_array * mask_array + 255.0 * (1.0 - mask_array)
    # Clip values just in case, and convert back to uint8 for potential saving/viewing
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Return the processed NumPy array
    # We need float32 for calculations later, so convert back before returning
    return img_array.astype(np.float32)


def calculate_pin_coords(num_pins, size):
    """Calculates the (x, y) coordinates of pins on the circumference."""
    center = size / 2.0
    radius = size / 2.0 - 1 # Small offset to keep pins inside bounds
    coords = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        # Calculate coordinates and ensure they are within image bounds
        x = int(round(center + radius * math.cos(angle)))
        y = int(round(center + radius * math.sin(angle)))
        x = max(0, min(size - 1, x)) # Clamp coordinates to image dimensions
        y = max(0, min(size - 1, y))
        coords.append((x, y))
    return coords

def get_line_pixels(p1, p2, img_size):
    """Gets all integer pixel coordinates for a line between two points using linspace."""
    x0, y0 = p1
    x1, y1 = p2

    # Calculate distance to determine number of steps
    dist = math.ceil(math.sqrt((x1 - x0)**2 + (y1 - y0)**2))
    n_steps = int(dist) # Number of points to interpolate

    if n_steps <= 0:
        return np.array([y0]), np.array([x0])

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


# --- Main Algorithm ---
# ... (generate_string_art remains the same, using the cache provided) ...
def generate_string_art(image_array, num_pins, max_lines, min_distance, min_loop, line_weight, pin_coords, line_cache):
    """Generates the sequence of pins for the string art."""
    print(f"Starting string art generation (pins={num_pins}, max_lines={max_lines}, weight={line_weight})...")
    start_time = time.time()

    img_size = image_array.shape[0]

    # --- Initialization ---
    error_image = 255.0 - image_array # error_image is float32

    output_res = img_size * SCALE
    output_image_pil = Image.new('L', (output_res, output_res), 255) # White background
    draw_output = ImageDraw.Draw(output_image_pil)

    line_sequence = []
    current_pin = 0
    line_sequence.append(current_pin)
    last_pins = deque(maxlen=min_loop)

    # --- Main Loop ---
    for line_num in range(max_lines):
        best_pin = -1
        max_error_reduction = -np.inf

        potential_pins = []
        for offset in range(min_distance, num_pins - min_distance):
            test_pin = (current_pin + offset) % num_pins
            # Check if pin is valid and line exists in cache
            if test_pin not in last_pins and (current_pin, test_pin) in line_cache:
                 potential_pins.append(test_pin)
        
        if not potential_pins:
             # Check if *any* line can be drawn from current_pin, ignoring last_pins if necessary
             fallback_pins = []
             for offset in range(min_distance, num_pins - min_distance):
                 test_pin = (current_pin + offset) % num_pins
                 if (current_pin, test_pin) in line_cache:
                      fallback_pins.append(test_pin)
             
             if not fallback_pins: # Truly stuck
                 print(f"Warning: No possible lines from pin {current_pin} found in cache. Stopping early at line {line_num + 1}.")
                 break
             else: # If stuck only due to last_pins, pick best from fallback
                 potential_pins = fallback_pins
                 # print(f"Note: No optimal move found avoiding recent pins at line {line_num + 1}, considering alternatives.")


        # Find best pin among potential ones
        for test_pin in potential_pins:
            rows, cols = line_cache[(current_pin, test_pin)]
            line_error = np.sum(error_image[rows, cols])

            if line_error > max_error_reduction:
                max_error_reduction = line_error
                best_pin = test_pin

        # --- Update State ---
        if best_pin == -1:
            # This might happen if all potential pins have error_reduction <= 0, or no valid pins left
             if not potential_pins: # Already handled above, but double-check
                 print(f"Critical Warning: No best_pin selected and no potential pins at line {line_num + 1}. Stopping.")
             else:
                 print(f"Warning: No pin found with positive error reduction at line {line_num + 1} (best reduction was {max_error_reduction:.2f}). Picking best available or stopping.")
                 # Optional: could pick the one with max_error_reduction even if negative/zero, or just stop. Stopping is safer.
                 # best_pin = potential_pins[0] # Example: Just pick the first available if stuck
                 break # Stop if no improvement can be made

        line_sequence.append(best_pin)
        rows, cols = line_cache[(current_pin, best_pin)]

        error_image[rows, cols] -= line_weight
        np.clip(error_image, 0, None, out=error_image)

        p1_scaled = (pin_coords[current_pin][0] * SCALE, pin_coords[current_pin][1] * SCALE)
        p2_scaled = (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)
        draw_output.line([p1_scaled, p2_scaled], fill=0, width=1)

        last_pins.append(current_pin)
        current_pin = best_pin

        if (line_num + 1) % 200 == 0:
            print(f"  Generated line {line_num + 1}/{max_lines}")

    end_time = time.time()
    print(f"String art generation finished ({len(line_sequence)-1} lines drawn) in {end_time - start_time:.2f} seconds.")
    return line_sequence, output_image_pil


# --- Main Execution ---
if __name__ == "__main__":
    # Necessary for multiprocessing safety on some platforms (Windows)
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Generate String Art Pin Sequence from an Image.")
    # ... (Arguments remain the same) ...
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("-o", "--output_seq", default="string_art_sequence.txt", help="Output file for the pin sequence (default: string_art_sequence.txt)")
    parser.add_argument("-p", "--output_png", default="string_art_preview.png", help="Output file for the preview image (default: string_art_preview.png)")
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins around the circle (default: {DEFAULT_N_PINS})")
    parser.add_argument("--lines", type=int, default=DEFAULT_MAX_LINES, help=f"Maximum number of lines (strings) to generate (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Weight (error reduction) per line (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing image size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Minimum pin distance for connections (default: {MIN_DISTANCE})")
    parser.add_argument("--minloop", type=int, default=MIN_LOOP, help=f"Number of recent pins to avoid reusing (default: {MIN_LOOP})")
    parser.add_argument("--previewpins", action='store_true', help="Draw markers for pins on the preview image.")


    args = parser.parse_args()

    # ... (Validation remains the same) ...
    if args.pins < 3 or args.lines < 1 or args.weight <= 0 or args.size < 50 or args.mindist < 1 or args.minloop < 0:
        print("Error: Invalid parameter values. Please check constraints (e.g., pins >= 3, lines >= 1, weight > 0, etc.).")
        exit(1)
    if args.mindist >= args.pins // 2:
         print(f"Warning: Minimum distance ({args.mindist}) might be too large for the number of pins ({args.pins}), limiting connections.")


    # 1. Load and preprocess image
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array = load_and_preprocess_image(args.input_image, args.size)
    if processed_image_array is None: exit(1)

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # *** CHANGE: Call the parallel pre-calculation function ***
    # 3. Pre-calculate Lines (Parallel)
    line_pixel_cache = precalculate_lines_parallel(pin_coordinates, args.pins, args.size, args.mindist)
    if not line_pixel_cache:
         print("Error: Line pre-calculation failed or produced no results.")
         exit(1)

    # 4. Generate String Art Sequence (uses the cache from step 3)
    sequence, preview_image = generate_string_art(
        processed_image_array, args.pins, args.lines, args.mindist,
        args.minloop, args.weight, pin_coordinates, line_pixel_cache
    )

    # 5. Save the Sequence
    try:
        with open(args.output_seq, 'w') as f:
            f.write(','.join(map(str, sequence)))
        print(f"Pin sequence saved to '{args.output_seq}'")
    except IOError as e:
        print(f"Error saving sequence file: {e}")

    # 6. Save the Preview Image
    try:
        if args.previewpins:
            draw_preview = ImageDraw.Draw(preview_image)
            pin_marker_radius = max(1, SCALE * 1)
            for x, y in pin_coordinates:
                scaled_x = x * SCALE
                scaled_y = y * SCALE
                draw_preview.ellipse(
                    (int(scaled_x - pin_marker_radius), int(scaled_y - pin_marker_radius),
                     int(scaled_x + pin_marker_radius), int(scaled_y + pin_marker_radius)),
                    fill=255 # White pins
                )

        preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e:
        print(f"Error saving preview image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the preview: {e}")

    print("Done.")