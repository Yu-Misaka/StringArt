import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
from collections import deque
import time
import os
import multiprocessing # Import the multiprocessing library
# ***** Add Scipy Import *****
from scipy.ndimage import sobel


# --- Constants ---
IMG_SIZE = 1080
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 4000 # User might reduce this now
MIN_DISTANCE = 15 # Might slightly reduce to allow more connections
MIN_LOOP = 10     # Might slightly reduce
DEFAULT_LINE_WEIGHT = 15
SCALE = 10
# ***** New Constant for Edge Weighting *****
DEFAULT_EDGE_WEIGHT = 0.8 # How much emphasis to put on edges (0 = none, >0 = more emphasis)

# --- Helper Functions ---

# ***** Modified Preprocessing *****
def load_and_preprocess_image(image_path, target_size, enhance_contrast=True):
    """Loads, crops, resizes, grayscales, optionally enhances contrast, and circle-crops."""
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA' or img.mode == 'P':
             img = img.convert('RGB')
        img = img.convert('L')
    except FileNotFoundError:
        print(f"Error: Input image file not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"Error opening or processing image: {e}")
        return None, None

    width, height = img.size
    short_side = min(width, height)
    left = (width - short_side) / 2
    top = (height - short_side) / 2
    right = (width + short_side) / 2
    bottom = (height + short_side) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # ***** Contrast Enhancement Step *****
    if enhance_contrast:
        # Autocontrast stretches histogram to full range
        img = ImageOps.autocontrast(img, cutoff=1) # Cutoff ignores % darkest/lightest pixels
        print("  Applied auto-contrast enhancement.")

    mask = Image.new('L', (target_size, target_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((1, 1, target_size-1, target_size-1), fill=255)

    img_array = np.array(img, dtype=np.float32)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    img_array = img_array * mask_array + 255.0 * (1.0 - mask_array)
    img_array = np.clip(img_array, 0, 255)

    # ***** Calculate Edge Map *****
    # Calculate gradient magnitude using Sobel filters
    sx = sobel(img_array, axis=0, mode='constant', cval=255.0) # Detect horizontal edges
    sy = sobel(img_array, axis=1, mode='constant', cval=255.0) # Detect vertical edges
    edge_map = np.hypot(sx, sy) # Magnitude of gradient
    # Normalize edge map (optional but good practice)
    if edge_map.max() > 0:
        edge_map = (edge_map / edge_map.max()) * 255.0
    # Make edges stronger where original image was darker (optional refinement)
    # edge_map *= (255.0 - img_array) / 255.0

    # Apply the circle mask to the edge map as well
    edge_map = edge_map * mask_array

    print("  Calculated edge map.")
    # Return both the processed image and the edge map
    return img_array.astype(np.float32), edge_map.astype(np.float32)


def calculate_pin_coords(num_pins, size):
    """Calculates the (x, y) coordinates of pins on the circumference."""
    center = size / 2.0
    radius = size / 2.0 - 1
    coords = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = int(round(center + radius * math.cos(angle)))
        y = int(round(center + radius * math.sin(angle)))
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        coords.append((x, y))
    return coords

def get_line_pixels(p1, p2, img_size):
    """Gets all integer pixel coordinates for a line between two points using linspace."""
    x0, y0 = p1
    x1, y1 = p2
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

# --- Main Algorithm ---

# ***** Modified Main Algorithm *****
def generate_string_art(image_array, edge_map, num_pins, max_lines, min_distance, min_loop, line_weight, edge_weight_factor, pin_coords, line_cache):
    """Generates the sequence of pins for the string art, incorporating edge weighting."""
    print(f"Starting string art generation (pins={num_pins}, max_lines={max_lines}, weight={line_weight}, edge_factor={edge_weight_factor:.2f})...")
    start_time = time.time()

    img_size = image_array.shape[0]
    error_image = 255.0 - image_array # High value = dark area in original

    output_res = img_size * SCALE
    output_image_pil = Image.new('L', (output_res, output_res), 0) # Black bg
    draw_output = ImageDraw.Draw(output_image_pil)

    line_sequence = []; current_pin = 0
    line_sequence.append(current_pin)
    last_pins = deque(maxlen=min_loop)

    # --- Main Loop ---
    for line_num in range(max_lines):
        best_pin = -1
        # ***** Use 'max_score' instead of 'max_error_reduction' *****
        max_score = -np.inf # Combined score (darkness + edge bonus)

        for offset in range(min_distance, num_pins - min_distance):
            test_pin = (current_pin + offset) % num_pins
            if test_pin in last_pins: continue
            if (current_pin, test_pin) not in line_cache: continue

            rows, cols = line_cache[(current_pin, test_pin)]
            if len(rows) == 0: continue # Skip empty lines

            # Calculate darkness error for this line
            darkness_error = np.sum(error_image[rows, cols])

            # ***** Calculate edge score for this line *****
            edge_score = np.sum(edge_map[rows, cols])

            # ***** Combine scores *****
            # The score is the darkness error plus a weighted edge score.
            # Higher edge_weight_factor gives more importance to lines crossing edges.
            current_score = darkness_error + edge_weight_factor * edge_score

            if current_score > max_score:
                max_score = current_score
                best_pin = test_pin

        if best_pin == -1:
            print(f"Warning: No suitable next pin found at line {line_num + 1}. Stopping early.")
            break

        line_sequence.append(best_pin)
        rows, cols = line_cache[(current_pin, best_pin)]

        # Subtract darkness (error reduction) - same as before
        error_image[rows, cols] -= line_weight
        np.clip(error_image, 0, None, out=error_image)

        # Draw line on the output image (white)
        p1_scaled = (pin_coords[current_pin][0] * SCALE, pin_coords[current_pin][1] * SCALE)
        p2_scaled = (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)
        draw_output.line([p1_scaled, p2_scaled], fill=255, width=1)

        last_pins.append(current_pin)
        current_pin = best_pin

        if (line_num + 1) % 200 == 0: print(f"  Generated line {line_num + 1}/{max_lines}")

    end_time = time.time()
    print(f"String art generation finished ({len(line_sequence)-1} lines drawn) in {end_time - start_time:.2f} seconds.")
    return line_sequence, output_image_pil


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate String Art Pin Sequence from an Image (Edge Weighted).")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("-o", "--output_seq", default="string_art_sequence.txt", help="Output file for the pin sequence")
    parser.add_argument("-p", "--output_png", default="string_art_preview_ew.png", help="Output file for the preview image") # Changed default name
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins (default: {DEFAULT_N_PINS})")
    parser.add_argument("--lines", type=int, default=DEFAULT_MAX_LINES, help=f"Maximum number of lines (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Line darkness weight (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Minimum pin distance (default: {MIN_DISTANCE})")
    parser.add_argument("--minloop", type=int, default=MIN_LOOP, help=f"Pin reuse avoidance count (default: {MIN_LOOP})")
    parser.add_argument("--previewpins", action='store_true', help="Draw pin markers on preview.")
    # ***** New Argument *****
    parser.add_argument("--ew", "--edge_weight", type=float, default=DEFAULT_EDGE_WEIGHT, dest='edge_weight', help=f"Weight factor for edges in line selection (default: {DEFAULT_EDGE_WEIGHT:.2f})")
    parser.add_argument("--no_contrast", action='store_false', dest='enhance_contrast', help="Disable automatic contrast enhancement during preprocessing.")
    parser.add_argument("--negative", action='store_true', help="Color negation.")

    args = parser.parse_args()

    # Validation
    if args.pins < 3 or args.lines < 1 or args.weight <= 0 or args.size < 50 or args.mindist < 1 or args.minloop < 0 or args.edge_weight < 0:
        print("Error: Invalid parameter values.")
        exit(1)
    if args.mindist >= args.pins // 2: print(f"Warning: Minimum distance ({args.mindist}) may be large.")


    # 1. Load, preprocess image, AND get edge map
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array, edge_map_array = load_and_preprocess_image(
        args.input_image,
        args.size,
        enhance_contrast=args.enhance_contrast # Use command line arg
    )
    if processed_image_array is None: exit(1)

    # Optional: Save debug images
    # Image.fromarray(processed_image_array.astype(np.uint8)).save("preprocessed_debug.png")
    # Image.fromarray(edge_map_array.astype(np.uint8)).save("edge_map_debug.png")
    # print("Saved debug preprocessed_debug.png and edge_map_debug.png")

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # 3. Pre-calculate Lines
    line_pixel_cache = precalculate_lines_parallel(pin_coordinates, args.pins, args.size, args.mindist)
    if not line_pixel_cache: print("Error: No valid lines pre-calculated."); exit(1)

    # 4. Generate String Art Sequence (pass edge map and weight factor)
    sequence, preview_image = generate_string_art(
        processed_image_array,
        edge_map_array, # Pass the edge map
        args.pins,
        args.lines,
        args.mindist,
        args.minloop,
        args.weight,
        args.edge_weight, # Pass the edge weight factor
        pin_coordinates,
        line_pixel_cache
    )

    # 5. Save the Sequence
    try:
        with open(args.output_seq, 'w') as f: f.write(','.join(map(str, sequence)))
        print(f"Pin sequence saved to '{args.output_seq}'")
    except IOError as e: print(f"Error saving sequence file: {e}")

    # 6. Save the Preview Image
    try:
        if args.previewpins:
            draw_preview = ImageDraw.Draw(preview_image)
            pin_marker_radius = max(1, SCALE * 1)
            for x, y in pin_coordinates:
                scaled_x = x * SCALE; scaled_y = y * SCALE
                draw_preview.ellipse((int(scaled_x-pin_marker_radius), int(scaled_y-pin_marker_radius), int(scaled_x+pin_marker_radius), int(scaled_y+pin_marker_radius)), fill=255) # White pins
        if args.negative:
          preview_image.save(args.output_png)
        else:
          preview_image = preview_image.point(lambda _: 255-_)
          preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e: print(f"Error saving preview image: {e}")
    except Exception as e: print(f"An unexpected error occurred: {e}")

    print("Done.")