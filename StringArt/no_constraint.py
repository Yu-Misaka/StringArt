import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
# import deque
import os
import time
from scipy.ndimage import sobel
import multiprocessing
# ***** Import Array and Lock for shared memory *****
from multiprocessing import Pool, Array, Lock
from functools import partial
import ctypes # To work with shared memory Array types

# --- Constants ---
IMG_SIZE = 500
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 1500 # segments
MIN_DISTANCE = 15
DEFAULT_LINE_WEIGHT = 25
SCALE = 2
DEFAULT_EDGE_WEIGHT = 0.8
DEFAULT_WORKERS = os.cpu_count()

# --- Global Variables for Shared Memory (Initialized in main) ---
# These will hold references usable by worker processes
shared_error_buffer = None
shared_edge_buffer = None
shared_rows_buffer = None
shared_cols_buffer = None
shared_segment_indices = None # Dictionary mapping segment_key -> (start_idx, end_idx)
shared_img_shape = None

# --- Helper Functions (Preprocessing, Pins, Line Pixels - unchanged) ---
# (Keep load_and_preprocess_image, calculate_pin_coords, get_line_pixels)
def load_and_preprocess_image(image_path, target_size, enhance_contrast=True):
    """Loads, preprocesses image, and generates edge map."""
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA' or img.mode == 'P': img = img.convert('RGB')
        img = img.convert('L')
    except FileNotFoundError: print(f"Error: File not found {image_path}"); return None, None
    except Exception as e: print(f"Error opening image: {e}"); return None, None

    width, height = img.size
    short_side = min(width, height)
    left = (width - short_side) / 2; top = (height - short_side) / 2
    right = (width + short_side) / 2; bottom = (height + short_side) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    if enhance_contrast:
        img = ImageOps.autocontrast(img, cutoff=1)
        print("  Applied auto-contrast enhancement.")

    mask = Image.new('L', (target_size, target_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((1, 1, target_size-1, target_size-1), fill=255)

    img_array = np.array(img, dtype=np.float32)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    img_array = img_array * mask_array + 255.0 * (1.0 - mask_array)
    img_array = np.clip(img_array, 0, 255)

    sx = sobel(img_array, axis=0, mode='constant', cval=255.0)
    sy = sobel(img_array, axis=1, mode='constant', cval=255.0)
    edge_map = np.hypot(sx, sy)
    if edge_map.max() > 0:
        edge_map = (edge_map / edge_map.max()) * 255.0
    edge_map = edge_map * mask_array
    print("  Calculated edge map.")

    return img_array.astype(np.float32), edge_map.astype(np.float32)

def calculate_pin_coords(num_pins, size):
    """Calculates the (x, y) coordinates of pins."""
    center = size / 2.0; radius = size / 2.0 - 1; coords = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = int(round(center + radius * math.cos(angle)))
        y = int(round(center + radius * math.sin(angle)))
        x = max(0, min(size - 1, x)); y = max(0, min(size - 1, y))
        coords.append((x, y))
    return coords

def get_line_pixels(p1, p2, img_size):
    """Gets pixel coordinates for a line."""
    x0, y0 = p1; x1, y1 = p2
    dist = math.ceil(math.sqrt((x1 - x0)**2 + (y1 - y0)**2))
    n_steps = int(dist)
    if n_steps <= 0: return np.array([]), np.array([])

    x_coords = np.linspace(x0, x1, n_steps + 1)
    y_coords = np.linspace(y0, y1, n_steps + 1)
    rows = np.clip(np.round(y_coords).astype(int), 0, img_size - 1)
    cols = np.clip(np.round(x_coords).astype(int), 0, img_size - 1)

    indices = np.lexsort((cols, rows))
    coords = np.vstack((rows[indices], cols[indices])).T
    unique_mask = np.concatenate(([True], np.any(coords[1:] != coords[:-1], axis=1)))
    unique_coords = coords[unique_mask]
    if len(unique_coords) == 0: return np.array([]), np.array([])
    return unique_coords[:, 0], unique_coords[:, 1]


# --- Precalculation (Modified to return flat arrays for shared memory) ---
def precalculate_lines_for_shared_mem(pin_coords, num_pins, img_size, min_distance):
    """Pre-calculates lines and returns data structured for shared memory."""
    print("Pre-calculating lines for shared memory...")
    start_time = time.time()
    segment_pixel_indices = {} # Map (i,j) -> (start_offset, num_pixels)
    all_rows_list = []
    all_cols_list = []
    current_offset = 0
    calculation_count = 0

    for i in range(num_pins):
        for j in range(i + 1, num_pins):
            dist = min((j - i) % num_pins, (i - j) % num_pins)
            if dist < min_distance: continue
            p1 = pin_coords[i]; p2 = pin_coords[j]
            rows, cols = get_line_pixels(p1, p2, img_size)
            num_pixels = len(rows)
            if num_pixels > 0:
                segment_key = (i, j)
                all_rows_list.append(rows)
                all_cols_list.append(cols)
                segment_pixel_indices[segment_key] = (current_offset, num_pixels)
                current_offset += num_pixels
                calculation_count += 1
        # Progress indicator can be added here if needed

    # Concatenate all pixel coordinates into large flat arrays
    flat_rows = np.concatenate(all_rows_list).astype(np.int32) # Use fixed size int
    flat_cols = np.concatenate(all_cols_list).astype(np.int32)

    end_time = time.time()
    print(f"Line pre-calculation finished ({calculation_count} unique lines cached, {len(flat_rows)} total pixels) in {end_time - start_time:.2f} seconds.")
    return segment_pixel_indices, flat_rows, flat_cols


# --- Initialization function for worker processes ---
def init_worker(error_buf, edge_buf, rows_buf, cols_buf, seg_indices, img_shape):
    """Initializer for worker processes to store shared memory refs."""
    global shared_error_buffer, shared_edge_buffer, shared_rows_buffer, shared_cols_buffer
    global shared_segment_indices, shared_img_shape
    shared_error_buffer = error_buf
    shared_edge_buffer = edge_buf
    shared_rows_buffer = rows_buf
    shared_cols_buffer = cols_buf
    shared_segment_indices = seg_indices
    shared_img_shape = img_shape
    # print(f"Worker {os.getpid()} initialized.") # Optional debug print


# --- Parallel Worker Function (Using Shared Memory) ---
def worker_find_best_segment_shared(segment_keys_chunk, edge_weight_factor):
    """
    Worker function using shared memory.
    Finds the best segment within a given chunk of keys.
    """
    # Reconstruct NumPy arrays from shared memory INSIDE the worker
    # Error image needs acquisition of lock for read safety, although less critical if main process only writes between iterations
    with shared_error_buffer.get_lock():
        error_np = np.frombuffer(shared_error_buffer.get_obj(), dtype=np.float32).reshape(shared_img_shape)
    # Edge map is read-only, lock might be less critical but safer
    with shared_edge_buffer.get_lock():
        edge_np = np.frombuffer(shared_edge_buffer.get_obj(), dtype=np.float32).reshape(shared_img_shape)
    # Pixel coordinate buffers (read-only within worker)
    rows_np = np.frombuffer(shared_rows_buffer.get_obj(), dtype=np.int32)
    cols_np = np.frombuffer(shared_cols_buffer.get_obj(), dtype=np.int32)


    local_best_segment = None
    local_max_score = -np.inf

    for segment_pins in segment_keys_chunk:
        start_idx, num_pixels = shared_segment_indices[segment_pins]
        end_idx = start_idx + num_pixels

        # Get pixel coordinates for this segment from the shared flat arrays
        rows = rows_np[start_idx:end_idx]
        cols = cols_np[start_idx:end_idx]

        # Calculate scores using the views of shared arrays
        # Ensure coordinates are valid indices for error_np/edge_np shape
        if rows.size > 0: # Check needed if get_line_pixels could return empty slices
            # Clip indices just in case (shouldn't be necessary if get_line_pixels is correct)
            # rows = np.clip(rows, 0, shared_img_shape[0] - 1)
            # cols = np.clip(cols, 0, shared_img_shape[1] - 1)

            try:
                darkness_error = np.sum(error_np[rows, cols])
                edge_score = np.sum(edge_np[rows, cols])
                current_score = darkness_error + edge_weight_factor * edge_score

                if current_score > local_max_score:
                    local_max_score = current_score
                    local_best_segment = segment_pins
            except IndexError:
                # This might happen if coordinates are somehow out of bounds
                print(f"Warning: IndexError accessing shared array for segment {segment_pins}. Rows max: {np.max(rows) if rows.size>0 else 'N/A'}, Cols max: {np.max(cols) if cols.size>0 else 'N/A'}. Shape: {shared_img_shape}")
                continue


    return local_max_score, local_best_segment


# --- Main Algorithm (Parallel Segment Selection using Shared Memory) ---
def generate_string_art_segments_parallel_shared(image_array, edge_map, num_pins, max_segments, min_distance, line_weight, edge_weight_factor, pin_coords, segment_pixel_indices, flat_rows, flat_cols, num_workers):
    """Generates segments using parallel processing and shared memory."""
    global shared_error_buffer, shared_edge_buffer, shared_rows_buffer, shared_cols_buffer
    global shared_segment_indices, shared_img_shape

    print(f"Starting PARALLEL (Shared Memory) generation (Workers={num_workers}, Segments={max_segments})...")
    start_time = time.time()

    img_size = image_array.shape[0]
    shared_img_shape = image_array.shape # Store shape for workers

    # --- Create Shared Memory Arrays ---
    error_size_flat = image_array.size
    shared_error_buffer = Array(ctypes.c_float, error_size_flat)
    shared_edge_buffer = Array(ctypes.c_float, error_size_flat)

    pixel_data_size = flat_rows.size
    shared_rows_buffer = Array(ctypes.c_int, pixel_data_size)
    shared_cols_buffer = Array(ctypes.c_int, pixel_data_size)

    # --- Copy initial data into shared memory ---
    # Error image (acquired lock for initial write)
    error_np_shared = np.frombuffer(shared_error_buffer.get_obj(), dtype=np.float32).reshape(shared_img_shape)
    initial_error = 255.0 - image_array
    np.copyto(error_np_shared, initial_error)

    # Edge map (acquired lock for initial write)
    edge_np_shared = np.frombuffer(shared_edge_buffer.get_obj(), dtype=np.float32).reshape(shared_img_shape)
    np.copyto(edge_np_shared, edge_map)

    # Pixel coordinate data
    rows_np_shared = np.frombuffer(shared_rows_buffer.get_obj(), dtype=np.int32)
    cols_np_shared = np.frombuffer(shared_cols_buffer.get_obj(), dtype=np.int32)
    np.copyto(rows_np_shared, flat_rows)
    np.copyto(cols_np_shared, flat_cols)

    # Segment indices dictionary doesn't need shared memory itself, just passed to initializer
    shared_segment_indices = segment_pixel_indices
    print("Shared memory arrays initialized.")

    # --- Setup Output Image ---
    output_res = img_size * SCALE
    output_image_pil = Image.new('L', (output_res, output_res), 0) # Black bg
    draw_output = ImageDraw.Draw(output_image_pil)
    segment_list = []

    # --- Prepare for Pooling ---
    all_segment_keys = list(segment_pixel_indices.keys())
    if not all_segment_keys: return [], output_image_pil
    num_segments_total = len(all_segment_keys)
    chunk_size = math.ceil(num_segments_total / num_workers)

    # --- Create Pool with Initializer ---
    # The initializer passes the shared memory references to each worker *once* when it starts.
    with Pool(processes=num_workers,
              initializer=init_worker,
              initargs=(shared_error_buffer, shared_edge_buffer,
                        shared_rows_buffer, shared_cols_buffer,
                        shared_segment_indices, shared_img_shape)) as pool:
        print("Worker pool initialized.")
        # --- Main Loop ---
        for segment_num in range(max_segments):
            step_start_time = time.time()

            # Prepare arguments for the workers for this iteration
            # Only edge_weight_factor changes (or doesn't) - everything else is global via initializer
            bound_worker_func = partial(worker_find_best_segment_shared,
                                        edge_weight_factor=edge_weight_factor)

            key_chunks = [all_segment_keys[i:i + chunk_size] for i in range(0, num_segments_total, chunk_size)]

            # Distribute work
            results = pool.map(bound_worker_func, key_chunks)

            # Aggregate results
            global_best_segment = None
            global_max_score = -np.inf
            for score, segment in results:
                if segment is not None and score > global_max_score:
                    global_max_score = score
                    global_best_segment = segment

            step_find_time = time.time()

            # --- Process Best Segment ---
            if global_best_segment is None or global_max_score <= 0:
                 if global_best_segment is None: print(f"Warning: No suitable segment found at step {segment_num + 1}.")
                 else: print(f"Stopping at step {segment_num + 1}: Max score non-positive ({global_max_score:.2f}).")
                 break

            segment_list.append(global_best_segment)
            pin_a, pin_b = global_best_segment

            # Retrieve pixel coordinates (from shared memory via indices dict)
            start_idx, num_pixels = shared_segment_indices[(pin_a, pin_b)]
            end_idx = start_idx + num_pixels
            # Lock the coordinate buffers while reading
            with shared_rows_buffer.get_lock(), shared_cols_buffer.get_lock():
                rows = np.copy(rows_np_shared[start_idx:end_idx]) # Copy needed coords
                cols = np.copy(cols_np_shared[start_idx:end_idx])

            # --- Update error image IN SHARED MEMORY (Sequential Part - needs lock) ---
            with shared_error_buffer.get_lock():
                # Get a view OF THE SHARED BUFFER again to update it
                error_np_shared_view = np.frombuffer(shared_error_buffer.get_obj(), dtype=np.float32).reshape(shared_img_shape)
                if rows.size > 0: # Ensure there are pixels to update
                    error_np_shared_view[rows, cols] -= line_weight
                    np.clip(error_np_shared_view[rows, cols], 0, None, out=error_np_shared_view[rows, cols]) # Clip in place


            # Draw line (Sequential Part)
            p1_scaled = (pin_coords[pin_a][0] * SCALE, pin_coords[pin_a][1] * SCALE)
            p2_scaled = (pin_coords[pin_b][0] * SCALE, pin_coords[pin_b][1] * SCALE)
            draw_output.line([p1_scaled, p2_scaled], fill=255, width=1)

            step_end_time = time.time()

            if (segment_num + 1) % 10 == 0 or segment_num < 10: # Report more frequently initially
                print(f"  Segment {segment_num + 1}/{max_segments} | Best Score: {global_max_score:.0f} | Time: {(step_end_time - step_start_time)*1000:.1f} ms (Find: {(step_find_time - step_start_time)*1000:.1f} ms)")

    total_end_time = time.time()
    print(f"Segment generation finished ({len(segment_list)} segments generated) in {total_end_time - start_time:.2f} seconds.")
    return segment_list, output_image_pil


# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows packaging

    parser = argparse.ArgumentParser(description="Generate String Art - Optimal Segments (Parallel Shared Memory).")
    # Arguments remain mostly the same
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("-o", "--output_segments", default="string_art_segments.txt", help="Output file for line segments")
    parser.add_argument("-p", "--output_png", default="string_art_preview_segments_parallel_sm.png", help="Output file for the preview image") # New default name
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins (default: {DEFAULT_N_PINS})")
    parser.add_argument("--segments", "--lines", type=int, default=DEFAULT_MAX_LINES, dest='max_segments', help=f"Max segments (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Line darkness weight (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Minimum pin distance (default: {MIN_DISTANCE})")
    parser.add_argument("--previewpins", action='store_true', help="Draw pin markers on preview.")
    parser.add_argument("--ew", "--edge_weight", type=float, default=DEFAULT_EDGE_WEIGHT, dest='edge_weight', help=f"Weight factor for edges (default: {DEFAULT_EDGE_WEIGHT:.2f})")
    parser.add_argument("--no_contrast", action='store_false', dest='enhance_contrast', help="Disable automatic contrast enhancement.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of worker processes (default: use all cores)")

    args = parser.parse_args()

    # Validation
    if args.pins < 3 or args.max_segments < 1 or args.weight <= 0 or args.size < 50 or args.mindist < 1 or args.edge_weight < 0 or args.workers < 1:
        print("Error: Invalid parameter values."); exit(1)
    if args.mindist >= args.pins // 2: print(f"Warning: Minimum distance ({args.mindist}) may be large.")
    actual_workers = min(args.workers, os.cpu_count()); print(f"Using {actual_workers} worker processes.")

    # 1. Load image and edge map
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array, edge_map_array = load_and_preprocess_image(
        args.input_image, args.size, enhance_contrast=args.enhance_contrast)
    if processed_image_array is None: exit(1)

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # 3. Pre-calculate Lines for Shared Memory
    segment_pixel_indices, flat_rows, flat_cols = precalculate_lines_for_shared_mem(
        pin_coordinates, args.pins, args.size, args.mindist)
    if not segment_pixel_indices: print("Error: No valid lines pre-calculated."); exit(1)

    # 4. Generate String Art using Shared Memory Parallelism
    segment_list, preview_image = generate_string_art_segments_parallel_shared(
        processed_image_array,
        edge_map_array,
        args.pins,
        args.max_segments,
        args.mindist,
        args.weight,
        args.edge_weight,
        pin_coordinates,
        segment_pixel_indices, # Pass dict mapping segment to index/len
        flat_rows,             # Pass flat array of row coords
        flat_cols,             # Pass flat array of col coords
        actual_workers
    )

    # 5. Save Segment List
    try:
        with open(args.output_segments, 'w') as f:
            for seg in segment_list: f.write(f"{seg[0]},{seg[1]}\n")
        print(f"Line segments saved to '{args.output_segments}'")
    except IOError as e: print(f"Error saving segments file: {e}")

    # 6. Save Preview Image
    try:
        if args.previewpins:
            draw_preview = ImageDraw.Draw(preview_image)
            pin_marker_radius = max(1, SCALE * 1)
            for x, y in pin_coordinates:
                scaled_x = x * SCALE; scaled_y = y * SCALE
                draw_preview.ellipse((int(scaled_x-pin_marker_radius), int(scaled_y-pin_marker_radius), int(scaled_x+pin_marker_radius), int(scaled_y+pin_marker_radius)), fill=255)
        preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e: print(f"Error saving preview image: {e}")
    except Exception as e: print(f"An unexpected error occurred: {e}")

    print("Done.")