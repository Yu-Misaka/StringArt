import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
from collections import deque # Not needed for segments, but keep imports clean
import os
import time
from scipy.ndimage import sobel # Keep for edge weighting

# --- Constants ---
IMG_SIZE = 500
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 1500 # Default might be lower now as convergence is faster
MIN_DISTANCE = 15
# MIN_LOOP is irrelevant now
DEFAULT_LINE_WEIGHT = 25
SCALE = 2
DEFAULT_EDGE_WEIGHT = 0.8

# --- Helper Functions (Preprocessing, Pins, Line Pixels - mostly unchanged) ---

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

    # --- Edge Map Calculation ---
    sx = sobel(img_array, axis=0, mode='constant', cval=255.0)
    sy = sobel(img_array, axis=1, mode='constant', cval=255.0)
    edge_map = np.hypot(sx, sy)
    if edge_map.max() > 0:
        edge_map = (edge_map / edge_map.max()) * 255.0
    edge_map = edge_map * mask_array # Apply circle mask
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
    if n_steps <= 0: return np.array([]), np.array([]) # Return empty if no steps

    x_coords = np.linspace(x0, x1, n_steps + 1)
    y_coords = np.linspace(y0, y1, n_steps + 1)
    rows = np.clip(np.round(y_coords).astype(int), 0, img_size - 1)
    cols = np.clip(np.round(x_coords).astype(int), 0, img_size - 1)

    indices = np.lexsort((cols, rows))
    coords = np.vstack((rows[indices], cols[indices])).T
    unique_mask = np.concatenate(([True], np.any(coords[1:] != coords[:-1], axis=1)))
    unique_coords = coords[unique_mask]
    if len(unique_coords) == 0: return np.array([]), np.array([]) # Handle empty unique case
    return unique_coords[:, 0], unique_coords[:, 1]

def precalculate_lines(pin_coords, num_pins, img_size, min_distance):
    """Pre-calculates pixel coordinates for valid lines."""
    print("Pre-calculating lines...")
    start_time = time.time(); line_cache = {}; calculation_count = 0
    for i in range(num_pins):
        # Optimization: only calculate j > i
        for j in range(i + 1, num_pins):
            dist = min((j - i) % num_pins, (i - j) % num_pins)
            if dist < min_distance: continue
            p1 = pin_coords[i]; p2 = pin_coords[j]
            rows, cols = get_line_pixels(p1, p2, img_size)
            # Only store if the line has pixels
            if len(rows) > 0:
                # Store only one direction (i, j) where i < j to simplify global search
                line_cache[(i, j)] = (rows, cols)
                calculation_count += 1
        if (i + 1) % 50 == 0 or i == num_pins - 1: print(f"  Pre-calculated lines originating from pin {i+1}/{num_pins}...")
    end_time = time.time()
    print(f"Line pre-calculation finished ({calculation_count} unique lines cached) in {end_time - start_time:.2f} seconds.")
    return line_cache

# --- Main Algorithm (Global Segment Selection) ---

def generate_string_art_segments(image_array, edge_map, num_pins, max_segments, min_distance, line_weight, edge_weight_factor, pin_coords, line_cache):
    """Generates the sequence of best line *segments*."""
    print(f"Starting string art generation (Max Segments={max_segments}, weight={line_weight}, edge_factor={edge_weight_factor:.2f})...")
    start_time = time.time()

    img_size = image_array.shape[0]
    error_image = 255.0 - image_array # High value = dark area in original

    output_res = img_size * SCALE
    output_image_pil = Image.new('L', (output_res, output_res), 0) # Black bg
    draw_output = ImageDraw.Draw(output_image_pil)

    segment_list = [] # Store the chosen segments (pin_a, pin_b)

    # --- Main Loop ---
    for segment_num in range(max_segments):
        best_segment = None
        max_score = -np.inf

        # ***** Global Search: Iterate through ALL cached line segments *****
        # line_cache now only contains (i, j) where i < j
        for segment_pins, (rows, cols) in line_cache.items():
            # No need to check length rows>0 here, already done in precalc

            # Calculate darkness error for this line
            darkness_error = np.sum(error_image[rows, cols])

            # Calculate edge score for this line
            edge_score = np.sum(edge_map[rows, cols])

            # Combine scores
            current_score = darkness_error + edge_weight_factor * edge_score

            if current_score > max_score:
                max_score = current_score
                best_segment = segment_pins # Store the tuple (pin_a, pin_b)

        # --- Process the Globally Best Segment Found ---
        if best_segment is None or max_score <= 0: # Stop if no improvement possible
             if best_segment is None:
                 print(f"Warning: No suitable segment found at step {segment_num + 1}. Stopping.")
             else:
                 print(f"Stopping at step {segment_num + 1}: Max score non-positive ({max_score:.2f}). Image likely saturated.")
             break

        segment_list.append(best_segment)
        pin_a, pin_b = best_segment

        # Get line pixels for the chosen segment
        # We know (pin_a, pin_b) is in the cache because we just iterated through it
        rows, cols = line_cache[(pin_a, pin_b)]

        # Subtract darkness (error reduction)
        error_image[rows, cols] -= line_weight
        np.clip(error_image, 0, None, out=error_image)

        # Draw line on the output image (white)
        p1_scaled = (pin_coords[pin_a][0] * SCALE, pin_coords[pin_a][1] * SCALE)
        p2_scaled = (pin_coords[pin_b][0] * SCALE, pin_coords[pin_b][1] * SCALE)
        draw_output.line([p1_scaled, p2_scaled], fill=255, width=1)

        # Progress indicator (might be slower now)
        if (segment_num + 1) % 50 == 0: # Report less frequently?
            print(f"  Generated segment {segment_num + 1}/{max_segments} (Score: {max_score:.0f})")
            # Optional: Save intermediate preview
            # output_image_pil.save(f"segment_preview_{segment_num+1}.png")

    end_time = time.time()
    # Note: Computation time per step will be higher due to the global search
    print(f"Segment generation finished ({len(segment_list)} segments generated) in {end_time - start_time:.2f} seconds.")
    return segment_list, output_image_pil

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate String Art using Globally Optimal Line Segments.")
    parser.add_argument("input_image", help="Path to the input image file.")
    # Changed output file names/descriptions
    parser.add_argument("-o", "--output_segments", default="string_art_segments.txt", help="Output file for line segments (pin_a,pin_b per line)")
    parser.add_argument("-p", "--output_png", default="string_art_preview_segments.png", help="Output file for the preview image")
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins (default: {DEFAULT_N_PINS})")
    # Renamed --lines to --segments
    parser.add_argument("--segments", "--lines", type=int, default=DEFAULT_MAX_LINES, dest='max_segments', help=f"Maximum number of line segments to generate (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Line darkness weight (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Minimum pin distance (default: {MIN_DISTANCE})")
    # Removed --minloop argument as it's not used
    parser.add_argument("--previewpins", action='store_true', help="Draw pin markers on preview.")
    parser.add_argument("--ew", "--edge_weight", type=float, default=DEFAULT_EDGE_WEIGHT, dest='edge_weight', help=f"Weight factor for edges (default: {DEFAULT_EDGE_WEIGHT:.2f})")
    parser.add_argument("--no_contrast", action='store_false', dest='enhance_contrast', help="Disable automatic contrast enhancement.")

    args = parser.parse_args()

    # Validation
    if args.pins < 3 or args.max_segments < 1 or args.weight <= 0 or args.size < 50 or args.mindist < 1 or args.edge_weight < 0:
        print("Error: Invalid parameter values."); exit(1)
    if args.mindist >= args.pins // 2: print(f"Warning: Minimum distance ({args.mindist}) may be large.")

    # 1. Load, preprocess image, AND get edge map
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array, edge_map_array = load_and_preprocess_image(
        args.input_image, args.size, enhance_contrast=args.enhance_contrast)
    if processed_image_array is None: exit(1)

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # 3. Pre-calculate Lines (Cache stores only unique pairs i < j)
    line_pixel_cache = precalculate_lines(pin_coordinates, args.pins, args.size, args.mindist)
    if not line_pixel_cache: print("Error: No valid lines pre-calculated."); exit(1)
    print(f"Number of unique line segments in cache: {len(line_pixel_cache)}")


    # 4. Generate String Art using Global Segment Selection
    segment_list, preview_image = generate_string_art_segments(
        processed_image_array,
        edge_map_array,
        args.pins,
        args.max_segments, # Use the renamed arg
        args.mindist,      # Still needed for precalc
        args.weight,
        args.edge_weight,
        pin_coordinates,
        line_pixel_cache
    )

    # 5. Save the Segment List
    try:
        # Save as "pin_a,pin_b" per line
        with open(args.output_segments, 'w') as f:
            for seg in segment_list:
                f.write(f"{seg[0]},{seg[1]}\n")
        print(f"Line segments saved to '{args.output_segments}'")
    except IOError as e: print(f"Error saving segments file: {e}")

    # 6. Save the Preview Image
    try:
        if args.previewpins:
            draw_preview = ImageDraw.Draw(preview_image)
            pin_marker_radius = max(1, SCALE * 1)
            for x, y in pin_coordinates:
                scaled_x = x * SCALE; scaled_y = y * SCALE
                draw_preview.ellipse((int(scaled_x-pin_marker_radius), int(scaled_y-pin_marker_radius), int(scaled_x+pin_marker_radius), int(scaled_y+pin_marker_radius)), fill=255) # White pins
        preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e: print(f"Error saving preview image: {e}")
    except Exception as e: print(f"An unexpected error occurred: {e}")

    print("Done.")