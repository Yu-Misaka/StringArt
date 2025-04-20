import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
from collections import deque
import os
import time

# --- Constants ---
IMG_SIZE = 500
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 2500 # Reduced default as thicker lines have more impact
MIN_DISTANCE = 20
MIN_LOOP = 20
DEFAULT_LINE_WEIGHT = 15 # Error reduction per line (can stay relatively fixed)
SCALE = 20
# *** NEW: Line Thickness parameters ***
DEFAULT_MIN_WIDTH = 1   # Minimum thickness for any line
DEFAULT_MAX_WIDTH = 3   # Maximum thickness for the most "important" lines

# --- Helper Functions (load_and_preprocess_image, calculate_pin_coords, get_line_pixels, precalculate_lines - remain the same as the Black Background version) ---
def load_and_preprocess_image(image_path, target_size):
    """Loads, crops, resizes, grayscales, and circle-crops the image."""
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA' or img.mode == 'P':
             img = img.convert('RGB')
        img = img.convert('L')
    except FileNotFoundError:
        print(f"Error: Input image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening or processing image: {e}")
        return None

    width, height = img.size
    short_side = min(width, height)
    left = (width - short_side) / 2
    top = (height - short_side) / 2
    right = (width + short_side) / 2
    bottom = (height + short_side) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    mask = Image.new('L', (target_size, target_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((1, 1, target_size-1, target_size-1), fill=255)

    img_array = np.array(img, dtype=np.float32)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    img_array = img_array * mask_array + 255.0 * (1.0 - mask_array)
    img_array = np.clip(img_array, 0, 255) # Clip before returning

    return img_array.astype(np.float32)


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


def precalculate_lines(pin_coords, num_pins, img_size, min_distance):
    """Pre-calculates pixel coordinates for all valid lines between pins."""
    print("Pre-calculating lines...")
    start_time = time.time()
    line_cache = {}
    calculation_count = 0

    for i in range(num_pins):
        for j in range(i + 1, num_pins):
            dist = min((j - i) % num_pins, (i - j) % num_pins)
            if dist < min_distance:
                 continue

            p1 = pin_coords[i]
            p2 = pin_coords[j]
            rows, cols = get_line_pixels(p1, p2, img_size)
            line_cache[(i, j)] = (rows, cols)
            line_cache[(j, i)] = (rows, cols)
            calculation_count += 1

        if (i + 1) % 50 == 0 or i == num_pins - 1:
             print(f"  Pre-calculated lines originating from pin {i+1}/{num_pins}...")

    end_time = time.time()
    print(f"Line pre-calculation finished ({calculation_count} lines cached) in {end_time - start_time:.2f} seconds.")
    return line_cache


# --- Main Algorithm ---

def generate_string_art(
    image_array, num_pins, max_lines, min_distance, min_loop, line_weight,
    pin_coords, line_cache, min_width, max_width # Added thickness params
    ):
    """Generates the sequence of pins for the string art with variable line thickness."""
    print(f"Starting string art generation (variable thickness {min_width}-{max_width})...")
    start_time = time.time()

    img_size = image_array.shape[0]
    error_image = 255.0 - image_array # High value = dark area in original

    output_res = img_size * SCALE
    output_image_pil = Image.new('L', (output_res, output_res), 0) # Black background
    draw_output = ImageDraw.Draw(output_image_pil)

    line_sequence = []
    current_pin = 0
    line_sequence.append(current_pin)
    last_pins = deque(maxlen=min_loop)

    # --- Main Loop ---
    for line_num in range(max_lines):
        best_pin = -1
        max_error_reduction = -np.inf
        best_line_pixels = None # Store pixels for the best line

        for offset in range(min_distance, num_pins - min_distance):
            test_pin = (current_pin + offset) % num_pins
            if test_pin in last_pins: continue
            if (current_pin, test_pin) not in line_cache: continue

            rows, cols = line_cache[(current_pin, test_pin)]
            if len(rows) == 0: continue # Skip zero-length lines

            line_error = np.sum(error_image[rows, cols])

            if line_error > max_error_reduction:
                max_error_reduction = line_error
                best_pin = test_pin
                best_line_pixels = (rows, cols) # Store the pixels

        if best_pin == -1 or best_line_pixels is None:
            print(f"Warning: No suitable next pin found at line {line_num + 1}. Stopping early.")
            break

        line_sequence.append(best_pin)
        rows, cols = best_line_pixels

        # --- Calculate Line Width ---
        line_length = len(rows)
        # Normalize error per pixel (average error along the line)
        # Max possible average error is 255.0
        avg_error_per_pixel = max(0.0, max_error_reduction / line_length) if line_length > 0 else 0.0
        # Map normalized error (0.0 to 1.0) to width range
        normalized_importance = min(1.0, avg_error_per_pixel / 255.0) # Clamp between 0 and 1
        current_width = min_width + (max_width - min_width) * normalized_importance
        # Round to nearest integer for drawing, ensure it's at least min_width
        draw_width = max(min_width, int(round(current_width)))

        # --- Update Error Image (still use fixed weight for stability) ---
        error_image[rows, cols] -= line_weight
        np.clip(error_image, 0, None, out=error_image)

        # --- Draw Line with Calculated Width ---
        p1_scaled = (pin_coords[current_pin][0] * SCALE, pin_coords[current_pin][1] * SCALE)
        p2_scaled = (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)
        draw_output.line([p1_scaled, p2_scaled], fill=255, width=draw_width) # Use calculated width

        # --- Update State ---
        last_pins.append(current_pin)
        current_pin = best_pin

        if (line_num + 1) % 200 == 0:
            print(f"  Generated line {line_num + 1}/{max_lines} (width: {draw_width})")

    end_time = time.time()
    print(f"String art generation finished ({len(line_sequence)-1} lines drawn) in {end_time - start_time:.2f} seconds.")
    return line_sequence, output_image_pil


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate String Art Pin Sequence (Variable Thickness).")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("-o", "--output_seq", default="string_art_sequence.txt", help="Output file for the pin sequence.")
    parser.add_argument("-p", "--output_png", default="string_art_preview_varthick.png", help="Output file for the preview image.")
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins (default: {DEFAULT_N_PINS})")
    parser.add_argument("--lines", type=int, default=DEFAULT_MAX_LINES, help=f"Max number of lines (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Error reduction per line (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing image size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Min pin distance (default: {MIN_DISTANCE})")
    parser.add_argument("--minloop", type=int, default=MIN_LOOP, help=f"Min loop avoidance (default: {MIN_LOOP})")
    parser.add_argument("--minwidth", type=int, default=DEFAULT_MIN_WIDTH, help=f"Min line thickness (default: {DEFAULT_MIN_WIDTH})")
    parser.add_argument("--maxwidth", type=int, default=DEFAULT_MAX_WIDTH, help=f"Max line thickness (default: {DEFAULT_MAX_WIDTH})")
    parser.add_argument("--previewpins", action='store_true', help="Draw pin markers on the preview.")

    args = parser.parse_args()

    # Validation
    if args.minwidth < 1 or args.maxwidth < args.minwidth:
        print("Error: Invalid thickness parameters. Ensure minwidth >= 1 and maxwidth >= minwidth.")
        exit(1)
    # Other validations (pins, lines, etc.) from previous version...
    if args.pins < 3 or args.lines < 1 or args.weight <= 0 or args.size < 50 or args.mindist < 1 or args.minloop < 0:
        print("Error: Invalid parameter values.")
        exit(1)
    if args.mindist >= args.pins // 2:
         print(f"Warning: Min distance ({args.mindist}) may be too large for the number of pins ({args.pins}).")

    # 1. Load and preprocess
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array = load_and_preprocess_image(args.input_image, args.size)
    if processed_image_array is None: exit(1)

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # 3. Pre-calculate Lines
    line_pixel_cache = precalculate_lines(pin_coordinates, args.pins, args.size, args.mindist)
    if not line_pixel_cache: exit(1)

    # 4. Generate String Art Sequence
    sequence, preview_image = generate_string_art(
        processed_image_array,
        args.pins,
        args.lines,
        args.mindist,
        args.minloop,
        args.weight,
        pin_coordinates,
        line_pixel_cache,
        args.minwidth, # Pass thickness args
        args.maxwidth
    )

    # 5. Save Sequence
    try:
        with open(args.output_seq, 'w') as f:
            f.write(','.join(map(str, sequence)))
        print(f"Pin sequence saved to '{args.output_seq}'")
    except IOError as e: print(f"Error saving sequence file: {e}")

    # 6. Save Preview Image
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
                    fill=255 # White dots for pins
                )
        preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e: print(f"Error saving preview image: {e}")
    except Exception as e: print(f"An unexpected error occurred while saving the preview: {e}")

    print("Done.")