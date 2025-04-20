import numpy as np
from PIL import Image, ImageDraw, ImageOps
import math
import argparse
from collections import deque
import os
import time

# --- Constants (Defaults, can be overridden by command-line args) ---
IMG_SIZE = 500
DEFAULT_N_PINS = 288
DEFAULT_MAX_LINES = 4000
MIN_DISTANCE = 20
MIN_LOOP = 20
DEFAULT_LINE_WEIGHT = 25 # Adjust as needed
SCALE = 15

# --- Helper Functions (remain the same as before) ---

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

def generate_string_art(image_array, num_pins, max_lines, min_distance, min_loop, line_weight, pin_coords, line_cache):
    """Generates the sequence of pins for the string art."""
    print(f"Starting string art generation (pins={num_pins}, max_lines={max_lines}, weight={line_weight})...")
    start_time = time.time()

    img_size = image_array.shape[0]

    # --- Initialization ---
    # Error image (same as before: high value = dark area in original)
    error_image = 255.0 - image_array # error_image is float32

    # Output image for visualization
    output_res = img_size * SCALE
    # ***** CHANGE 1: Initialize with black background *****
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

        for offset in range(min_distance, num_pins - min_distance):
            test_pin = (current_pin + offset) % num_pins
            if test_pin in last_pins:
                continue
            if (current_pin, test_pin) not in line_cache:
                continue

            rows, cols = line_cache[(current_pin, test_pin)]
            line_error = np.sum(error_image[rows, cols])

            if line_error > max_error_reduction:
                max_error_reduction = line_error
                best_pin = test_pin

        if best_pin == -1:
            print(f"Warning: No suitable next pin found at line {line_num + 1}. Stopping early.")
            break

        line_sequence.append(best_pin)
        rows, cols = line_cache[(current_pin, best_pin)]
        error_image[rows, cols] -= line_weight
        np.clip(error_image, 0, None, out=error_image)

        # Draw line on the output image (scaled coordinates)
        p1_scaled = (pin_coords[current_pin][0] * SCALE, pin_coords[current_pin][1] * SCALE)
        p2_scaled = (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)
        # ***** CHANGE 2: Draw lines in white *****
        draw_output.line([p1_scaled, p2_scaled], fill=255, width=1) # Draw thin white lines

        last_pins.append(current_pin)
        current_pin = best_pin

        if (line_num + 1) % 200 == 0:
            print(f"  Generated line {line_num + 1}/{max_lines}")

    end_time = time.time()
    print(f"String art generation finished ({len(line_sequence)-1} lines drawn) in {end_time - start_time:.2f} seconds.")
    return line_sequence, output_image_pil


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate String Art Pin Sequence from an Image (Black Background, White Lines).") # Updated description
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("-o", "--output_seq", default="string_art_sequence.txt", help="Output file for the pin sequence (default: string_art_sequence.txt)")
    parser.add_argument("-p", "--output_png", default="string_art_preview_bw.png", help="Output file for the preview image (default: string_art_preview_bw.png)") # Changed default name
    parser.add_argument("--pins", type=int, default=DEFAULT_N_PINS, help=f"Number of pins around the circle (default: {DEFAULT_N_PINS})")
    parser.add_argument("--lines", type=int, default=DEFAULT_MAX_LINES, help=f"Maximum number of lines (strings) to generate (default: {DEFAULT_MAX_LINES})")
    parser.add_argument("--weight", type=float, default=DEFAULT_LINE_WEIGHT, help=f"Weight (error reduction) per line (default: {DEFAULT_LINE_WEIGHT})")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help=f"Internal processing image size (default: {IMG_SIZE})")
    parser.add_argument("--mindist", type=int, default=MIN_DISTANCE, help=f"Minimum pin distance for connections (default: {MIN_DISTANCE})")
    parser.add_argument("--minloop", type=int, default=MIN_LOOP, help=f"Number of recent pins to avoid reusing (default: {MIN_LOOP})")
    parser.add_argument("--previewpins", action='store_true', help="Draw markers for pins on the preview image.")

    args = parser.parse_args()

    # Basic parameter validation
    if args.pins < 3 or args.lines < 1 or args.weight <= 0 or args.size < 50 or args.mindist < 1 or args.minloop < 0:
        print("Error: Invalid parameter values. Please check constraints.")
        exit(1)
    if args.mindist >= args.pins // 2:
         print(f"Warning: Minimum distance ({args.mindist}) may be too large for the number of pins ({args.pins}).")


    # 1. Load and preprocess the image
    print(f"Loading and preprocessing '{args.input_image}'...")
    processed_image_array = load_and_preprocess_image(args.input_image, args.size)
    if processed_image_array is None: exit(1)

    # 2. Calculate Pin Coordinates
    pin_coordinates = calculate_pin_coords(args.pins, args.size)

    # 3. Pre-calculate Lines
    line_pixel_cache = precalculate_lines(pin_coordinates, args.pins, args.size, args.mindist)
    if not line_pixel_cache:
         print("Error: No valid lines could be pre-calculated.")
         exit(1)

    # 4. Generate String Art Sequence
    sequence, preview_image = generate_string_art(
        processed_image_array,
        args.pins,
        args.lines,
        args.mindist,
        args.minloop,
        args.weight,
        pin_coordinates,
        line_pixel_cache
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
                # ***** CHANGE 3: Draw pin markers in white *****
                draw_preview.ellipse(
                    (int(scaled_x - pin_marker_radius), int(scaled_y - pin_marker_radius),
                     int(scaled_x + pin_marker_radius), int(scaled_y + pin_marker_radius)),
                    fill=255 # White dots for pins
                )

        preview_image.save(args.output_png)
        print(f"String art preview saved to '{args.output_png}'")
    except IOError as e:
        print(f"Error saving preview image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the preview: {e}")

    print("Done.")