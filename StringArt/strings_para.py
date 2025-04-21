# strings.py (Parallel Version)

import sys
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import imageio.v2 as imageio
from skimage.transform import resize as imresize
from skimage.color import rgb2gray
from PIL import Image

import math
from collections import defaultdict
import os # Added for cpu_count
from joblib import Parallel, delayed # Added for parallel processing

# Assuming bresenham.py is in the same directory or accessible
from bresenham import bresenham, circle


def image(filename, size):
    img = imresize(rgb2gray(imageio.imread(filename)), (size, size))
    return img

# Helper function to process a single edge for parallel execution
def _process_edge(edge_index, edge_code, radius, hooks):
    """Calculates pixel codes for a single edge.

    Args:
        edge_index (int): The unique index (column index) for this edge.
        edge_code (tuple): The (hook_idx1, hook_idx2) tuple.
        radius (int): The radius of the main circle.
        hooks (np.array): Array of hook coordinates.

    Returns:
        tuple: (list_of_pixel_codes, edge_index)
               Returns (None, edge_index) if start and end points are the same after int conversion.
    """
    i, j = edge_code
    ni = hooks[i]
    nj = hooks[j]

    # Avoid calculating path for identical integer coordinates (can happen with rounding)
    start_pt = [int(ni[0]), int(ni[1])]
    end_pt = [int(nj[0]), int(nj[1])]
    if start_pt == end_pt:
        return None, edge_index # Indicate no path

    pixels = bresenham(start_pt, end_pt).path
    pixel_codes = []
    size_1d = radius * 2 + 1
    for pixel in pixels:
        # Ensure pixel is within bounds [-radius, radius] before calculating code
        px, py = pixel[0], pixel[1]
        if -radius <= px <= radius and -radius <= py <= radius:
             # Map (px, py) from [-radius, radius] coordinate system to 1D index
             # Center (0,0) maps to (radius, radius) in a 0-indexed grid
             pixel_code = (py + radius) * size_1d + (px + radius)
             pixel_codes.append(pixel_code)
    return pixel_codes, edge_index


def build_arc_adjecency_matrix(n, radius):
    print("building sparse adjecency matrix (in parallel)")
    hooks = np.array([[math.cos(np.pi * 2 * i / n), math.sin(np.pi * 2 * i / n)] for i in range(n)])
    # Scale and center hooks (assuming center is 0,0 for bresenham initially)
    # Bresenham expects integer coords, but keep high precision for hook locations
    # The int conversion will happen inside _process_edge
    hooks_scaled = radius * hooks # No immediate int conversion here

    # 1. Generate all unique edge pairs (hook indices)
    edge_codes = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_codes.append((i, j))
    num_edges = len(edge_codes)
    print(f"Total potential edges: {num_edges}")

    # 2. Process edges in parallel
    n_cores = os.cpu_count()
    print(f"Using {n_cores} cores for processing edges...")
    # Use backend="loky" for better robustness than default "multiprocessing"
    results = Parallel(n_jobs=n_cores, backend="loky", verbose=10)(
        delayed(_process_edge)(edge_idx, edge_codes[edge_idx], radius, hooks_scaled)
        for edge_idx in range(num_edges)
    )

    print("Parallel processing finished. Aggregating results...")
    # 3. Aggregate results
    row_ind = []
    col_ind = []
    valid_edge_indices = [] # Keep track of edges that actually produced pixels

    current_valid_col_idx = 0
    final_edge_codes = [] # Store only the edge codes that resulted in a path

    for pixel_codes, original_edge_idx in results:
        if pixel_codes is not None and len(pixel_codes) > 0: # Check if the edge produced a path
            row_ind.extend(pixel_codes)
            # Use the *new* column index corresponding to valid edges
            col_ind.extend([current_valid_col_idx] * len(pixel_codes))
            valid_edge_indices.append(original_edge_idx)
            final_edge_codes.append(edge_codes[original_edge_idx])
            current_valid_col_idx += 1

    num_valid_edges = current_valid_col_idx
    print(f"Number of valid edges with paths: {num_valid_edges}")

    # 4. Create the sparse matrix
    # Use float32 for potentially better performance/memory with lsqr
    data = np.ones(len(row_ind), dtype=np.float32)
    shape = ((2 * radius + 1) * (2 * radius + 1), num_valid_edges)
    print(f"Sparse matrix shape: {shape}")
    print(f"Number of non-zero entries (pixels covered): {len(row_ind)}")

    sparse_matrix = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape, dtype=np.float32)

    # Return the final hooks, the matrix, and *only* the edge codes corresponding to columns in the matrix
    return sparse_matrix, hooks_scaled, final_edge_codes # Return scaled hooks


# --- The rest of the functions remain largely the same ---
# (build_circle_adjecency_matrix is unused, leave as is or remove)
# (build_image_vector, reconstruct, reconstruct_and_save, dump_arcs)


def build_circle_adjecency_matrix(radius, small_radius):
    # (This function is not used in the main path and not parallelized here)
    print("building sparse adjecency matrix for circles (not parallelized)")
    edge_codes = []
    row_ind = []
    col_ind = []
    pixels = circle(small_radius)
    grid_range = range(-radius+small_radius+1, radius-small_radius, 1) # Adjusted range slightly
    size_1d = radius * 2 + 1
    edge_idx_counter = 0
    for i, cx in enumerate(grid_range):
        for j, cy in enumerate(grid_range):
            edge_codes.append((i, j)) # Store original grid refs if needed
            edge_pixel_codes = []
            for pixel in pixels:
                px, py = cx + pixel[0], cy + pixel[1]
                if -radius <= px <= radius and -radius <= py <= radius:
                    pixel_code = (py + radius) * size_1d + (px + radius)
                    edge_pixel_codes.append(pixel_code)

            if edge_pixel_codes: # Only add if circle contains valid pixels
                row_ind.extend(edge_pixel_codes)
                col_ind.extend([edge_idx_counter] * len(edge_pixel_codes))
                edge_idx_counter += 1

    if not row_ind: # Handle case where no circles fit
        print("Warning: No valid circle positions found for the given radii.")
        shape = ((2*radius+1)*(2*radius+1), 0)
        return scipy.sparse.csr_matrix(shape, dtype=np.float32), [], []

    data = np.ones(len(row_ind), dtype=np.float32)
    shape = ((2 * radius + 1) * (2 * radius + 1), edge_idx_counter)
    sparse = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape, dtype=np.float32)
    hooks = [] # Hooks not relevant for this mode in the same way
    # Return edge codes corresponding to actual columns
    return sparse, hooks, edge_codes[:edge_idx_counter]


def build_image_vector(img, radius):
    # representing the input image as a sparse column vector of pixels:
    assert img.shape[0] == img.shape[1]
    img_size = img.shape[0]
    row_ind = []
    data = []
    size_1d = radius * 2 + 1
    img_center_offset = img_size // 2

    for y_img, line in enumerate(img):
        for x_img, pixel_value in enumerate(line):
            # Map image coords (0..img_size-1) to global coords centered at 0
            global_x = x_img - img_center_offset
            global_y = y_img - img_center_offset

            # Check if the pixel falls within the main circle's bounds
            if -radius <= global_x <= radius and -radius <= global_y <= radius:
                # Map global coords [-radius, radius] to 1D index
                pixel_code = (global_y + radius) * size_1d + (global_x + radius)
                # Use float32 for consistency
                data.append(float(pixel_value))
                row_ind.append(pixel_code)

    # Ensure data is float32
    data = np.array(data, dtype=np.float32)
    col_ind = np.zeros(len(row_ind), dtype=int) # Column index is always 0
    shape = ((2 * radius + 1) * (2 * radius + 1), 1)
    sparse_b = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape, dtype=np.float32)
    return sparse_b


def reconstruct(x, sparse, radius):
    # Ensure x is float32 for consistency if needed, though dot product handles it
    # x = np.asarray(x, dtype=np.float32)
    b_approx = sparse.dot(x)
    img_shape = (2 * radius + 1, 2 * radius + 1)
    # Ensure reshaping uses the correct dimensions
    if b_approx.shape[0] != img_shape[0] * img_shape[1]:
         raise ValueError(f"Shape mismatch: b_approx shape {b_approx.shape} cannot be reshaped to {img_shape}")

    # Reshape directly if it's already a 1D array/vector
    b_image = b_approx.reshape(img_shape)

    # Clipping should ideally happen *before* scaling for saving
    # b_image = np.clip(b_image, 0, ?? ) # Clip based on expected range?
    return b_image # Return float image


def reconstruct_and_save(x, sparse, radius, filename):
    # Apply brightness correction *before* clipping and saving
    # Note: Correction factor might need adjustment depending on image content
    brightness_correction = 1.2 # Start with 1.2, adjust if needed

    # Ensure x is a NumPy array for multiplication
    x_corr = np.asarray(x) * brightness_correction

    # Reconstruct the image (returns floats)
    b_image = reconstruct(x_corr, sparse, radius)

    # Determine the appropriate range for clipping/scaling
    # Option 1: Clip to [0, 1] if input image was 0-1 normalized
    # Option 2: Use percentiles to avoid extreme outliers affecting scaling
    # Option 3: Fixed range if negative values were allowed and clipped later

    # Let's assume the target range after reconstruction (ideally) is [0, 1]
    # like the input grayscale image, before brightness correction.
    # We clip based on a reasonable expected range after correction.
    # Clipping to [0, 1] might be too restrictive if brightness > 1 pushes values higher.
    # Let's clip negatives to 0, and let positives scale.
    b_image = np.clip(b_image, 0, None) # Clip only negatives

    # Normalize to 0-255 for saving as uint8 PNG
    # Avoid division by zero if image is all black
    max_val = np.max(b_image)
    if max_val > 1e-6: # Use a small threshold
         img_normalized = b_image / max_val
    else:
         img_normalized = b_image # Already zero or near-zero

    img_array_uint8 = (255 * img_normalized).astype(np.uint8)

    # Convert array to image and save as PNG
    image = Image.fromarray(img_array_uint8, mode='L') # Specify 'L' for grayscale
    image.save(filename)
    print(f"Saved reconstructed image to {filename}")


def dump_arcs(solution, hooks, edge_codes, filename):
    try:
        with open(filename, "w") as f:
            n = len(hooks)
            print(n, file=f)
            # hooks are now potentially float, format appropriately
            for i, (x, y) in enumerate(hooks):
                print(f"{i}\t{x:.6f}\t{y:.6f}", file=f) # Use fixed precision format
            print(file=f)

            if len(edge_codes) != len(solution):
                 # This check is crucial after filtering edges in parallel build
                 raise ValueError(f"Mismatch: {len(edge_codes)} edge codes != {len(solution)} solution values.")

            for (i, j), value in zip(edge_codes, solution):
                if abs(value) < 1e-9: # Use tolerance for floating point comparison
                    continue
                # Check if value is very close to an integer
                if abs(value - round(value)) < 1e-9:
                    print(f"{i}\t{j}\t{int(round(value))}", file=f)
                else:
                    print(f"{i}\t{j}\t{value:.6f}", file=f) # Print floats with precision
        print(f"Saved arc data to {filename}")
    except IOError as e:
        print(f"Error writing to {filename}: {e}")
    except ValueError as e:
        print(f"Error preparing data for dump_arcs: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python strings.py <input_image_filename> <output_prefix>")
        sys.exit(1)
    filename, output_prefix = sys.argv[1:]

    n = 288  # Number of hooks
    radius = 1000 # Radius in pixels (adjust as needed)

    # Choose mode: 'arc' or 'circle'
    mode = 'arc'

    if mode == 'arc':
        sparse, hooks, edge_codes = build_arc_adjecency_matrix(n, radius)
    elif mode == 'circle':
        # Note: build_circle_adjecency_matrix is not parallelized
        sparse, hooks, edge_codes = build_circle_adjecency_matrix(radius, 10) # Example small radius
    else:
        raise ValueError("Unknown mode selected")

    # Exit if matrix building failed (e.g., no valid edges/circles)
    if sparse.shape[1] == 0:
         print("Error: Adjacency matrix has no columns (no valid edges/elements found). Exiting.")
         sys.exit(1)


    # square image with same center as the circle.
    shrinkage = 0.75
    # Ensure image size is calculated correctly based on diameter (2*radius)
    img_size = int(radius * 2 * shrinkage)
    print(f"Loading image {filename} and resizing to {img_size}x{img_size}")
    img = image(filename, img_size)
    sparse_b = build_image_vector(img, radius)
    # Optional: Save original image mapped to the circle space
    # try:
    #     orig_img_name = output_prefix + "-original-mapped.png"
    #     img_array = sparse_b.toarray().reshape((2 * radius + 1, 2 * radius + 1))
    #     img_array_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    #     Image.fromarray(img_array_uint8, mode='L').save(orig_img_name)
    #     print(f"Saved mapped original image to {orig_img_name}")
    # except Exception as e:
    #     print(f"Could not save mapped original image: {e}")


    # finding the solution, a weighting of edges:
    print("Solving linear system (lsqr)...")
    # Use float32 for potentially better performance/memory if matrices are float32
    b_vector = np.array(sparse_b.todense(), dtype=np.float32).flatten()

    # lsqr arguments: A, b, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None
    # Consider increasing iter_lim if convergence is slow for large problems
    result = scipy.sparse.linalg.lsqr(sparse, b_vector, show=True) # Show progress
    print(f"LSQR finished: stop condition={result[1]}, iterations={result[2]}")
    x = result[0].astype(np.float32) # Ensure result is float32

    # --- Reconstruction and Saving ---
    # Allow negative (pure mathematical result)
    reconstruct_and_save(x, sparse, radius, output_prefix + "-allow-negative.png")

    # Clip negative values (physically realistic)
    x_non_negative = np.clip(x, 0, None) # Clip only negatives, keep upper unbound for now
    reconstruct_and_save(x_non_negative, sparse, radius, output_prefix + "-unquantized.png")
    if mode == 'arc': # Only dump arcs if in arc mode
         dump_arcs(x_non_negative, hooks, edge_codes, output_prefix + "-unquantized.txt")

    # --- Quantization (Optional) ---
    quantization_level = 50 # Example: 50 levels. None means no quantization.
    clip_factor = 0.3 # Clip weights above 30% of the max quantized level

    x_final = x_non_negative.copy() # Start with non-negative weights

    if quantization_level is not None and quantization_level > 0:
        print(f"Quantizing weights to {quantization_level} levels...")
        max_edge_weight_orig = np.max(x_final)

        if max_edge_weight_orig > 1e-9: # Avoid division by zero/NaN
            x_quantized = np.round(x_final / max_edge_weight_orig * quantization_level)

            # Clip values larger than clip_factor times maximum *quantized* value.
            max_quantized_val = np.max(x_quantized)
            clip_limit = int(max_quantized_val * clip_factor)
            print(f"Clipping quantized weights above {clip_limit} ({clip_factor*100:.1f}% of max {max_quantized_val:.1f})")
            x_quantized = np.clip(x_quantized, 0, clip_limit)

            # Scale it back for reconstruction (optional, could reconstruct with 0-level weights too)
            x_final = x_quantized / quantization_level * max_edge_weight_orig

            if mode == 'arc': # Only dump arcs if in arc mode
                 # Dump the integer quantized weights
                 dump_arcs(x_quantized.astype(int), hooks, edge_codes, output_prefix + ".txt")
        else:
            print("Max weight is near zero, skipping quantization.")
            x_quantized = np.zeros_like(x_final) # All weights are effectively zero
            if mode == 'arc':
                 dump_arcs(x_quantized.astype(int), hooks, edge_codes, output_prefix + ".txt")

        reconstruct_and_save(x_final, sparse, radius, output_prefix + ".png")


        # --- Statistics (only if quantized and in arc mode) ---
        if mode == 'arc':
            print("Calculating statistics for quantized arcs...")
            arc_count = 0
            total_distance = 0.0
            hist = defaultdict(int)
            num_used_arcs = 0

            # Ensure x_quantized exists and is integer type for stats
            x_quantized_int = x_quantized.astype(int)

            for multiplicity in x_quantized_int:
                hist[multiplicity] += 1
                arc_count += multiplicity # Total number of string wraps

            # Calculate distance only for arcs with multiplicity > 0
            used_indices = np.where(x_quantized_int > 0)[0]
            num_used_arcs = len(used_indices)

            for idx in used_indices:
                multiplicity = x_quantized_int[idx]
                if idx < len(edge_codes): # Safety check
                    hook_index1, hook_index2 = edge_codes[idx]
                    # Ensure hooks are NumPy arrays for vector subtraction
                    hook1 = np.asarray(hooks[hook_index1], dtype=float)
                    hook2 = np.asarray(hooks[hook_index2], dtype=float)
                    # Distance calculation based on scaled hook coordinates
                    distance = np.linalg.norm(hook1 - hook2)
                    # Normalize distance relative to circle diameter (2*radius)
                    normalized_distance = distance / (2 * radius)
                    total_distance += normalized_distance * multiplicity
                else:
                    print(f"Warning: Edge index {idx} out of bounds for edge_codes.")


            print("Quantized Weight Histogram (Weight: Count):")
            for multiplicity in sorted(hist.keys()):
                if multiplicity == 0: continue # Skip zero weight count usually
                print(f"  {multiplicity}: {hist[multiplicity]}")
            print(f"Total arc segments (sum of multiplicities): {arc_count}")
            print(f"Number of unique hook pairs used (multiplicity > 0): {num_used_arcs}")
            # Total distance assumes diameter=1, so divide by diameter (2*radius) for normalization
            print(f"Total normalized distance (relative to diameter=1): {total_distance:.2f}")

    else: # If no quantization, the final image is the unquantized one
        print("Skipping quantization.")
        # x_final is already x_non_negative
        reconstruct_and_save(x_final, sparse, radius, output_prefix + ".png")
        if mode == 'arc':
             # Save the non-negative float weights as the final .txt
             dump_arcs(x_final, hooks, edge_codes, output_prefix + ".txt")


if __name__ == "__main__":
    main()