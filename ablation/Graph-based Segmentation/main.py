from skimage import io
import matplotlib.pyplot as plt
from filter import *
from segment_graph import *
import time
import numpy as np
import os


# --------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           in_image: image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#           num_ccs: number of connected components in the segmentation.
# --------------------------------------------------------------------------------
def segment(in_image, sigma, k, min_size, log_file):
    start_time = time.time()
    height, width, band = in_image.shape
    log_message = f"Height: {height}, Width: {width}\n"
    log_file.write(log_message)

    smooth_red_band = smooth(in_image[:, :, 0], sigma)
    smooth_green_band = smooth(in_image[:, :, 1], sigma)
    smooth_blue_band = smooth(in_image[:, :, 2], sigma)

    # build graph
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y)
                num += 1
            if y < height - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x, y + 1)
                num += 1

            if (x < width - 1) and (y < height - 1):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y + 1)
                num += 1

            if (x < width - 1) and (y > 0):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y - 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y - 1)
                num += 1

    # Segment the image
    u = segment_graph(width * height, num, edges, k)

    # Post-process small components
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    output = np.zeros(shape=(height, width, 3))

    # Pick random colors for each component
    colors = np.zeros(shape=(height * width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            output[y, x, :] = colors[comp, :]

    elapsed_time = time.time() - start_time
    log_message = f"Execution time: {int(elapsed_time / 60)} minute(s) and {int(elapsed_time % 60)} seconds\n"
    log_file.write(log_message)

    # Display the results
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image)
    a.set_title('Original Image')
    a = fig.add_subplot(1, 2, 2)
    output = output.astype(int)
    plt.imshow(output)
    a.set_title('Segmented Image')
    plt.show()

    # Save the segmented image
    output_path = "segmented_image.jpg"
    io.imsave(output_path, output.astype(np.uint8))
    log_message = f"Segmented image saved as: {output_path}\n"
    log_file.write(log_message)


if __name__ == "__main__":
    sigma = 0.5
    k = 1000
    min_size = 50
    input_path = "input_image.jpg"
    log_file_path = "segmentation_log.txt"

    # Open log file
    with open(log_file_path, 'w') as log_file:
        # Log the parameters
        log_file.write(f"Input image path: {input_path}\n")
        log_file.write(f"sigma: {sigma}, k: {k}, min_size: {min_size}\n")

        # Loading the image
        input_image = io.imread(input_path)
        log_file.write("Loading is done.\n")
        log_file.write("Processing...\n")

        # Run segmentation
        segment(input_image, sigma, k, min_size, log_file)
