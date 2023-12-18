import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image

def convolution(image_path, kernel_size):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)

    # Define a simple convolution kernel (e.g., a blur kernel)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    # Perform convolution using the signal.convolve2d function from SciPy
    convolved_image = signal.convolve2d(image_array, kernel, mode='same', boundary='symm')

    # Display the original and convolved images
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(convolved_image, cmap='gray')
    plt.title('Convolved Image')

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform image convolution with a specified kernel.')
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    parser.add_argument('kernel_size', type=int, help='Size of the convolution kernel.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    convolution(args.image_path, args.kernel_size)
