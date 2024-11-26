import os
import cv2
import numpy as np

def create_gaussian_noise(image_shape):
    """Create a noise matrix sampled from a normal Gaussian distribution (mean=0, variance=1)."""
    noise = np.random.normal(0, 1, image_shape)  # Mean=0, Variance=1
    # Scale noise to [0, 255]
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * 255
    return noise.astype(np.uint8)

def apply_noise_to_image(image, alpha=0.5):
    """Combine original image with noise using the formula: (1 - alpha) * original + alpha * noise."""
    # Create noise with the same shape as the input image
    noise = create_gaussian_noise(image.shape)
    
    # Blend the original image with noise
    noisy_image = (1 - alpha) * image + alpha * noise
    # Rescale the pixel values to be within [0, 255]
    min_val = np.min(noisy_image)
    max_val = np.max(noisy_image)
    rescaled_image = 255 * (noisy_image - min_val) / (max_val - min_val)
    return rescaled_image.astype(np.uint8)


def process_images(input_folder, output_folder, alpha=0.5):
    """Read images from input folder, apply noise, and save to output folder."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Add other formats if needed
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is not None:
                # Apply noise to the image
                noisy_image = apply_noise_to_image(image, alpha)
                
                # Save the noisy image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, noisy_image)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to load image: {img_path}")

# Example usage
input_folder = '/workspace/Spline/code/Data/real/'
output_folder = '/workspace/Spline/code/Data/noise/0.5/'

# Set alpha to control the amount of noise (e.g., alpha = 0.5 means equal weighting between noise and original image)
alpha = 0.5  # You can change this to increase or decrease the effect of the noise

process_images(input_folder, output_folder, alpha)
