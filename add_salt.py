import os
import cv2
import numpy as np

def add_salt_and_pepper_noise(image, p=0.05):
    """
    Add salt and pepper noise to an image.

    Parameters:
    - image: Input image to add noise.
    - p: Probability that controls the proportion of pixels that will be altered.
    
    Returns:
    - Noisy image.
    """
    noisy_image = np.copy(image)

    # Generate a random matrix of the same shape as the image
    random_matrix = np.random.rand(*image.shape[:2])

    # Salt noise (pixels with random value less than p/2 are set to 255)
    noisy_image[random_matrix < (p / 2)] = 255

    # Pepper noise (pixels with random value greater than 1 - p/2 are set to 0)
    noisy_image[random_matrix > (1 - p / 2)] = 0

    return noisy_image

def process_images(input_folder, output_folder, p=0.05):
    """
    Read images from input folder, add salt and pepper noise, and save them to output folder.
    
    Parameters:
    - input_folder: Folder containing input images.
    - output_folder: Folder to save the noisy images.
    - p: Probability that controls the proportion of pixels that will be altered.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Adjust for image extensions
            # Read the image
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is not None:
                # Add salt and pepper noise to the image
                noisy_image = add_salt_and_pepper_noise(image, p)

                # Save the noisy image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, noisy_image)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to load image: {img_path}")

# Example usage
input_folder = '/workspace/Spline/code/Data/real/'
output_folder = '/workspace/Spline/code/Data/salt/0.005/'

# Set the probability p for salt and pepper noise
p = 0.005  # Adjust p to control the noise amount

process_images(input_folder, output_folder, p)



