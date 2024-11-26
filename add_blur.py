import os
import cv2

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur to an image.
    
    Parameters:
    - image: The input image.
    - kernel_size: Size of the Gaussian kernel. It should be an odd number (e.g., (5, 5)).
    - sigma: Standard deviation for the Gaussian kernel. If 0, it is automatically calculated based on the kernel size.
    
    Returns:
    - Blurred image.
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def process_images(input_folder, output_folder, kernel_size=(5, 5), sigma=0):
    """
    Read images from input folder, apply Gaussian blur, and save to output folder.
    
    Parameters:
    - input_folder: Folder containing the input images.
    - output_folder: Folder where blurred images will be saved.
    - kernel_size: Size of the Gaussian kernel.
    - sigma: Standard deviation for the Gaussian kernel.
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
                # Apply Gaussian blur
                blurred_image = apply_gaussian_blur(image, kernel_size, sigma)

                # Save the blurred image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, blurred_image)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to load image: {img_path}")

# Example usage
input_folder = '/workspace/Spline/code/Data/real/'
output_folder = '/workspace/Spline/code/Data/blur/13/'

kernel = 13
# Set Gaussian blur parameters
kernel_size = (kernel, kernel)  # Size of the kernel (must be odd numbers)
sigma = 0  # Standard deviation (0 means it's computed from kernel size)

process_images(input_folder, output_folder, kernel_size, sigma)
