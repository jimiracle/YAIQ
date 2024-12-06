import os
from PIL import Image

def resize_images_in_directory(input_dir, output_dir, target_size=(224, 224)):
    '''
    Resize all images in the specified directory to the target size.

    Args:
        input_dir (str): Directory containing the images to resize.
        output_dir (str): Directory to save the resized images.
        target_size (tuple): The target size (width, height) to resize the images.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with Image.open(img_path) as img:
                img_resized = img.resize(target_size)
                img_resized.save(output_path)
                print(f"Resized and saved {filename} to {output_path}")

input_directory = '../../demo/0'
output_directory = '../../demo/0_resize'
resize_images_in_directory(input_directory, output_directory)
