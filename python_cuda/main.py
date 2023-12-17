import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import numpy as np
import math
import sys
import timeit
from PIL import Image



def apply_filter_cpu(input_image, kernel):
    height, width = input_image.shape
    kernel_height, kernel_width = kernel.shape

    # Padding the image to handle borders
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(input_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Output image
    output_image = np.zeros_like(input_image)

    # Applying the filter
    for row in range(height):
        for col in range(width):
            for i in range(-pad_height, pad_height + 1):
                for j in range(-pad_width, pad_width + 1):
                    y = min(max(row + i, 0), height - 1)
                    x = min(max(col + j, 0), width - 1)
                    output_image[row, col] += kernel[i + pad_height, j + pad_width] * padded_image[y + pad_height, x + pad_width]

    return output_image

def gaussian_blur_python(input_image:str,output_image:str):
    
    try:
        img = Image.open(input_image)
        input_array = np.array(img)
        red_channel = input_array[:, :, 0].copy()
        green_channel = input_array[:, :, 1].copy()
        blue_channel = input_array[:, :, 2].copy()
    except FileNotFoundError:
        sys.exit("Cannot load image file")


    time_started = timeit.default_timer()

    sigma = 20 
    kernel_width = int(3 * sigma)
    if kernel_width % 2 == 0:
        kernel_width = kernel_width - 1  

    kernel_matrix = np.empty((kernel_width, kernel_width), np.float32)
    kernel_half_width = kernel_width // 2
    for i in range(-kernel_half_width, kernel_half_width + 1):
        for j in range(-kernel_half_width, kernel_half_width + 1):
            kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                    np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                    / (2 * np.pi * sigma ** 2)
            )
    gaussian_kernel = kernel_matrix / kernel_matrix.sum()


  
    height, width = input_array.shape[:2]
    dim_block = 32
    dim_grid_x = math.ceil(width / dim_block)
    dim_grid_y = math.ceil(height / dim_block)

  
    mod = compiler.SourceModule(open('puthon_cuda\gaussian_blur.cu').read())
    apply_filter = mod.get_function('applyFilter')

     

    for channel in (red_channel, green_channel, blue_channel):
        apply_filter(
            drv.In(channel),
            drv.Out(channel),
            np.uint32(width),
            np.uint32(height),
            drv.In(gaussian_kernel),
            np.uint32(kernel_width),
            block=(dim_block, dim_block, 1),
            grid=(dim_grid_x, dim_grid_y)
        )
        # apply_filter_cpu(channel,gaussian_kernel)


    output_array = np.empty_like(input_array)
    output_array[:, :, 0] = red_channel
    output_array[:, :, 1] = green_channel
    output_array[:, :, 2] = blue_channel



    out_img = Image.fromarray(output_array)

    out_img.save(output_image)



    time_ended = timeit.default_timer()
    print('Total processing time: ', time_ended - time_started, 's')



gaussian_blur_python("images\\480.jpg","python_cuda\export\\480.jpg")
gaussian_blur_python("images\\1080.jpg","python_cuda\export\\1080.jpg")
gaussian_blur_python("images\\2k.jpg","python_cuda\export\\2k.jpg")
gaussian_blur_python("images\\4k.jpg","python_cuda\export\\4k.jpg")


