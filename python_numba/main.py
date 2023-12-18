
import numpy as np
import math
import sys
import timeit
from PIL import Image
from numba import cuda
 
     
@cuda.jit
def apply_filter_numba(input, output, width, height, kernel, kernelWidth):
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if row < height and col < width:
        half = kernelWidth // 2
        blur = 0.0
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                y = max(0, min(height - 1, row + i))
                x = max(0, min(width - 1, col + j))
                w = kernel[i + half][j + half]
                blur += w * input[x][y]
                

        output[x][y] = np.uint8(blur)   

def gaussian_blur_python(input_image:str,output_image:str):
    img = Image.open(input_image)
    input_array = np.array(img).astype(np.uint8)
    red_channel = input_array[:, :, 0].copy().astype(np.uint8)
    green_channel = input_array[:, :, 1].copy().astype(np.uint8)
    blue_channel = input_array[:, :, 2].copy().astype(np.uint8)

    

    time_started = timeit.default_timer()

    sigma = 20 
    kernel_width = int(3 * sigma)
    if kernel_width % 2 == 0:
        kernel_width = kernel_width - 1  

    kernel_matrix = np.empty((kernel_width, kernel_width))
    kernel_half_width = kernel_width // 2
    for i in range(-kernel_half_width, kernel_half_width + 1):
        for j in range(-kernel_half_width, kernel_half_width + 1):
            kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                    np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                    / (2 * np.pi * sigma ** 2)
            )
    gaussian_kernel = kernel_matrix / kernel_matrix.sum()

    height, width = input_array.shape[:2]
    dim_block = (16, 16)
    dim_grid = (math.ceil(width / dim_block[0]), math.ceil(height / dim_block[1]))

    # Allocate memory on device
    d_red_channel = cuda.to_device(red_channel)
    d_green_channel = cuda.to_device(green_channel)
    d_blue_channel = cuda.to_device(blue_channel)
    d_gaussian_kernel = cuda.to_device(gaussian_kernel)
    out_red = cuda.to_device(np.zeros_like(red_channel).astype(np.uint8))
    out_green = cuda.to_device(np.zeros_like(green_channel).astype(np.uint8))
    out_blue = cuda.to_device(np.zeros_like(blue_channel).astype(np.uint8))
    # # Processing
    apply_filter_numba[dim_grid, dim_block](d_red_channel, out_red, width, height, d_gaussian_kernel, kernel_width)
    apply_filter_numba[dim_grid, dim_block](d_green_channel, out_green, width, height, d_gaussian_kernel, kernel_width)
    apply_filter_numba[dim_grid, dim_block](d_blue_channel, out_blue, width, height, d_gaussian_kernel, kernel_width)

   
    

     

    output_array = np.empty_like(input_array)
    output_array[:, :, 0] = out_red.copy_to_host()
    output_array[:, :, 1] = out_green.copy_to_host()
    output_array[:, :, 2] = out_blue.copy_to_host()
    out_img = Image.fromarray(output_array)
    out_img.save(output_image)
    time_ended = timeit.default_timer()
    print('Total processing time: ', time_ended - time_started, 's')




# gaussian_blur_python("images\\480.jpg","python_numba\export\\480.jpg")
# gaussian_blur_python("images\\1080.jpg","python_numba\export\\1080.jpg")
# gaussian_blur_python("images\\2k.jpg","python_numba\export\\2k.jpg")
gaussian_blur_python("images\\4k.jpg","python_numba\export\\4k.jpg")



