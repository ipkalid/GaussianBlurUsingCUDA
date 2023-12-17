#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <math.h>
#include <time.h>

#define M_PI 3.14159265358979323846264338327950288
#define SIGMA 20

#define SIGMA 20
#define DIM_BLOCK 32

__global__ void applyFilter(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth)
{
    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < height && col < width)
    {
        const int half = kernelWidth / 2;
        float blur = 0.0;

        for (int i = -half; i <= half; i++)
        {
            for (int j = -half; j <= half; j++)
            {
                const unsigned int y = max(0, min(height - 1, row + i));
                const unsigned int x = max(0, min(width - 1, col + j));

                const float w = kernel[(j + half) + (i + half) * kernelWidth];
                blur += w * input[x + y * width];
            }
        }
        output[col + row * width] = static_cast<unsigned char>(blur);
    }
}

void generateGaussianKernel(float *kernel, int kernelWidth)
{
    float sigma = SIGMA;
    int kernelHalfWidth = kernelWidth / 2;
    float sum = 0.0;

    for (int i = -kernelHalfWidth; i <= kernelHalfWidth; i++)
    {
        for (int j = -kernelHalfWidth; j <= kernelHalfWidth; j++)
        {
            float sqDist = i * i + j * j;
            float value = exp(-sqDist / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[(i + kernelHalfWidth) * kernelWidth + (j + kernelHalfWidth)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernelWidth * kernelWidth; i++)
    {
        kernel[i] /= sum;
    }
}

int main()
{
    // Load image using OpenCV
    cv::Mat img = cv::imread("fullhd.jpg", cv::IMREAD_COLOR);
    if (img.empty())
    {
        printf("Cannot load image file\n");
        return -1;
    }

    clock_t start = clock();

    // Generate Gaussian kernel
    int kernelWidth = 2 * (SIGMA * 3) + 1;
    float *kernel = (float *)malloc(kernelWidth * kernelWidth * sizeof(float));
    generateGaussianKernel(kernel, kernelWidth);

    // Split the image into its 3 channels
    cv::Mat channels[3];
    cv::split(img, channels);

    // Process each channel
    for (int c = 0; c < 3; c++)
    {
        // Convert to suitable format
        channels[c].convertTo(channels[c], CV_8UC1);

        // Allocate memory on device
        unsigned char *d_input_channel, *d_output_channel;
        float *d_kernel;
        cudaMalloc((void **)&d_input_channel, img.rows * img.cols);
        cudaMalloc((void **)&d_output_channel, img.rows * img.cols);
        cudaMalloc((void **)&d_kernel, kernelWidth * kernelWidth * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input_channel, channels[c].data, img.rows * img.cols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, kernelWidth * kernelWidth * sizeof(float), cudaMemcpyHostToDevice);

        // Set up dimensions for CUDA kernel
        dim3 dimBlock(DIM_BLOCK, DIM_BLOCK);
        dim3 dimGrid((img.cols + DIM_BLOCK - 1) / DIM_BLOCK, (img.rows + DIM_BLOCK - 1) / DIM_BLOCK);

        // Apply filter
        applyFilter<<<dimGrid, dimBlock>>>(d_input_channel, d_output_channel, img.cols, img.rows, d_kernel, kernelWidth);
        cudaDeviceSynchronize();

        // Copy result back to host
        cudaMemcpy(channels[c].data, d_output_channel, img.rows * img.cols, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input_channel);
        cudaFree(d_output_channel);
        cudaFree(d_kernel);
    }

    // Merge channels back into one image
    cv::Mat outputImg;
    cv::merge(channels, 3, outputImg);

    // Save output image
    cv::imwrite("output.jpg", outputImg);

    // Free host memory
    free(kernel);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time taken by program is : %f seconds\n", time_taken);
    return 0;
}