#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "config.h"

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define THREADS_PER_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)

#define SHARED_PADDING (WINDOW_SIZE / 2) + (GRAD_SIZE / 2) 

#define dceil(a, b) ((a) % (b) != 0 ? ((a) / (b) + 1) : ((a) / (b)))

// __global__
// void harrisCornerDetector_kernel(float *input, float *output, int height, int width) {
//     const int shared_size = THREADS_PER_BLOCK + 2 * THREADS_PER_BLOCK * SHARED_PADDING + SHARED_PADDING * SHARED_PADDING;
//     __shared__ float image_dx[shared_size];
//     __shared__ float image_dy[shared_size];

//     uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
//     uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
// }

__global__
void sobel_x_kernel(float *image, float *output, int height, int width, int pheight, int pwidth) {
    uint pixelY = blockIdx.y * blockDim.y + threadIdx.y + pwidth;
    uint pixelX = blockIdx.x * blockDim.x + threadIdx.x + pheight;

    if(pixelX - pwidth < width && pixelY - pheight < height) {
        float value = 0.0f; 
        // left horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = 1.0f ? i != 0 : 2.0f;
            value += weight * image[(pixelY + i) * (pwidth + width) + pixelX - 1];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = -1.0f ? i != 0 : -2.0f;
            value += weight * image[(pixelY + i) * (pwidth + width) + pixelX + 1];
        }
        output[pixelY * width + pixelX] = value;
        printf("Value: %f\n", value);
    }
}

__global__
void grad_y_kernel(float *image, float *output, int height, int width, int pheight, int pwidth) {
    
}

/* input is a grayscale image of size height by width */
void harrisCornerDetectorStaged(float *pinput, float *output, int height, int width, int pheight, int pwidth) {
    cudaSetDevice(0);
    //checkCUDAError("test", 1);
    // Create space for the image on the GPU
    float *device_input;
    float *device_output;

    const int pimage_size = sizeof(float) * (height + pheight) * (width + pwidth);
    const int image_size = sizeof(float) * height * width;
    cudaMalloc(&device_input, pimage_size);
    cudaMalloc(&device_output, image_size);

    // Copy input arrays to the GPU
    cudaMemcpy(device_input, pinput, pimage_size, cudaMemcpyHostToDevice);

    const dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
    const dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
    printf("HERE\n");
    sobel_x_kernel<<<grid, threadBlock>>>(device_input, device_output, height, width, 1, 1);
    cudaDeviceSynchronize();
    printf("DONE\n");
    //harrisCornerDetector_kernel(device_input, device_output, height, width);

    // Copy result to CPU
    cudaMemcpy(output, device_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    printf("Finshed kernel\n");
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}