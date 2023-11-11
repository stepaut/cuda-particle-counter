﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <iostream>


// Count of points in 2D area
const unsigned COUNT = 1000;
// Begin of axises
const int MIN = 0;
// End of axises
const int MAX = 1000;
// Size of checking block
const unsigned SIZE = 100;
// Bins per axis
const int BINS = (MAX - MIN) / SIZE;


void generateArray(unsigned seed, int* a) {
	for (int i = 0; i < COUNT; i++) {
		srand(seed * (i+1) + i * i);
		a[i] = rand() % (MAX - MIN) + MIN;
	}
}

__global__ void wherePoint(int* x, int* y, unsigned *res) {
	// --- The number of threads does not cover all the data size
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (i < COUNT) {
		int binX = -1;
		int binY = -1;

		char c = 0;

		for (int k = 0; k < BINS; k++) {
			if (MIN + k * SIZE <= x[i] && x[i] <= MIN + (k + 1) * SIZE) {
				binX = k;
				c++;
			}

			if (MIN + k * SIZE <= y[i] && y[i] <= MIN + (k + 1) * SIZE) {
				binY = k;
				c++;
			}

			if (c == 2) {
				break;
			}
		}

		atomicAdd(&res[binX * BINS + binY], 1);
		i += offset;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t runCuda(int* x, int* y, unsigned* res)
{
	int* dev_x = 0;
	int* dev_y = 0;
	unsigned* dev_res = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_res, BINS * BINS * sizeof(unsigned));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, COUNT * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_y, COUNT * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_x, x, COUNT * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_y, y, COUNT * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	wherePoint << <1, COUNT >> > (dev_x, dev_y, dev_res);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(res, dev_res, BINS * BINS * sizeof(unsigned), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_res);
	cudaFree(dev_x);
	cudaFree(dev_y);

	return cudaStatus;
}

int main() {
	int x[COUNT];
	generateArray(4213, x);
	int y[COUNT];
	generateArray(7028, y);

	int dx[COUNT];
	generateArray(9038, dx);
	int dy[COUNT];
	generateArray(1001, dy);

	unsigned res[BINS * BINS] = {};

	clock_t start, stop;

	start = clock();
	cudaError_t cudaStatus = runCuda(x, y, res);
	stop = clock();
	float elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
	printf("Time to generate (CPU): %3.1f ms\n", elapsedTime);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "run failed!");
		return 1;
	}

	for (int i = 0; i < BINS; i++) {
		for (int j = 0; j < BINS; j++) {
			std::cout << res[i * BINS + j];
			std::cout << "\t";
		}
		std::cout << "\n";
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}