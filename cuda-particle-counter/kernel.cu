#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <iostream>


// Count of points in 2D area
const unsigned COUNT = 1000000;
// Begin of axises
const int MIN = 0;
// End of axises
const int MAX = 100;
// Size of checking block
const unsigned SIZE = 20;
// Bins per axis
const int BINS = (MAX - MIN) / SIZE;
// Iterations of movement
const unsigned ITERS = 50;
// Print arrays
const bool DOOUTPUT = false;


void generateCoordinate(unsigned seed, int* a) {
	for (int i = 0; i < COUNT; i++) {
		srand(seed * (i + 1) + i * i);
		a[i] = rand() % (MAX - MIN) + MIN;
	}
}

void generateSpeed(unsigned seed, int* a) {
	for (int i = 0; i < COUNT; i++) {
		srand(seed * (i + 1) + i * i);
		a[i] = rand() % 20 - 10;
	}
}

void applyMovement(int* a, int* da) {
	for (int i = 0; i < COUNT; i++) {
		a[i] += da[i];

		if (a[i] > MAX) {
			a[i] = MAX;
		}

		if (a[i] < MIN) {
			a[i] = MIN;
		}
	}
}

void print2DArray(unsigned* a, unsigned shape) {
	for (int i = 0; i < shape; i++) {
		for (int j = 0; j < shape; j++) {
			std::cout << a[i * shape + j];
			std::cout << "\t";
		}
		std::cout << "\n";
	}
}

void print1DArray(int* a, unsigned shape) {
	for (int j = 0; j < shape; j++) {
		std::cout << a[j];
		std::cout << "\t";
	}
	std::cout << "\n";
}

void setArrayToZero(unsigned* a, unsigned shape) {
	for (int i = 0; i < shape; i++) {
		a[i] = 0;
	}
}

void wherePointCPU(int* x, int* y, unsigned* res) {
	for (int i = 0; i < COUNT; i++) {
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

		res[binX * BINS + binY] += 1;
	}
}

__global__ void wherePoint(int* x, int* y, unsigned* res) {
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

	int threadsPerBlock = 256;
	int blocks = COUNT / threadsPerBlock;

	// Launch a kernel on the GPU with one thread for each element.
	wherePoint << <blocks, threadsPerBlock >> > (dev_x, dev_y, dev_res);

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
	int *x = new int[COUNT];
	generateCoordinate(4213, x);
	//print1DArray(x, COUNT);

	int *y = new int[COUNT];
	generateCoordinate(7028, y);
	//print1DArray(y, COUNT);

	int *dx = new int[COUNT];
	generateSpeed(9038, dx);
	//print1DArray(dx, COUNT);

	int *dy = new int[COUNT];
	generateSpeed(1001, dy);
	//print1DArray(dy, COUNT);

	unsigned *res = new unsigned[BINS * BINS];

	clock_t start, stop;
	cudaError_t cudaStatus;
	float elapsedTime;
	cudaEvent_t startC, stopC;
	cudaEventCreate(&startC);
	cudaEventCreate(&stopC);

	float totalTimeCUDA = 0;
	float totalTimeCPU = 0;

	for (int k = 0; k < ITERS; k++) {
		applyMovement(x, dx);
		applyMovement(y, dy);

		memset(res, 0, BINS * BINS);

		cudaEventRecord(startC, 0);
		cudaStatus = runCuda(x, y, res);
		cudaEventRecord(stopC, 0);
		cudaEventSynchronize(stopC);
		cudaEventElapsedTime(&elapsedTime, startC, stopC);

		totalTimeCUDA += elapsedTime;
		printf("Time: %3.1f ms\n", elapsedTime);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "run failed!");
			return 1;
		}

		if (DOOUTPUT)
			print2DArray(res, BINS);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	std::cout << "------------------------------------------------------------------- Without CUDA:\n";

	// same without CUDA:
	generateCoordinate(4213, x);
	generateCoordinate(7028, y);

	for (int k = 0; k < ITERS; k++) {
		applyMovement(x, dx);
		applyMovement(y, dy);

		setArrayToZero(res, BINS * BINS);

		start = clock();
		wherePointCPU(x, y, res);
		stop = clock();
		float elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
		totalTimeCPU += elapsedTime;
		printf("Time: %3.1f ms\n", elapsedTime);

		if (DOOUTPUT)
			print2DArray(res, BINS);
	}

	printf("CUDA Time: %3.1f ms\n", totalTimeCUDA);
	printf("CPU Time: %3.1f ms\n", totalTimeCPU);

	return 0;
}