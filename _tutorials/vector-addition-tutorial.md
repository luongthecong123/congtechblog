---
layout: tutorial
title: "Building Your First CUDA Application: Vector Addition"
date: 2025-01-10
difficulty: Beginner
duration: "30 minutes"
order: 1
prerequisites:
  - "CUDA Toolkit installed"
  - "Basic C/C++ knowledge"
  - "NVIDIA GPU with CUDA support"
---

# Building Your First CUDA Application: Vector Addition

In this tutorial, we'll build a complete CUDA application that performs vector addition on the GPU. This is a classic first CUDA program that demonstrates the fundamental concepts.

## Learning Objectives

By the end of this tutorial, you'll understand:
- How to write a CUDA kernel
- Memory allocation and transfer between CPU and GPU
- Error handling in CUDA
- Performance comparison between CPU and GPU

## Step 1: Setting Up the Project

Create a new file called `vector_add.cu`:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000000  // Vector size
#define THREADS_PER_BLOCK 256

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
```

## Step 2: Writing the CUDA Kernel

The kernel function runs on the GPU:

```cpp
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out of bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Key Points:
- `__global__` indicates this function runs on the GPU
- `blockIdx.x` and `threadIdx.x` help calculate the unique thread ID
- Always check bounds to avoid memory access errors

## Step 3: CPU Version for Comparison

```cpp
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

## Step 4: Main Function

```cpp
int main() {
    // Host (CPU) arrays
    float *h_a, *h_b, *h_c, *h_c_cpu;
    
    // Device (GPU) arrays
    float *d_a, *d_b, *d_c;
    
    // Size in bytes
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    h_c_cpu = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify result with CPU version
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    
    // Check for errors
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_c[i] - h_c_cpu[i]) > 1e-5) {
            success = false;
            break;
        }
    }
    
    printf("Vector addition %s!\n", success ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_c_cpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

## Step 5: Compilation and Execution

Compile your program:

```bash
nvcc -o vector_add vector_add.cu
```

Run the program:

```bash
./vector_add
```

## Step 6: Adding Performance Measurement

Let's measure the performance difference:

```cpp
// Add timing for GPU version
cudaEvent_t start_gpu, stop_gpu;
cudaEventCreate(&start_gpu);
cudaEventCreate(&stop_gpu);

cudaEventRecord(start_gpu);
vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
cudaEventRecord(stop_gpu);
cudaEventSynchronize(stop_gpu);

float gpu_time;
cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

// Add timing for CPU version
clock_t start_cpu = clock();
vectorAddCPU(h_a, h_b, h_c_cpu, N);
clock_t end_cpu = clock();
float cpu_time = ((float)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

printf("GPU Time: %.2f ms\n", gpu_time);
printf("CPU Time: %.2f ms\n", cpu_time);
printf("Speedup: %.2fx\n", cpu_time / gpu_time);
```

## Understanding the Results

When you run this program, you should see:
- The vector addition passes verification
- GPU time vs CPU time comparison
- Speedup factor

## Common Issues and Solutions

### 1. Compilation Errors
- Ensure CUDA Toolkit is properly installed
- Check that your GPU supports CUDA

### 2. Runtime Errors
- Always use error checking macros
- Verify memory allocation succeeded

### 3. Performance Issues
- For small arrays, GPU might be slower due to overhead
- Try increasing vector size to see GPU benefits

## Exercise

Modify the program to:
1. Handle different data types (int, double)
2. Implement vector subtraction
3. Use multiple streams for overlapping computation

## Conclusion

Congratulations! You've successfully created your first CUDA application. You learned:
- Basic CUDA kernel structure
- Memory management between CPU and GPU
- Error handling best practices
- Performance measurement techniques

## Next Steps

In the next tutorial, we'll explore matrix multiplication and learn about shared memory optimization.

---

*Take time to experiment with different vector sizes and observe how performance changes.*
