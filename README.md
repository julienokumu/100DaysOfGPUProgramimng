**Day 0**: Hello GPU World Kernel

**Resource**: Read Chapter 1 of Programming Massively Parallel Processors

**What I learnt**:
- CUDA kernels are functions executed on the GPU by multiple threads in parallel
- The 'threadIdx.x' variable gives each thread a unique ID within the block
- 'cudaDeviceSynchronize()' ensures the GPU finishes before the program exits
- Learnt about different GPU's like NVIDIA Geforce RTX 4090 and AMD Radeon RX 7900 XTX
- Learnt about how kernels utilize GPU's
- Learnt about the GPU design and how it emphasizes on throughtput
- Learnt some error handling in CUDA
- Learnt about Amdahl's Law

**Challenges Faced**:
- Had to install the CUDA toolkit on my laptop to gain experience
- Due to no personal access to a GPU, I had to compile my code in Google Colab
- Using a nvccjupyter notebook didn't work so I had to manually compile the code
- Mistook 'cudaGetErrorString' for 'cudaGetLastErrorString'

**Performance Observations**:
- Kernel ran instantly, probably because it just prints messages

**Thoughts**:
- What happens when I launch more blocks?
-------------------------------------------------------------------------------------------------------------------------------------

**Day 1**: Vector Addition Kernel

**Resource**: Read chapter 2.1-2.3 of PMPP

**What I learnt**:
- Learnt about Data and Task Parallelism and how they differ
- Learnt about how C pointers work and how we use them in writing CUDA kernels

**Challenges Faced**:
- Understanding the maths in grid_stride loop were quite difficult, the core computation as well
- Took some time to gain intuition on C pointers

**Performance Observations**:
- Kernel ran instantly as well

**Thoughts**:
- I wonder how matrix multiplications work
-------------------------------------------------------------------------------------------------------------------------------------

**Day 2**: Vector Addition Kernel with Timing

**Resources**: Read Chapter 2.4 of PMPP, Read chapter 1-2 of the Best Cuda Practices Guide

**What I learnt**:
- Learnt about the APOD design cycle
- Learnt about CUDA API functions for managing device global memory
- Learnt that CUDA's malloc is a lot like C's
- Leant how to structure my args in cudaMemcpy
- Learnt how the pointer variable should be cast in cudaMalloc
- Learnt aboutr Gustafson's Law
- Solved a vector addition LeetGPU

**Challenges Faced**:
 - Struggled to understand the abstract math behind calculating the global idx and the grid-stride loop
 - Took some time solving a LeetGPU, had to figure out how to write a test program to follow the implementation details
 
**Performance Observations**:
- Compared sequential and parallel addition of vectors on CPU and GPU respectively and achieved a 29x speedup in the computation, which means the sequential method of the CPU must have been really slow
 
**Thoughts**:
- Wondering how many lines of code the largest kernels is
-------------------------------------------------------------------------------------------------------------------------------------

**Day 3**: Array Multiplication Kernel with Timing

**Resources**: Finished Chapter 2 of PMPP

**What I Learnt**:
- Learnt about kernel functions and threading
- Learnt about the built in variables like blockDim.x and threadIdx.x and how many dimensions they handle
- Learnt the difference between SPMD and SIMD
- Learnt about qualifier keywords

**Challenges Faced**:
- NO challenges faced today, getting the hang of the cuda syntax

**Performance Observations**:
- Tested GPU execution time and found that, the larger the number of threads the less the number of blocks and the faster the execution time and viceversa

**Thoughts**:
- How to write 2D kernels

-------------------------------------------------------------------------------------------------------------------------------------

**Day 4**: Array Multiplication Kernel, GPU vs CPU

**Resources**: Read Chapter 3 of PMPP

**What I Learnt**:
- Learnt about Multi-dimensional Grids
- Learnt how to write 2D and 3D arrays

**Challenges Faced**:
- Had a little trouble with some syntax errors but quickly fixed them

**Performance Observations**:
- A GPU achieves a max speedup of 2.82x
- The larger the block size, the faster the kernel and the more the speedup
- CPU execution time was faster on the 64 block size

**Thoughts**:
- Why do CPU's perform better with smaller threads?
-------------------------------------------------------------------------------------------------------------------------------------

**Day 5**: 2D Matrix Addition Kernel

**Resources**: Reread Chapter 3 of PMPP

**What I learnt**:
- Learnt how to handle 2D thread and block indexing
- Learnt how to implement strides
- Created a mental model to help me gain intuition on the dimesions of a grid and block

**Challenges Faced**:
- Took me a bit of time to understand the purpose of stride_x and stride_y

**Thoughts**:
- I wonder how CPU's perform of 2D computations
-------------------------------------------------------------------------------------------------------------------------------------

**Day 6**: 2D Matrix Addition Kernel GPU vs CPU

**Resources**: Chapter 3.4 of PMPP

**What I Learnt**:

- Read about matrix multiplication tiling

**Challenges Faced**:
- Debugging some syntax errors in the code gave me hell, but getting better at debugging cuda code

**Thoughts**:
- Same thought as yesterday, will find an answer
------------------------------------------------------------------------------------------------------------------------------------

**Day 7**: 2D Tiled Matrix Multiplication Kernel with Shared Memory

**Resources**: Chapter 4 of PMPP

**What I learnt**:
- Learnt about tiling and shared memory and how they are key optimization techniques

**Challenges Faced**:
- It was challenging implementing the tiling and understanding the matrix computations

**Thoughts**:
- I wonder how CPUs perform compared to GPU in matrix multiplications
------------------------------------------------------------------------------------------------------------------------------------

**Day 8**: Optimized a previous Array Multiplication Kernel I wrote
- Very busy today but i managed to write this kernel.

**Performance Observations**:
- Removed the grid-stride loop and moved the cudaMemcpy out of the timing loop; doing this significantly improved the gpu execution time, from 0.199ms on 155 blocks to 0.098ms on 157 blocks

---------------------------------------------------------------------------------------------------------------------------
