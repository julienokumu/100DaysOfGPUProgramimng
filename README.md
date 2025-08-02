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
----------------------------------------------------------------------------------------------

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
---------------------------------------------------------------------------------------------------

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
 
  **Thought**:
  - Wondering how many lines of code the largest kernels is
---------------------------------------------------------------------------------------------------------------------------
