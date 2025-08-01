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
