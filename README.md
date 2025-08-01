**Day 0**: Hello GPU World

**Resource**: Read Chapter 1 of Programming Massively Parallel Processors

**What I learned**:
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

