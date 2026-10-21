# GEMM: from naive to AVX2

Single-precision matrix multiply (C = A x B, row-major) in C++17, optimized
step by step and benchmarked on one core of an [i5-12600K, pinned to a P-core].

## Results (GFLOP/s, single thread, n = 1040)

| Version | GFLOP/s | What changed |
|---|---|---|
| naive (ijk) | 4.3 | baseline |
| ikj | 8.9 | loop reorder: stride-1 access to B and C |
| reg4x4 | 13.1 | 4x4 tile held in registers |
| avx4x8 | 22.5 | AVX2/FMA, 8 lanes |
| avx4x16 | 39.1 | 8 independent accumulators hide FMA latency |
| avx4x16 + k-blocking | 52.6 | k chunks of 256 keep B strip in L1 |
| OpenBLAS (reference) | 136.8 | |

All versions validated against the naive result: max error, normalized by
the largest output value, about 2.4e-7.

## What I learned
- Loop order alone: [fill in].
- n = 1024 vs 1008/1040: naive drops about 4x at exactly 1024. [your
  explanation: power-of-two row stride maps B's rows to the same cache sets].
- Register blocking: [fill in what the loads-per-flop count was].
- avx4x8 vs avx4x16: [fill in; the FMA latency / accumulator count idea].
- Cache blocking (textbook version) did not help here, but k-blocking did.
  [fill in why you think so].

## Where it falls short
About 30 to 42% of single-threaded OpenBLAS. Not implemented: packing,
L2 blocking, prefetching, threading.

## Build
g++ -O2 -march=native -std=c++17 gemm.cpp -o gemm -lopenblas
OPENBLAS_NUM_THREADS=1 taskset -c 0 ./gemm
