#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>

using Matrix = std::vector<float>;
constexpr int BS = 64;

Matrix make_random(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Matrix m(n * n);
  for (auto& x : m) x = dist(rng);
  return m;
}

void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

void gemm_ikj(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  std::fill(C.begin(), C.end(), 0.0f);
  for (int i = 0; i < n; i++){
    for (int k = 0; k < n; k++){
      float a = A[i * n + k];
      float sum = 0.0f;
      for (int j = 0; j < n; j++) {
        C[i * n + j] += a * B[k * n + j];
   }
  }
 }
}

double max_rel_error(const Matrix& x, const Matrix& ref) {
  double worst = 0.0;
  for (size_t i = 0; i < x.size(); i++) {
    double err = std::fabs(x[i] - ref[i]) / std::max(std::fabs(ref[i]), 1e-6f);
    if (err > worst){
      worst = err; 
    }
  }
  return worst;
}

double bench(void (*f)(const Matrix&, const Matrix&, Matrix&, int), int n) {
  Matrix A = make_random(n, 1);
  Matrix B = make_random(n, 2);
  Matrix C(n * n);

  f(A,B,C,n);

  double best = 1e30;
  for (int rep = 0; rep < 5; rep++) {
    auto t0 = std::chrono::steady_clock::now();
    f(A,B,C,n);
    auto t1 = std::chrono::steady_clock::now();

   double secs = std::chrono::duration<double>(t1 - t0).count();
    
    if (secs < best){
       best = secs;
    }
    
  }

  volatile float sink = C[0];
  (void)sink;

  return 2.0 * n * n * n / best / 1e9;

}

void gemm_blocked(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  std::fill(C.begin(), C.end(), 0.0f);
  for (int ii = 0; ii < n; ii += BS) {
  for (int kk = 0; kk < n; kk += BS) {
    for (int jj = 0; jj < n; jj += BS) {
      for (int i = ii; i < ii + BS; i++) {
        for (int k = kk; k < kk + BS; k++) {
          float a = A[i * n + k];
          for (int j = jj; j < jj + BS; j++) {
            C[i * n + j] += a * B[k * n + j];
          }
        }
      }
    }
  }
 }
}



int main() {
  const int n = 256;
  Matrix A = make_random(n, 1);
  Matrix B = make_random(n, 2);
  Matrix C_naive(n * n);
  Matrix C_ikj(n * n);

  gemm_naive(A, B, C_naive, n);
  gemm_ikj(A, B, C_ikj, n);
  std::cout << "max rel error: " << max_rel_error(C_ikj, C_naive) << "\n";

  for (int size : {256, 512, 1024}) {
    std::cout << "n=" << size
              << "  naive: " << bench(gemm_naive, size) << " GFLOP/s"
              << "  ikj: "   << bench(gemm_ikj, size)   << " GFLOP/s\n";

  }
}
