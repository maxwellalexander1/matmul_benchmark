#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

using Matrix = std::vector<float>;

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

int main() {
  const int n = 256;
  Matrix A = make_random(n, 1);
  Matrix B = make_random(n, 2);
  Matrix C(n * n);

  gemm_naive(A, B, C, n);
  Matrix C_ref = C;
  std::fill(C.begin(), C.end(), 0.0f);
  gemm_naive(A, B, C, n);
  std::cout << max_rel_error(C, C_ref);
  std::cout << C[0] << " " << C[1] << " " << C[n] << "\n";
}
