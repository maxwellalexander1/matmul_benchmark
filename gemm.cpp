#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <cblas.h>

using Matrix = std::vector<float>;
using GemmFn = void (*)(const Matrix&, const Matrix&, Matrix&, int);
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
  double scale = 0.0;
  for (size_t i = 0; i < ref.size(); i++) {
    scale = std::max(scale, (double)std::fabs(ref[i]));
  }

  double worst = 0.0;
  for (size_t i = 0; i < x.size(); i++) {
    double err = std::fabs((double)x[i] - (double)ref[i]) / scale;
    worst = std::max(worst, err);
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

void gemm_reg4x4(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  for (int i = 0; i < n; i += 4) {
    for (int j = 0; j < n; j += 4) {
      float c[4][4] = {};

      for (int k = 0; k < n; k++) {
        for (int di = 0; di < 4; di++) {
          for (int dj = 0; dj < 4; dj++) {
            c[di][dj] += A[(i + di) * n + k] * B[k * n + (j + dj)];
          }
        }
      }

      for (int di = 0; di < 4; di++) {
        for (int dj = 0; dj < 4; dj++) {
          C[(i + di) * n + (j + dj)] = c[di][dj];
        }
      }
    }
  }
}

void gemm_avx4x16(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  for (int i = 0; i < n; i += 4) {
    for (int j = 0; j < n; j += 16) {
      __m256 c[4][2];
      for (int di = 0; di < 4; di++) {
        c[di][0] = _mm256_setzero_ps();
        c[di][1] = _mm256_setzero_ps();
      }

      for (int k = 0; k < n; k++) {
        __m256 b0 = _mm256_loadu_ps(&B[k * n + j]);       // columns j..j+7
        __m256 b1 = _mm256_loadu_ps(&B[k * n + j + 8]);   // columns j+8..j+15
        for (int di = 0; di < 4; di++) {
          __m256 a = _mm256_set1_ps(A[(i + di) * n + k]);
          c[di][0] = _mm256_fmadd_ps(a, b0, c[di][0]);
          c[di][1] = _mm256_fmadd_ps(a, b1, c[di][1]);
        }
      }

      for (int di = 0; di < 4; di++) {
        _mm256_storeu_ps(&C[(i + di) * n + j],     c[di][0]);
        _mm256_storeu_ps(&C[(i + di) * n + j + 8], c[di][1]);
      }
    }
  }
}

void gemm_avx4x8(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  for (int i = 0; i < n; i += 4) {
    for (int j = 0; j < n; j += 8) {
      __m256 c[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps(), _mm256_setzero_ps()};

      for (int k = 0; k < n; k++) {
        __m256 b = _mm256_loadu_ps(&B[k * n + j]);
            for (int di = 0; di < 4; di++) {
          __m256 a = _mm256_set1_ps(A[(i + di) * n + k]); 
          c[di] = a * b + c[di];
        }
      }

      for (int di = 0; di < 4; di++) {
        _mm256_storeu_ps(&C[(i + di) * n + j], c[di]);
      }
    }
  }
}

constexpr int KC = 256;

void gemm_avx4x16_kc(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  std::fill(C.begin(), C.end(), 0.0f);
  for (int kk = 0; kk < n; kk += KC) {
    int kend = std::min(kk + KC, n);
    for (int j = 0; j < n; j += 16) {
      for (int i = 0; i < n; i += 4) {
        __m256 c[4][2];
        for (int di = 0; di < 4; di++) {
          c[di][0] = _mm256_loadu_ps(&C[(i + di) * n + j]);
          c[di][1] = _mm256_loadu_ps(&C[(i + di) * n + j + 8]);
        }

        for (int k = kk; k < kend; k++) {
          __m256 b0 = _mm256_loadu_ps(&B[k * n + j]);
          __m256 b1 = _mm256_loadu_ps(&B[k * n + j + 8]);
          for (int di = 0; di < 4; di++) {
            __m256 a = _mm256_set1_ps(A[(i + di) * n + k]);
            c[di][0] = _mm256_fmadd_ps(a, b0, c[di][0]);
            c[di][1] = _mm256_fmadd_ps(a, b1, c[di][1]);
          }
        }

        for (int di = 0; di < 4; di++) {
          _mm256_storeu_ps(&C[(i + di) * n + j],     c[di][0]);
          _mm256_storeu_ps(&C[(i + di) * n + j + 8], c[di][1]);
        }
      }
    }
  }
}

void gemm_openblas(const Matrix& A, const Matrix& B, Matrix& C, int n) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
}

int main() {
  struct Impl { const char* name; GemmFn f; };
  Impl impls[] = {
    {"naive",   gemm_naive},
    {"ikj",     gemm_ikj},
    //{"blocked", gemm_blocked},
    {"reg4x4",  gemm_reg4x4},
    {"avx4x8",  gemm_avx4x8},
    {"avx4x16", gemm_avx4x16},
    {"av4x16_kc", gemm_avx4x16_kc},
    {"openblas", gemm_openblas},
  };

  const int n = 256;
  Matrix A = make_random(n, 1);
  Matrix B = make_random(n, 2);
  Matrix ref(n * n);
  gemm_naive(A, B, ref, n);

  for (auto& impl : impls) {
    Matrix C(n * n);
    impl.f(A, B, C, n);
    std::cout << impl.name << " err: " << max_rel_error(C, ref) << "\n";
  }

  for (int size : {256, 512, 1008, 1024, 1040}) {
    std::cout << "n=" << size << "\n";
    for (auto& impl : impls) {
      std::cout << "  " << impl.name << ": " << bench(impl.f, size) << " GFLOP/s\n";
    }
  }
}
