// clang++ attention_kernel.cc -framework Accelerate -std=c++11 -o
// attention_kernel.o

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <vector>

static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock() {
  double the_time, norm_sec;
  struct timeval tv;

  gettimeofday(&tv, NULL);

  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = (double)tv.tv_sec;

  norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

  the_time = norm_sec + tv.tv_usec * 1.0e-6;

  return the_time;
}

typedef float real;
;

void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int m,
          int n, int k, const real alpha, const real *x, const real *y,
          const real beta, real *z) {
  int lda = (TransA == CblasNoTrans) ? k : m;
  int ldb = (TransB == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, x, lda, y, ldb,
              beta, z, n);
}

void gemv(const CBLAS_TRANSPOSE TransA, int m, int n, const real alpha,
          const real *a, const real *x, const real beta, real *y) {
  cblas_sgemv(CblasRowMajor, TransA, m, n, alpha, a, n, x, 1, beta, y, 1);
}

void axpy(int n, const real alpha, const real *x, real *y) {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

void copy(int n, const real *x, real *y) { cblas_scopy(n, x, 1, y, 1); }

void scale(int n, const real alpha, real *x) { cblas_sscal(n, alpha, x, 1); }
real dot(int n, const real *x, const real *y) {
  return cblas_sdot(n, x, 1, y, 1);
}

real norm(int n, const real *x) { return cblas_snrm2(n, x, 1); }

void softmax(int d, real *x) {
  real t;
  vDSP_maxv(x, 1, &t, d);
  t = -t;
  vDSP_vsadd(x, 1, &t, x, 1, d);
  vvexpf(x, x, &d);
  vDSP_sve(x, 1, &t, d);
  t = 1.0 / t;
  scale(d, t, x);
}

void vectorized_softmax(int m, int n, real *x) {
  real t;
  int mn = m * n;
  vDSP_maxv(x, 1, &t, mn);
  t = -t;
  vDSP_vsadd(x, 1, &t, x, 1, mn);
  vvexpf(x, x, &mn);
  for (int i = 0; i < m; i++) {
    real *xi = x + i * n;
    vDSP_sve(xi, 1, &t, n);
    t = 1.0 / t;
    scale(n, t, xi);
  }
}

/* q, k, v are row major ordered as nb * nh * ns * nd
  nb = batch size
  nh = num heads
  ns = sequence length
  nd = head dimension
*/
void attention_kernel_naive(size_t nb, size_t ns, size_t nh, size_t nd,
                            const real *wq, const real *wk, const real *wv,
                            real *cache, real *output) {

  real alpha = real(1) / sqrt(real(nd));
  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < nh; j++) {
      const real *vt = wv + i * (nh * ns * nd) + j * (ns * nd);
      for (int k = 0; k < ns; k++) {
        const real *qt = wq + i * (nh * ns * nd) + j * (ns * nd) + k * (nd);
        for (int l = 0; l < ns; l++) {
          const real *kt = wk + i * (nh * ns * nd) + j * (ns * nd) + l * (nd);
          cache[l] = dot(nd, qt, kt) * alpha;
        }
        softmax(ns, cache);
        real *ot = output + i * (nh * ns * nd) + j * (ns * nd) + k * (nd);
        gemv(CblasTrans, ns, nd, 1.0f, vt, cache, 0.f, ot);
      }
    }
  }
}

void attention_kernel_gemm_thread_block(size_t nb, size_t ns, size_t nh,
                                        size_t nd, size_t th_block_start,
                                        size_t th_block_end, const real *wq,
                                        const real *wk, const real *wv,
                                        real *cache, real *output) {

  real alpha = real(1) / sqrt(real(nd));
  for (int i = th_block_start; i < th_block_end; i++) {
    const real *qt = wq + i * (ns * nd);
    const real *kt = wk + i * (ns * nd);
    gemm(CblasNoTrans, CblasTrans, ns, ns, nd, alpha, qt, kt, 0.f, cache);
    vectorized_softmax(ns, ns, cache);
    const real *vt = wv + i * (ns * nd);
    real *ot = output + i * (ns * nd);
    gemm(CblasNoTrans, CblasNoTrans, ns, nd, ns, 1.f, cache, vt, 0.f, ot);
  }
}

real std_randn() {
  real u = ((real)rand() / (RAND_MAX)) * 2 - 1;
  real v = ((real)rand() / (RAND_MAX)) * 2 - 1;
  real r = u * u + v * v;
  if (r == 0 || r > 1)
    return std_randn();
  real c = sqrt(-2 * log(r) / r);
  return u * c;
}

void test() {

  size_t nb = 1;
  size_t ns = 2;
  size_t nh = 3;
  size_t nd = 4;

  size_t size = nb * nh * ns * nd;
  real q[24] = {-1.1787, 1.9830, 0.2984,  0.9705, -0.4595, 0.2125,
                0.8224,  0.9413, -0.0455, 0.4372, -0.1033, -0.6554,
                0.8065,  1.8598, 1.5230,  2.5783, 0.5982,  0.6150,
                -1.1190, 1.2630, -0.6106, 0.5770, -0.0874, -0.1796};
  real k[24] = {0.2716,  1.5794,  -2.2525, 0.2980,  -0.5299, 0.8881,
                0.5754,  -0.2009, 0.1338,  -0.6609, -1.5340, 0.8433,
                -0.6067, 0.3367,  -0.0270, -0.5836, -0.7974, -0.2649,
                0.2520,  -0.3843, -1.2837, 0.9515,  1.1459,  0.4369};
  real v[24] = {0.4433,  -0.0081, 0.7909,  1.6537, 0.5692,  1.1120,
                -0.8437, -0.2954, 0.5560,  0.5363, -0.7680, 0.2366,
                -0.7792, 0.1498,  -0.6616, 1.5427, 0.5015,  0.7759,
                1.7218,  0.6977,  -1.7757, 0.8460, 1.7959,  0.0765};

  real *cache = new real[ns * ns];
  real *output = new real[size];

  real expected_output[24] = {
      0.505204,  0.542646, -0.012822, 0.695340, 0.536290,  0.819209,
      -0.416420, 0.214089, -0.313592, 0.284579, -0.698704, 1.087240,
      -0.090629, 0.349120, -0.716471, 0.869136, -0.777030, 0.815258,
      1.763403,  0.348928, -0.854486, 0.817642, 1.765924,  0.327799};

  attention_kernel_gemm_thread_block(nb, ns, nh, nd, 0, nb * nh, q, k, v, cache,
                                     output);
  for (int i = 0; i < size; i++) {
    assert(abs(expected_output[i] - output[i]) < 1e-6);
  }
  delete[] cache;
  delete[] output;
}
int main() {

  test();
  size_t nb = 8;
  size_t ns = 512;
  size_t nh = 32;
  size_t nd = 128;

  size_t num_reps = 5;

  size_t size = nb * nh * ns * nd;

  size_t num_threads = 8;
  real *k = new real[size];
  real *q = new real[size];
  real *v = new real[size];
  real *cache = new real[num_threads * ns * ns];
  real *output0 = new real[size];
  real *output1 = new real[size];
  real *output2 = new real[size];
  for (int i = 0; i < size; i++) {
    k[i] = std_randn();
    q[i] = std_randn();
    v[i] = std_randn();
  }
  double elapsed = 0;

  for (int i = 0; i < num_reps; i++) {
    double start = dclock();
    attention_kernel_naive(nb, ns, nh, nd, q, k, v, cache, output0);
    elapsed += dclock() - start;
  }
  elapsed /= num_reps;
  printf("Time taken to compute attention of %zu batches of %zu sequences with "
         "%zu heads in %zu head dimension is %f seconds\n",
         nb, ns, nh, nd, elapsed);

  elapsed = 0;
  for (int i = 0; i < num_reps; i++) {
    double start = dclock();
    attention_kernel_gemm_thread_block(nb, ns, nh, nd, 0, nb * nh, q, k, v,
                                       cache, output1);
    elapsed += dclock() - start;
  }
  elapsed /= num_reps;
  printf("Time taken to compute attention with blas of %zu batches of %zu "
         "sequences with "
         "%zu heads in %zu head dimension is %f seconds\n",
         nb, ns, nh, nd, elapsed);

  for (int i = 0; i < size; i++) {
    assert(abs(output0[i] - output1[i]) < 1e-5);
  }
  size_t thread_block_size = nb * nh / num_threads;
  elapsed = 0;
  for (int i = 0; i < num_reps; i++) {
    double start = dclock();
    std::vector<std::thread> threads;
    size_t thread_block_start = 0;
    size_t thread_block_end = thread_block_size;
    for (int th = 0; th < num_threads; th++) {
      threads.push_back(std::thread(attention_kernel_gemm_thread_block, nb, ns,
                                    nh, nd, thread_block_start,
                                    thread_block_end, q, k, v,
                                    cache + th * ns * ns, output2));
      thread_block_start = thread_block_end;
      thread_block_end += thread_block_size;
    }
    for (auto &thread : threads) {
      thread.join();
    }
    elapsed += dclock() - start;
  }
  elapsed /= num_reps;
  printf("Time taken to compute threaded attention with blas of %zu batches of "
         "%zu "
         "sequences with "
         "%zu heads in %zu head dimension is %f seconds\n",
         nb, ns, nh, nd, elapsed);

  for (int i = 0; i < size; i++) {
    assert(abs(output1[i] - output2[i]) < 1e-5);
  }
  delete[] k;
  delete[] q;
  delete[] v;

  delete[] cache;
  delete[] output0;
  delete[] output1;
  delete[] output2;
}
