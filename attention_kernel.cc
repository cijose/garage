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

void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int m,
          int n, int k, const float alpha, const float *x, const float *y,
          const float beta, float *z) {
  int lda = (TransA == CblasNoTrans) ? k : m;
  int ldb = (TransB == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, x, lda, y, ldb,
              beta, z, n);
}

void gemv(const CBLAS_TRANSPOSE TransA, int m, int n, const float alpha,
          const float *a, const float *x, const float beta, float *y) {
  cblas_sgemv(CblasRowMajor, TransA, m, n, alpha, a, n, x, 1, beta, y, 1);
}

void axpy(int n, const float alpha, const float *x, float *y) {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

void copy(int n, const float *x, float *y) { cblas_scopy(n, x, 1, y, 1); }

void scale(int n, const float alpha, float *x) { cblas_sscal(n, alpha, x, 1); }
float dot(int n, const float *x, const float *y) {
  return cblas_sdot(n, x, 1, y, 1);
}

inline float max(float a, float b) { return (a) > (b) ? a : b; }

float norm(int n, const float *x) { return cblas_snrm2(n, x, 1); }

void softmax(int d, float *x) {
  float t;
  vDSP_maxv(x, 1, &t, d);
  t = -t;
  vDSP_vsadd(x, 1, &t, x, 1, d);
  vvexpf(x, x, &d);
  vDSP_sve(x, 1, &t, d);
  t = 1.0 / t;
  scale(d, t, x);
}

void softmax(size_t n, size_t d, float *x) {
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    float max_val = x[i * d];
    for (int j = 1; j < d; j++) {
      max_val = max(max_val, x[i * d + j]);
    }
#pragma omp parallel for
    for (size_t j = 0; j < d; j++) {
      x[i * d + j] = exp(x[i * d + j] - max_val);
    }
    float normalizer = 0;
    for (size_t j = 0; j < d; j++) {
      normalizer += x[i * d + j];
    }
#pragma omp parallel for
    for (size_t j = 0; j < d; j++) {
      x[i * d + j] /= normalizer;
    }
  }
}

void vectorized_softmax(int m, int n, float *x) {
  float t;
  int mn = m * n;
  vDSP_maxv(x, 1, &t, mn);
  t = -t;
  vDSP_vsadd(x, 1, &t, x, 1, mn);
  vvexpf(x, x, &mn);
  for (int i = 0; i < m; i++) {
    float *xi = x + i * n;
    vDSP_sve(xi, 1, &t, n);
    t = 1.0 / t;
    scale(n, t, xi);
  }
}

void row_max_exp_row_sum(int m, int n, float *a, float *b, float *c) {
  for (int i = 0; i < m; i++) {
    vDSP_maxv(a + i * n, 1, &b[i], n);
    b[i] = -b[i];
    vDSP_vsadd(a + i * n, 1, &b[i], a + i * n, 1, n);
    b[i] = -b[i];
    vvexpf(a + i * n, a + i * n, &n);
    vDSP_sve(a + i * n, 1, &c[i], n);
  }
}

void black_magic(int br, int bc, int d, const float *vj, const float *pij,
                 const float *mij, const float *lij, float *li, float *mi,
                 float *pijvj, float *oi) {
  gemm(CblasNoTrans, CblasNoTrans, br, d, bc, 1.0f, pij, vj, 0.f, pijvj);
  for (int i = 0; i < br; i++) {
    float mi_new = max(mi[i], mij[i]);
    float a = li[i] * exp(mi[i] - mi_new);
    float b = exp(mij[i] - mi_new);
    float li_new = a + b * lij[i];
    li[i] = li_new;
    mi[i] = mi_new;
    a /= li_new;
    b /= li_new;
    vDSP_vsmsma(oi + i * d, 1, &a, pijvj + i * d, 1, &b, oi + i * d, 1, d);
  }
}

void flash_attention(int n, int d, int br, int bc, const float *q,
                     const float *k, const float *v, float *cache, float *o) {

  size_t flash_memory_size = 2 * n + br * bc + 2 * br + br * d;
  int tr = ceil((float)(n) / (float)(br));
  int tc = ceil((float)(n) / (float)(bc));
  memset(o, 0, sizeof(float) * n * d);
  memset(cache, 0, sizeof(float) * flash_memory_size);
  float *c = cache + n;
  for (int i = 0; i < n; i++) {
    c[i] = FLT_MIN;
  }
  float *sij = c + n;
  float *mij = sij + br * bc;
  float *lij = mij + br;
  float *pijvj = lij + br;
  float alpha = 1.f / sqrt((float)d);
  for (int j = 0; j < tc; j++) {
    const float *kj = k + j * (bc * d);
    const float *vj = v + j * (bc * d);
    for (int i = 0; i < tr; i++) {
      const float *qi = q + i * (br * d);
      gemm(CblasNoTrans, CblasTrans, br, bc, d, alpha, qi, kj, 0.f, sij);
      row_max_exp_row_sum(br, bc, sij, mij, lij);
      black_magic(br, bc, d, vj, sij, mij, lij, cache + i * br, c + i * br,
                  pijvj, o + i * (br * d));
    }
  }
}

/* q, k, v are row major ordered as nb * nh * ns * nd
  nb = batch size
  nh = num heads
  ns = sequence length
  nd = head dimension
*/
inline int min(int a, int b) { return a < b ? a : b; }
void attention_kernel_gemm_thread_block(size_t nb, size_t ns, size_t nh,
                                        size_t nd, size_t th_block_start,
                                        size_t th_block_end, const float *wq,
                                        const float *wk, const float *wv,
                                        float *cache, float *output) {

  // float alpha = float(1) / sqrt(float(nd));
  for (int i = th_block_start; i < th_block_end; i++) {
    const float *qt = wq + i * (ns * nd);
    const float *kt = wk + i * (ns * nd);
    const float *vt = wv + i * (ns * nd);
    float *ot = output + i * (ns * nd);

    flash_attention(ns, nd, min(ns, nd), min(ns, nd), qt, kt, vt, cache, ot);
    // gemm(CblasNoTrans, CblasTrans, ns, ns, nd, alpha, qt, kt, 0.f, cache);
    // vectorized_softmax(ns, ns, cache);
    // gemm(CblasNoTrans, CblasNoTrans, ns, nd, ns, 1.f, cache, vt, 0.f, ot);
  }
}

float std_randn() {
  float u = ((float)rand() / (RAND_MAX)) * 2 - 1;
  float v = ((float)rand() / (RAND_MAX)) * 2 - 1;
  float r = u * u + v * v;
  if (r == 0 || r > 1)
    return std_randn();
  float c = sqrt(-2 * log(r) / r);
  return u * c;
}

void test() {

  size_t nb = 1;
  size_t ns = 2;
  size_t nh = 3;
  size_t nd = 4;

  size_t size = nb * nh * ns * nd;
  float q[24] = {-1.1787, 1.9830, 0.2984,  0.9705, -0.4595, 0.2125,
                 0.8224,  0.9413, -0.0455, 0.4372, -0.1033, -0.6554,
                 0.8065,  1.8598, 1.5230,  2.5783, 0.5982,  0.6150,
                 -1.1190, 1.2630, -0.6106, 0.5770, -0.0874, -0.1796};
  float k[24] = {0.2716,  1.5794,  -2.2525, 0.2980,  -0.5299, 0.8881,
                 0.5754,  -0.2009, 0.1338,  -0.6609, -1.5340, 0.8433,
                 -0.6067, 0.3367,  -0.0270, -0.5836, -0.7974, -0.2649,
                 0.2520,  -0.3843, -1.2837, 0.9515,  1.1459,  0.4369};
  float v[24] = {0.4433,  -0.0081, 0.7909,  1.6537, 0.5692,  1.1120,
                 -0.8437, -0.2954, 0.5560,  0.5363, -0.7680, 0.2366,
                 -0.7792, 0.1498,  -0.6616, 1.5427, 0.5015,  0.7759,
                 1.7218,  0.6977,  -1.7757, 0.8460, 1.7959,  0.0765};

  float *cache = new float[nd * nd * ns * ns];
  float *output = new float[size];

  float expected_output[24] = {
      0.505204,  0.542646, -0.012822, 0.695340, 0.536290,  0.819209,
      -0.416420, 0.214089, -0.313592, 0.284579, -0.698704, 1.087240,
      -0.090629, 0.349120, -0.716471, 0.869136, -0.777030, 0.815258,
      1.763403,  0.348928, -0.854486, 0.817642, 1.765924,  0.327799};

  attention_kernel_gemm_thread_block(nb, ns, nh, nd, 0, nb * nh, q, k, v, cache,
                                     output);
  for (int i = 0; i < size; i++) {
    // printf("%d : %f : %f\n", i, expected_output[i], output[i]);
    assert(abs(expected_output[i] - output[i]) < 1e-3);
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
  float *k = new float[size];
  float *q = new float[size];
  float *v = new float[size];
  float *cache = new float[num_threads * ns * ns];
  float *output0 = new float[size];
  float *output1 = new float[size];
  for (int i = 0; i < size; i++) {
    k[i] = std_randn();
    q[i] = std_randn();
    v[i] = std_randn();
  }
  double elapsed = 0;

  for (int i = 0; i < num_reps; i++) {
    double start = dclock();
    attention_kernel_gemm_thread_block(nb, ns, nh, nd, 0, nb * nh, q, k, v,
                                       cache, output0);
    elapsed += dclock() - start;
  }
  elapsed /= num_reps;
  printf("Time taken to compute attention with blas of %zu batches of %zu "
         "sequences with "
         "%zu heads in %zu head dimension is %f seconds\n",
         nb, ns, nh, nd, elapsed);

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
                                    cache + th * ns * ns, output1));
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
    assert(abs(output0[i] - output1[i]) < 1e-5);
  }
  delete[] k;
  delete[] q;
  delete[] v;

  delete[] cache;
  delete[] output0;
  delete[] output1;
}
