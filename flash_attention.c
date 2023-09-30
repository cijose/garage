
#include <Accelerate/Accelerate.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

// On Apple
// clang flash_attention.cc -framework Accelerate  -o flash_attention
//  -fPIC  -O3

static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */

inline float max(float a, float b) { return (a) > (b) ? a : b; }

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

void gemm(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
          int m, int n, int k, const float alpha, const float *x,
          const float *y, const float beta, float *z) {
  int lda = (TransA == CblasNoTrans) ? k : m;
  int ldb = (TransB == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, x, lda, y, ldb,
              beta, z, n);
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
  for (int j = 0; j < tc; j++) {
    const float *kj = k + j * (bc * d);
    const float *vj = v + j * (bc * d);
    for (int i = 0; i < tr; i++) {
      const float *qi = q + i * (br * d);
      gemm(CblasNoTrans, CblasTrans, br, bc, d, 1.0f, qi, kj, 0.f, sij);
      row_max_exp_row_sum(br, bc, sij, mij, lij);
      black_magic(br, bc, d, vj, sij, mij, lij, cache + i * br, c + i * br,
                  pijvj, o + i * (br * d));
    }
  }
}

void naive_attention(int n, int d, const float *q, const float *k,
                     const float *v, float *cache, float *o) {

  memset(o, 0, sizeof(float) * n * d);
  gemm(CblasNoTrans, CblasTrans, n, n, d, 1.0f, q, k, 0.f, cache);
  softmax(n, n, cache);
  gemm(CblasNoTrans, CblasNoTrans, n, d, n, 1.0f, cache, v, 0.f, o);
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

  size_t n = 1024;
  size_t d = 64;
  size_t m = 8 * d;

  size_t size = n * d;
  size_t num_exp = 50;
  float *k = (float *)malloc(sizeof(float) * size);
  float *q = (float *)malloc(sizeof(float) * size);
  float *v = (float *)malloc(sizeof(float) * size);
  for (int i = 0; i < size; i++) {
    k[i] = std_randn();
    q[i] = std_randn();
    v[i] = std_randn();
  }

  float *o_naive = (float *)malloc(sizeof(float) * n * d);
  float *o_flash = (float *)malloc(sizeof(float) * n * d);
  float *sqkt = (float *)malloc(sizeof(float) * n * n);
  double elapsed_time_naive = 0;
  for (int i = 0; i < num_exp; i++) {
    double start_time = dclock();
    naive_attention(n, d, q, k, v, sqkt, o_naive);
    double end_time = dclock();
    elapsed_time_naive += end_time - start_time;
  }
  printf("Time taken to compute naive attention of %zu sequences with %zu "
         "dimension is %f seconds\n",
         n, d, elapsed_time_naive / (float)num_exp);

  int br = d;
  int bc = d;
  size_t flash_memory_size = 2 * n + br * bc + 2 * br + br * d;
  float *cache = (float *)malloc(sizeof(float) * (flash_memory_size));
  float memory_saved = (float)(n * n) / (float)(flash_memory_size);
  printf("memory size = %zu bytes, naive memeory_size = %zu bytes, "
         "memory_saved = %fx, br = "
         "%d bc = %d\n",
         flash_memory_size * sizeof(float), n * n * sizeof(float), memory_saved,
         br, bc);
  double elapsed_time_flash = 0;
  for (int i = 0; i < num_exp; i++) {
    double start_time = dclock();
    flash_attention(n, d, br, bc, q, k, v, cache, o_flash);
    double end_time = dclock();
    elapsed_time_flash += end_time - start_time;
  }
  for (int i = 0; i < size; i++) {
    //    printf("%d : %f : %f\n", i, o_naive[i], o_flash[i]);
    assert(fabsf(o_naive[i] - o_flash[i]) < 1e-3);
  }
  printf("Time taken to compute flash attention of %zu sequences with %zu "
         "dimension is %f seconds\n",
         n, d, elapsed_time_flash / (float)num_exp);

  printf("Flash attention is %f faster than naive attenion and uses %f less "
         "memory\n",
         elapsed_time_naive / elapsed_time_flash, memory_saved);
  free(o_naive);
  free(o_flash);
  free(q);
  free(k);
  free(v);
  free(sqkt);
  free(cache);
}
int main() {
  test();

  return 0;
}
