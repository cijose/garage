
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

void row_max(int m, int n, const float *a, float *b) {
  for (int i = 0; i < m; i++) {
    b[i] = a[i * n];
    for (int j = 1; j < n; j++) {
      b[i] = max(b[i], a[i * n + j]);
    }
  }
}

void row_sum(int m, int n, const float *a, float *b) {
  for (int i = 0; i < m; i++) {
    b[i] = a[i * n];
    for (int j = 1; j < n; j++) {
      b[i] += a[i * n + j];
    }
  }
}

void stable_exp(int m, int n, const float *rmax, float *a) {
  for (int i = 0; i < m * n; i++) {
    a[i] = exp(a[i] - rmax[i / n]);
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
    for (int j = 0; j < d; j++) {
      oi[i * d + j] = a * oi[i * d + j] + b * pijvj[i * d + j];
    }
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
      float *oi = o + i * (br * d);
      float *li = cache + i * br;
      float *ci = c + i * br;
      gemm(CblasNoTrans, CblasTrans, br, bc, d, 1.0f, qi, kj, 0.f, sij);
      row_max(br, bc, sij, mij);
      stable_exp(br, bc, mij, sij);
      row_sum(br, bc, sij, lij);
      black_magic(br, bc, d, vj, sij, mij, lij, li, ci, pijvj, oi);
    }
  }
}

void naive_attention(int n, int d, const float *q, const float *k,
                     const float *v, float *o) {

  memset(o, 0, sizeof(float) * n * d);
  float *sqkt = (float *)malloc(sizeof(float) * n * n);
  gemm(CblasNoTrans, CblasTrans, n, n, d, 1.0f, q, k, 0.f, sqkt);
  softmax(n, n, sqkt);
  gemm(CblasNoTrans, CblasNoTrans, n, d, n, 1.0f, sqkt, v, 0.f, o);
  free(sqkt);
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
  double start_time = dclock();
  naive_attention(n, d, q, k, v, o_naive);
  double end_time = dclock();
  double elapsed_time_naive = end_time - start_time;
  printf("Time taken to compute naive attention of %zu sequences with %zu "
         "dimension is %f seconds\n",
         n, d, elapsed_time_naive);

  int br = d;
  int bc = 2 * d;
  size_t flash_memory_size = 2 * n + br * bc + 2 * br + br * d;
  float *cache = (float *)malloc(sizeof(float) * (flash_memory_size));
  float memory_saved = (float)(n * n) / (float)(flash_memory_size);
  printf(
      "memory size = %zu, naive memeory_size = %zu, memory_saved = %fx, br = "
      "%d bc = %d\n",
      flash_memory_size, n * n, memory_saved, br, bc);
  start_time = dclock();
  flash_attention(n, d, br, bc, q, k, v, cache, o_flash);
  end_time = dclock();
  double elapsed_time_flash = end_time - start_time;
  for (int i = 0; i < size; i++) {
    //    printf("%d : %f : %f\n", i, o_naive[i], o_flash[i]);
    assert(fabsf(o_naive[i] - o_flash[i]) < 1e-3);
  }
  printf("Time taken to compute flash attention of %zu sequences with %zu "
         "dimension is %f seconds\n",
         n, d, elapsed_time_flash);

  printf("Flash attention is %f slower than naive attenion but uses %f less "
         "memory\n",
         elapsed_time_flash / elapsed_time_naive, memory_saved);
  free(o_naive);
  free(o_flash);
  free(q);
  free(k);
  free(v);
  free(cache);
}
int main() {
  test();

  return 0;
}
