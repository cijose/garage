#include <assert.h>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

static double gtod_ref_time_sec = 0.0;
typedef float real;

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

inline int min(int a, int b) { return (a) < (b) ? a : b; }
inline int max(int a, int b) { return (a) > (b) ? a : b; }
void matmul(int m, int n, int d, const real *a, const real *b, real *c) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      real o = 0;
      for (int k = 0; k < d; k++) {
        o += a[i * d + k] * b[j * d + k];
      }
      c[i * n + j] = o;
    }
  }
}

void row_max(int m, int n, const real *a, real *b) {
  for (int i = 0; i < m; i++) {
    b[i] = -std::numeric_limits<real>::infinity();
    for (int j = 0; j < n; j++) {
      b[i] = max(b[i], a[i * n + j]);
    }
  }
}

void row_sum(int m, int n, const real *a, real *b) {
  for (int i = 0; i < m; i++) {
    b[i] = 0;
    for (int j = 0; j < n; j++) {
      b[i] += a[i * n + j];
    }
  }
}

void stable_exp(int m, int n, const real *rmax, const real *a, real *b) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      b[i * n + j] = exp(a[i * n + j] - rmax[i]);
    }
  }
}

void black_magic(int br, int bc, int d, const real *vj, const real *pij,
                 const real *mij, const real *lij, real *li, real *mi,
                 real *oi) {

  for (int i = 0; i < br; i++) {

    real mi_new = max(mi[i], mij[i]);
    real li_new = exp(mi[i] - mi_new) * li[i] + exp(mij[i] - mi_new) * lij[i];
    for (int j = 0; j < d; j++) {
      real pijvj = 0;
      for (int p = 0; p < bc; p++) {
        pijvj += pij[i * bc + p] * vj[p * d + j];
      }
      oi[i * d + j] = (li[i] * exp(mi[i] - mi_new) * oi[i * d + j] +
                       exp(mij[i] - mi_new) * pijvj) /
                      li_new;
    }
    li[i] = li_new;
    mi[i] = mi_new;
  }
}
void flash_attention(int n, int d, int m, const real *q, const real *k,
                     const real *v, real *o) {

  int bc = ceil(real(m) / real(4 * d));
  int br = min(ceil(real(m) / real(4 * d)), d);

  memset(o, 0, sizeof(real) * n * d);

  real *l = (real *)malloc(sizeof(real) * (2 * n + br * bc + 2 * br));
  real *c = l + n;
  memset(l, 0, sizeof(real) * n);
  for (int i = 0; i < n; i++) {
    c[i] = -std::numeric_limits<real>::infinity();
  }
  int tr = ceil(real(n) / real(br));
  int tc = ceil(real(n) / real(bc));

  real *sij = c + n;
  real *pij = sij;
  real *mij = sij + br * bc;
  real *lij = mij + br;
  for (int j = 0; j < tc; j++) {
    const real *kj = k + j * (bc * d);
    const real *vj = v + j * (bc * d);
    for (int i = 0; i < tr; i++) {
      const real *qi = q + i * (br * d);
      real *oi = o + i * (br * d);
      real *li = l + i * br;
      real *ci = c + i * br;
      matmul(br, bc, d, qi, kj, sij);
      row_max(br, bc, sij, mij);
      stable_exp(br, bc, mij, sij, pij);
      row_sum(br, bc, pij, lij);
      black_magic(br, bc, d, vj, pij, mij, lij, li, ci, oi);
    }
  }
  free(l);
}

void softmax(int n, int d, const real *x, real *y) {
  for (int i = 0; i < n; i++) {
    real max_val = -std::numeric_limits<real>::infinity();
    for (int j = 0; j < d; j++) {
      max_val = max(max_val, x[i * d + j]);
    }
    real normalizer = 0;
    for (int j = 0; j < d; j++) {
      real expx = exp(x[i * d + j] - max_val);
      normalizer += expx;
      y[i * d + j] = expx;
    }
    for (int j = 0; j < d; j++) {
      y[i * d + j] /= normalizer;
    }
  }
}

void naive_attention(int n, int d, const real *q, const real *k, const real *v,
                     real *o) {

  memset(o, 0, sizeof(real) * n * d);
  real *sqkt = (real *)malloc(sizeof(real) * n * n);
  matmul(n, n, d, q, k, sqkt);
  softmax(n, n, sqkt, sqkt);
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < d; j++) {
        o[i * d + j] += sqkt[i * n + k] * v[k * d + j];
      }
    }
  }
  free(sqkt);
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

  size_t n = 512;
  size_t d = 128;
  size_t m = 8 * d;

  size_t size = n * d;
  real *k = new real[size];
  real *q = new real[size];
  real *v = new real[size];
  for (int i = 0; i < size; i++) {
    k[i] = std_randn();
    q[i] = std_randn();
    v[i] = std_randn();
  }

  real *o_naive = (real *)malloc(sizeof(real) * n * d);
  real *o_flash = (real *)malloc(sizeof(real) * n * d);
  double start_time = dclock();
  naive_attention(n, d, q, k, v, o_naive);
  double end_time = dclock();
  printf("Time taken to compute naive attention of %zu sequences with %zu "
         "dimension is %f seconds\n",
         n, d, end_time - start_time);

  start_time = dclock();
  flash_attention(n, d, m, q, k, v, o_flash);
  end_time = dclock();
  printf("Time taken to compute flash attention of %zu sequences with %zu "
         "dimension is %f seconds\n",
         n, d, end_time - start_time);

  for (int i = 0; i < size; i++) {
    // printf("%d : %f : %f\n", i, o_naive[i], o_flash[i]);
    assert(abs(o_naive[i] - o_flash[i]) < 1e-5);
  }
  free(o_naive);
  free(o_flash);
  free(q);
  free(k);
  free(v);
}
int main() {
  test();

  return 0;
}
