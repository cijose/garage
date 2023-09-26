#ifndef _LINALG_H_
#define _LINALG_H_

#include <Accelerate/Accelerate.h>

void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, int m,
          int n, int k, const real alpha, const real *x, const real *y,
          const real beta, real *z) {
  int lda = (TransA == CblasNoTrans) ? k : m;
  int ldb = (TransB == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, x, lda, y, ldb,
              beta, z, n);
}

#endif
