// clang++ -std=c++11 Convolutions.cc -o Convolutions
// -L/usr/local/opt/openblas/lib  -I/usr/local/opt/openblas/include -lblas

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
extern "C" {
#include <cblas.h>
}
using namespace std;

typedef float real;

void cblas_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                const int M, const int N, const int K, const float alpha,
                const float *A, const float *B, const float beta, float *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

void gaussian_random_vector(real mean, real stddev, real *xvec, size_t size) {
  std::minstd_rand rng(1);
  std::normal_distribution<real> ndist(0, 1);
  for (size_t i = 0; i < size; i++) {
    xvec[i] = stddev * ndist(rng) + mean;
  }
}

real relative_error(size_t n, const real *x, const real *y) {
  real rel_error = 0;
  for (size_t i = 0; i < n; i++) {
    real a = x[i];
    real b = y[i];
    rel_error += (fabs(a - b)) / fmax(fabs(a), fabs(b) + 1e-5);
  }
  rel_error /= n;
  return rel_error;
}

/*
 * sout = 1 + ((sin - kernel_size + padding) / stride)
 */
size_t convolution1d_naive(const real *input, const real *kernels,
                           size_t batch_size, size_t sequence_length,
                           size_t input_dimension, size_t output_dimension,
                           size_t kernel_size, size_t stride, size_t padding,
                           real *output) {

  size_t index = 0;
  for (size_t b = 0; b < batch_size; b++) {
    const real *input_b = input + b * sequence_length * input_dimension;
    for (size_t s = 0; s < sequence_length - kernel_size + padding + 1;
         s += stride) {
      const real *input_bs = input_b + s * input_dimension;
      for (size_t d = 0; d < output_dimension; d++) {
        const real *kernel_d = kernels + d * kernel_size * input_dimension;
        real value = 0;
        for (size_t k = 0; k < kernel_size * input_dimension; k++) {
          if (s * input_dimension + k >= sequence_length * input_dimension) {
            break;
          }
          value += input_bs[k] * kernel_d[k];
        }
        output[index] = value;
        index++;
      }
    }
  }
  return index;
}

size_t toeplitz_unroll(const real *input, size_t batch_size,
                       size_t sequence_length, size_t input_dimension,
                       size_t kernel_size, size_t stride, size_t padding,
                       real *toeplitz_matrix) {
  size_t index = 0;
  for (size_t b = 0; b < batch_size; b++) {
    const real *input_b = input + b * sequence_length * input_dimension;
    for (size_t s = 0; s < sequence_length - kernel_size + padding + 1;
         s += stride) {
      const real *input_bs = input_b + s * input_dimension;
      for (size_t k = 0; k < kernel_size * input_dimension; k++) {
        if (s * input_dimension + k >= sequence_length * input_dimension) {
          break;
        }
        toeplitz_matrix[index] = input_bs[k];
        index++;
      }
    }
  }
  return index;
}

size_t convolution1d_gemm(const real *input, const real *kernels,
                          size_t batch_size, size_t sequence_length,
                          size_t input_dimension, size_t output_dimension,
                          size_t kernel_size, size_t stride, size_t padding,
                          real *output) {

  size_t sequence_length_output = floor(
      1.0 + (real(sequence_length - kernel_size + padding) / real(stride)));
  real *toeplitz_matrix = new real[batch_size * sequence_length_output *
                                   kernel_size * input_dimension];
  size_t index =
      toeplitz_unroll(input, batch_size, sequence_length, input_dimension,
                      kernel_size, stride, padding, toeplitz_matrix);
  cblas_gemm(CblasNoTrans, CblasTrans, batch_size * sequence_length_output,
             output_dimension, kernel_size * input_dimension, (real)1.,
             toeplitz_matrix, kernels, (real)0., output);
  delete[] toeplitz_matrix;
  return index;
}

int main() {
  size_t batch_size = 1;
  size_t sequence_length_input = 1000;
  size_t input_dimension = 40;
  size_t kernel_size = 15;
  size_t output_dimension = 100;
  size_t stride = 1;
  size_t padding = 0;
  size_t sequence_length_output =
      floor(1.0 + (real(sequence_length_input - kernel_size + padding) /
                   real(stride)));
  size_t input_size = batch_size * sequence_length_input * input_dimension;
  size_t output_size = batch_size * sequence_length_output * output_dimension;
  size_t total_kernel_size = output_dimension * kernel_size * input_dimension;
  real *input = new real[input_size];
  real *output1 = new real[output_size];
  real *output2 = new real[output_size];
  real *kernels = new real[total_kernel_size];

  gaussian_random_vector(0.0, 1.0, input, input_size);
  gaussian_random_vector(0.0, 1.0, kernels, total_kernel_size);

  auto start = std::chrono::high_resolution_clock::now();
  size_t index = convolution1d_naive(
      input, kernels, batch_size, sequence_length_input, input_dimension,
      output_dimension, kernel_size, stride, padding, output1);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Naive convolution elapsed time: " << elapsed.count()
            << " seconds\n";

  start = std::chrono::high_resolution_clock::now();
  index = convolution1d_naive(input, kernels, batch_size, sequence_length_input,
                              input_dimension, output_dimension, kernel_size,
                              stride, padding, output2);
  finish = std::chrono::high_resolution_clock::now();

  elapsed = finish - start;
  std::cout << "GEMM convolution elapsed time: " << elapsed.count()
            << " seconds\n";

  cout << "Index: " << index << "  Output size " << output_size << endl;

  cout << "Relative error: " << relative_error(output_size, output1, output2)
       << endl;
  delete[] input;
  delete[] output1;
  delete[] output2;
  delete[] kernels;
}
