/*
 *  Cervnet is a c++ library which implements Kronecker recurrent units (KRU) and several other recurrent neural networks.  
 *
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Cijo Jose <cijo.jose@alumni.epfl.ch>
 *
 *  This file is part of Cervnet.
 *
 *  Cervnet is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  Cervnet is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
// nvcc KroneckerForwardBackward.cu -ccbin=g++-4.9  -std=c++11 -lcublas -o KroneckerForwardBackward.bin

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstring>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>
#include <cublas_v2.h>
const uint32_t CUDA_NUM_THREADS = 1024;
inline uint32_t CUDA_GET_BLOCKS(const uint32_t N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
#define CUDA_CHECK(condition) assert(condition == cudaSuccess)


void cpu_kronecker_forward_kernel(int M, int N, int rowk,
                                       int colk, int stride,
                                       const float* W_k, const float* X, float * Y) {
  int index = 0;
  for(int m = 0; m < M; m++){
    const float* X_m = X + m * N;
    for(int p = 0; p < rowk; p++){
      for(int q = 0; q < stride; q++){
        Y[index] = 0;
        for(int r= 0; r < colk; r++){
          Y[index] += X_m[r * stride + q] * W_k[p * colk + r];
        }
        index++;
      }
    }
  }
}

/*

The code computes Y = XW^T

Input matrix X \in R^{M \times N}

Wsize is an array of size 2K containing the number of rows and columns each
Kronecker factors {P_0, Q_0 ..., P_{K−1}, Q_{K-1}} : \prod_{k=0}^{K−1} P_k = D : : \prod_{k=0}^{K−1} Q_k = N
                 
Kronecker factors \{W_0, ..., W_{K−1}\} : W_k \in R^{P_k\times Q_k},

Ysize is an array of length K such that \sum_{k = 0}^{K - 1} Ysize[i] = size of Y, 1.e each entry gives the memory required to
store the product with corresponding Kronecker factors.  

size of Y =  O(MDK) = \sum_{k = 0}^{K - 1} Ysize[i]

*/

void cpu_kronecker_forward(int M, int N, const float *X,
                                int K, const int* Wsize,
                                const float* W, const int* Ysize, float *Y) {
  int offset = 0, k;
  const float* X_k = X;
  float *Y_k = Y;
  for(k = 0; k <  K; k++)  {
    int rowk = Wsize[2 * k];
    int colk = Wsize[2 * k + 1];
    int stride = N / colk;
    const float* W_k = W + offset;
    if(k > 0) {
      assert(Ysize[k-1] == M * N);
    }
    cpu_kronecker_forward_kernel(M, N, rowk,
                                                colk, stride, W_k,
                                                X_k, Y_k);
    N = stride;
    M = M * rowk;
    offset += rowk * colk;
    X_k = Y_k;
    assert(Ysize[k] == M * N);
    Y_k += Ysize[k];
  }
}


__global__ void gpu_kronecker_forward_kernel(int M, int N, int rowk,
                                                  int colk, int stride,
                                                  const float* W_k, const float* X, float * Y) {
  int TH = threadIdx.x + blockIdx.x * blockDim.x;
  if(TH < M * rowk * stride) {
    int m = TH / (rowk * stride);
    int pq = TH % (rowk * stride);
    int p = pq / stride;
    int q = pq % stride;
    const float* X_m = X + m * N;
    Y[TH] = 0;
    for(int r= 0; r < colk; r++){
      Y[TH] += X_m[r * stride + q] * W_k[p * colk + r];
    }
  }
}

/*
  size of Y is O(MDK)
*/
void gpu_kronecker_forward(int M, int N, const float *X,
                                int K, const int* Wsize,
                                const float* W, const int* Ysize, float *Y) {
  int offset = 0, k;
  const float* X_k = X;
  float *Y_k = Y;
  for(k = 0; k <  K; k++)  {
    int rowk = Wsize[2 * k];
    int colk = Wsize[2 * k + 1];
    int stride = N / colk;
    const float* W_k = W + offset;
    if(k > 0) {
      assert(Ysize[k-1] == M * N);
    }
    gpu_kronecker_forward_kernel<<<CUDA_GET_BLOCKS(M * rowk * stride), CUDA_NUM_THREADS>>>(M, N, rowk,
                                                                                                colk, stride, W_k,
                                                                                                X_k, Y_k);
    N = stride;
    M = M * rowk;
    offset += rowk * colk;
    X_k = Y_k;
    assert(Ysize[k] == M * N);
    Y_k += Ysize[k];
  }
}


void cpu_kronecker_backward_kernel1(int M, int N, int rowk, int colk,
                                                   int stride, const float* X, const float* gradY, float* gradW_k) {
  int index = 0;
  for(int m = 0; m < M; m++){
    const float* X_m = X + m * N;
    for(int p = 0; p < rowk; p++){
      for(int q = 0; q < stride; q++){
        for(int r= 0; r < colk; r++){
          gradW_k[p * colk + r] += X_m[r * stride + q] * gradY[index];
        }
        index++;
      }
    }
  }
}

void cpu_kronecker_backward_kernel2(int M, int N, int rowk,
                                         int colk, int stride,
                                         const float* W_k, const float* gradY, float * gradX) {
  int index = 0;
  for(int m = 0; m < M; m++){
    const float* gradY_m = gradY + m * N;
    for(int r= 0; r < colk; r++){
      for(int q = 0; q < stride; q++){
        gradX[index] = 0;
        for(int p = 0; p < rowk; p++){
          gradX[index] += gradY_m[p * stride + q] * W_k[p * colk + r];
        }
        index++;
      }
    }
  }
  //cout<<index<<endl;
}

/*
Y = XW^T = forward
gradX = gradY * W
gradW += gradY^T * X
size of Y is O(MDK)
I use the memory of Y to do intermediate computations so the
Y values are  not preserved
*/
void cpu_kronecker_backward(int M, int N, int D, const float *X,
                                 int K, const int* Worder,
                                 const float* W, const float* gradY,
                                 const int* Ysize, float* Y, float* gradW, float* gradX) {
  int P  = M * D;
  int stride = 1;
  int stride1 = 1;
  int offsetW = 0;
  int offsetY = 0;
  for(int k = 0 ; k < K; k++) {
    offsetW += Worder[2* k] * Worder[2 * k + 1];
    offsetY += Ysize[k];
  }
  const float* gradY_k = gradY;
  for(int k = K - 1; k >= 0; k--) {
    offsetY -= Ysize[k];
    int rowk = Worder[2 * k];
    int colk = Worder[2 * k + 1];
    int Q = stride * colk;
    int S = stride1 * rowk;
    P /= rowk;
    offsetW -= rowk * colk;
    const float* W_k = W + offsetW;
    float* gradW_k = gradW + offsetW;
    const float* X_k = nullptr;
    float* gradX_k = nullptr;
    if(k == 0) {
      X_k = X;
      gradX_k = gradX;

    }
    else {
      gradX_k =  Y + offsetY - Ysize[k - 1];
      X_k = gradX_k;
    }
    cpu_kronecker_backward_kernel1(P, Q, rowk, colk, stride, X_k,
                                        gradY_k, gradW_k);
    cpu_kronecker_backward_kernel2(P, S, rowk, colk, stride, W_k,
                                        gradY_k, gradX_k);
    gradY_k = gradX_k;
    stride = Q;
    stride1 = S * colk / rowk;
  }
}

__global__ void gpu_kronecker_backward_kernel1(int M, int N, int rowk, int colk,
                                                    int stride, const float* X, const float* gradY, float* gradW_k) {
  int TH = threadIdx.x + blockIdx.x * blockDim.x;
  if(TH < M * rowk * stride) {
    int m = TH / (rowk * stride);
    int pq = TH % (rowk * stride);
    int p = pq / stride;
    int q = pq % stride;
    const float* X_m = X + m * N;
    for(int r= 0; r < colk; r++){
      atomicAdd(&gradW_k[p * colk + r], X_m[r * stride + q] * gradY[TH]);
    }
  }
}

__global__  void gpu_kronecker_backward_kernel2(int M, int N, int rowk,
                                                     int colk, int stride,
                                                     const float* W_k, const float* gradY, float * gradX) {
  int TH = threadIdx.x + blockIdx.x * blockDim.x;
  if(TH < M * colk * stride) {
    int m = TH / (colk * stride);
    int rq = TH % (colk * stride);
    int r = rq / stride;
    int q = rq % stride;
    const float* gradY_m = gradY + m * N;
    gradX[TH] = 0;
    for(int p = 0; p < rowk; p++){
      gradX[TH] += gradY_m[p * stride + q] * W_k[p * colk + r];
    }
  }
}

/*
Y = XW = forward
gradX = gradY * W^T
gradW += X^T * gradY
size of Y is O(MDK)
I use the memory of Y to do intermediate computations so the
Y values are  not preserved
*/
void gpu_kronecker_backward(int M, int N, int D, const float *X,
                                 int K, const int* Worder,
                                 const float* W, const float* gradY,
                                 const int* Ysize, float* Y, float* gradW, float* gradX) {
  int P  = M * D;
  int stride = 1;
  int stride1 = 1;
  int offsetW = 0;
  int offsetY = 0;
  for(int k = 0 ; k < K; k++) {
    offsetW += Worder[2* k] * Worder[2 * k + 1];
    offsetY += Ysize[k];
  }
  const float* gradY_k = gradY;
  for(int k = K - 1; k >= 0; k--) {
    offsetY -= Ysize[k];
    int rowk = Worder[2 * k];
    int colk = Worder[2 * k + 1];
    int Q = stride * colk;
    int S = stride1 * rowk;
    P /= rowk;
    offsetW -= rowk * colk;
    const float* W_k = W + offsetW;
    float* gradW_k = gradW + offsetW;
    const float* X_k = nullptr;
    float* gradX_k = nullptr;
    if(k == 0) {
      X_k = X;
      gradX_k = gradX;
    }
    else {
      gradX_k =  Y + offsetY - Ysize[k - 1];
      X_k = gradX_k;
    }
    gpu_kronecker_backward_kernel1<<<CUDA_GET_BLOCKS(P * rowk * stride),
        CUDA_NUM_THREADS>>>(P, Q, rowk, colk, stride, X_k, gradY_k, gradW_k);
    gpu_kronecker_backward_kernel2<<<CUDA_GET_BLOCKS(P * colk * stride),
        CUDA_NUM_THREADS>>>(P, S, rowk, colk, stride, W_k, gradY_k, gradX_k);
    gradY_k = gradX_k;
    stride = Q;
    stride1 = S * colk / rowk;
  }
}


void sort(int N, int* arr) {
  int i, key, j;
  for (i = 1; i < N; i++) {
    key = arr[i];
    j = i-1;
    while (j >= 0 && arr[j] > key) {
      arr[j+1] = arr[j];
      j = j-1;
    }
    arr[j+1] = key;
  }
}

void sieve(int N, int** primes, int &num_primes) {
  assert(N >= 2);
  bool *A = new bool [N + 1];
  memset(A, 0, (N + 1)  * sizeof(bool));
  for(int i = 2; i <= sqrt(N); i++) {
    if(A[i] == false) {
      for(int  j = i * i; j <= N; j += i) {
        A[j] =  true;
      }
    }
  }
  num_primes = 0;
  for(int i = 2; i <= N; i++) {
    if(!A[i]) { num_primes++; }
  }
  (*primes) = new int [num_primes];
  int count  = 0;
  for(int i = 2; i <= N; i++) {
    if(!A[i]) {
      (*primes)[count] = i;
      count++;
    }
  }
  delete [] A;
}


/*
  Given a whole number N with >= M factors, the function returns M numbers
   such that the prduct of these M numbers is N.
   Moreover \sum_i^{M} factors_sizes[i]^2 is minimum.
*/
void get_factor_sizes(int N, int M,
                      int* factor_sizes) {
  std::vector<int> factors;
  int num_primes = 0;
  int* primes = nullptr;
  sieve(N, &primes, num_primes);
  for(int i =0; i < num_primes; i++) {
    int n = N;
    while (true) {
      if(n % primes[i] == 0 && n > 0) {
        factors.push_back(primes[i]);
        n /= primes[i];
      }
      else {
        break;
      }
    }
  }
  delete [] primes;
  assert(factors.size() >= M);
  if(factors.size() == M) {
    std::sort(factors.begin(), factors.end());
    for(int  i = 0; i < M; i++) {
      factor_sizes[i] = factors[i];
    }
  }
  else {
    int K = factors.size();
    int i =  0;
    while(K > M) {
      factors[i + 1] *= factors[i];
      std::sort(factors.begin(), factors.end());
      i++;
      K--;
    }
    for(int j = i; j < factors.size(); j++) {
      factor_sizes[j - i] = factors[j];
    }
  }
}



void knuth_shuffle(std::vector<int>& A) {
  for(int i = 0 ; i < A.size(); i++) {
    int r = i +  int((A.size() - i - 1) * (float(rand()) / float(RAND_MAX)));
    int tmp = A[i];
    A[i] = A[r];
    A[r] =  tmp;
  }
}


int kronecker_output_memory(int M, int N, int K, 
                            const int* Wsize, int* Ysize) {
  int size = 0;
  for(int  k =0; k < K; k++) {
    int rowk = Wsize[2 * k];
    int colk = Wsize[2 * k +  1];
    int stride = N / colk;
    N = stride;
    M = M * rowk;
    Ysize[k] = M * N;
    size += Ysize[k];
  }
  return size;
}



void kronecker_factor_sizes(int input_dim, int output_dim, int max_factors,
                            std::vector<int>& input_dim_factor, std::vector<int>& output_dim_factor) {

  int *input_dim_factors = new int [max_factors];
  int *output_dim_factors = new int [max_factors];
  get_factor_sizes(input_dim, max_factors, input_dim_factors);
  get_factor_sizes(output_dim, max_factors, output_dim_factors);
  for(int i = 0; i < max_factors; i++) {
    input_dim_factor.push_back(input_dim_factors[i]);
    output_dim_factor.push_back(output_dim_factors[i]);
  }
  delete [] input_dim_factors;
  delete [] output_dim_factors;

}

void print(int N , const float *X) {
  for(int i = 0; i < N; i++){
    std::cout<<X[i]<<" ";
  }
  std::cout<<std::endl<<std::endl;
}


float relative_error(int N,  const float* grad_analytic,  const float* grad_numerical) {
  float rel_error = 0;
  for(int i = 0 ; i < N; i++) {
    float a =  grad_analytic[i];
    float b =  grad_numerical[i];
    rel_error += fabs(a - b) / (fabs(a) + fabs(b));
  }
  rel_error /= float(N);
  return rel_error;
}


void forward_backward_checker(int M, int N, int D, int max_factors) {
  std::vector<int> factorsN, factorsD;
  kronecker_factor_sizes(N, D, max_factors, factorsN, factorsD);
  assert(factorsN.size() == factorsD.size());
  int numF = factorsN.size();
  int *Wsize = new int [numF * 2];
  int wsize = 0;
  //std::cout<<numF<<std::endl;
  for(int f =0; f < numF; f++) {
    Wsize[2 * f] = factorsD[f];
    Wsize[2 * f + 1] = factorsN[f];
    wsize += factorsD[f] * factorsN[f];
  }
  int* Ysize = new int [numF];
  int ysize = kronecker_output_memory(M, N, numF, Wsize, Ysize);

  cublasHandle_t h;
  cublasCreate(&h);
  
  float *W  =nullptr;
  CUDA_CHECK(cudaMallocHost(&W, wsize * sizeof(float)));
  float *X = nullptr;
  CUDA_CHECK(cudaMallocHost(&X, M * N * sizeof(float)));
  float *Y1 = new float [ysize];
  float *Y2 = new float [ysize];

  float *W_gpu  =nullptr;
  CUDA_CHECK(cudaMalloc(&W_gpu, wsize * sizeof(float)));
  float *X_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&X_gpu, M * N * sizeof(float)));
  float *Y_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&Y_gpu, ysize * sizeof(float)));

  float *gradW_analytical  = new float [wsize];
  float *gradW_numerical  = new float [wsize];
  float *gradX_analytical = new float [M * N];
  float *gradX_numerical  = new float [M * N];

  float *Ypdx = new float [ysize];
  float *Ymdx = new float [ysize];
  float *gradY = new float [M * D]; 

  float *gradY_gpu = nullptr; 
  CUDA_CHECK(cudaMalloc(&gradY_gpu, M * D * sizeof(float)));
  float *gradW_analytical_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&gradW_analytical_gpu, wsize * sizeof(float)));
  float *gradX_analytical_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&gradX_analytical_gpu, M * N * sizeof(float)));
  float *gradW_numerical_gpu = nullptr;
  CUDA_CHECK(cudaMallocHost(&gradW_numerical_gpu, wsize * sizeof(float)));
  float *gradX_numerical_gpu = nullptr;
  CUDA_CHECK(cudaMallocHost(&gradX_numerical_gpu, M * N * sizeof(float)));
  float *Ypdx_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&Ypdx_gpu, ysize * sizeof(float)));
  float *Ymdx_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&Ymdx_gpu, ysize * sizeof(float)));

  for(int  i = 0; i < M * N; i++) {
    X[i] = float(rand()) / float(RAND_MAX);
  }
  for(int  i = 0; i < M * D; i++) {
    gradY[i] = float(rand()) / float(RAND_MAX);
  }
  for(int  i = 0; i < wsize; i++) {
    W[i] = float(rand()) / float(RAND_MAX);
  }
  CUDA_CHECK(cudaMemcpy(W_gpu, W, wsize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(X_gpu, X, M *  N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gradY_gpu, gradY, M *  D * sizeof(float),
                        cudaMemcpyHostToDevice));
  
  cpu_kronecker_forward(M, N, X, numF, Wsize,
                             W, Ysize, Y1);
  gpu_kronecker_forward(M, N, X_gpu, numF, Wsize,
                             W_gpu, Ysize, Y_gpu);
  CUDA_CHECK(cudaMemcpy(Y2, Y_gpu, ysize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::cout<<"Relative error of forward pass cpu vs gpu "<<relative_error(ysize, Y1, Y2)<<std::endl;
  cpu_kronecker_forward(M, N, X, numF, Wsize,
                             W, Ysize, Y1);
  memset(gradW_analytical, 0, sizeof(float) * wsize);
  CUDA_CHECK(cudaMemcpy(gradW_analytical_gpu, gradW_analytical, wsize * sizeof(float),
                        cudaMemcpyHostToDevice));
  cpu_kronecker_backward(M, N, D, X, numF, Wsize, W,
                              gradY, Ysize, Y1, gradW_analytical, gradX_analytical);
  gpu_kronecker_backward(M, N, D, X, numF, Wsize, W,
                              gradY_gpu, Ysize, Y_gpu, gradW_analytical_gpu, gradX_analytical_gpu);

  CUDA_CHECK(cudaMemcpy(gradW_numerical, gradW_analytical_gpu, wsize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gradX_numerical, gradX_analytical_gpu, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::cout<<"Relative error of backward pass on gradW cpu vs gpu "<<relative_error(wsize, gradW_numerical, gradW_analytical)<<std::endl;
  std::cout<<"Relative error of backward pass on gradX cpu vs gpu "<<relative_error(M * N, gradX_numerical, gradX_analytical)<<std::endl;


  
  /*
    Numerical gradient check follows.
  */
  int offset = ysize - M * D;
  float dx = 1e-2;
  for(int i =0 ; i < M * N; i++) {
    X[i] += dx;
    CUDA_CHECK(cudaMemcpy(X_gpu + i, X + i, sizeof(float),
                          cudaMemcpyHostToDevice));
    cpu_kronecker_forward(M, N, X, numF, Wsize, W, Ysize, Ypdx);
    gpu_kronecker_forward(M, N, X_gpu, numF, Wsize, W_gpu, Ysize, Ypdx_gpu);
    X[i] -= float(2) * dx;
    CUDA_CHECK(cudaMemcpy(X_gpu + i, X + i, sizeof(float),
                          cudaMemcpyHostToDevice));
    cpu_kronecker_forward(M, N, X, numF, Wsize, W, Ysize, Ymdx);
    gpu_kronecker_forward(M, N, X_gpu, numF, Wsize, W_gpu, Ysize, Ymdx_gpu);
    X[i] += dx;
    CUDA_CHECK(cudaMemcpy(X_gpu + i, X + i, sizeof(float),
                          cudaMemcpyHostToDevice));
    gradX_numerical[i] = 0;
    for(int  j  = 0; j < M * D; j++) {
      gradX_numerical[i] += (Ypdx[offset + j] - Ymdx[offset + j]) * gradY[j] / (float(2) * dx);
    }
    float sout;
    cublasSdot(h, M * D, Ypdx_gpu + offset, 1,
               gradY_gpu, 1, &sout);
    gradX_numerical_gpu[i] = sout;
    cublasSdot(h, M * D, Ymdx_gpu + offset, 1, gradY_gpu, 1, &sout);
    gradX_numerical_gpu[i] -= sout;
    gradX_numerical_gpu[i] /= (float(2) * dx);

  }

  for(int i =0 ; i < wsize; i++) {
    W[i] += dx;
    CUDA_CHECK(cudaMemcpy(W_gpu + i, W + i, sizeof(float),
                          cudaMemcpyHostToDevice));
    cpu_kronecker_forward(M, N, X, numF, Wsize, W, Ysize, Ypdx);
    gpu_kronecker_forward(M, N, X_gpu, numF, Wsize, W_gpu, Ysize, Ypdx_gpu);
    W[i] -= float(2) * dx;
    CUDA_CHECK(cudaMemcpy(W_gpu + i, W + i, sizeof(float),
                          cudaMemcpyHostToDevice));
    cpu_kronecker_forward(M, N, X, numF, Wsize, W, Ysize, Ymdx);
    gpu_kronecker_forward(M, N, X_gpu, numF, Wsize, W_gpu, Ysize, Ymdx_gpu);
    W[i] += dx;
    CUDA_CHECK(cudaMemcpy(W_gpu + i, W + i, sizeof(float),
                          cudaMemcpyHostToDevice));
    gradW_numerical[i] = 0;
    for(int  j  = 0; j < M * D; j++) {
      gradW_numerical[i] += (Ypdx[offset + j] - Ymdx[offset + j]) * gradY[j] / (float(2) * dx);
    }

    float sout;
    cublasSdot(h, M * D, Ypdx_gpu + offset, 1,
               gradY_gpu, 1, &sout);
    gradW_numerical_gpu[i] = sout;
    cublasSdot(h, M * D, Ymdx_gpu + offset, 1, gradY_gpu, 1, &sout);
    gradW_numerical_gpu[i] -= sout;
    gradW_numerical_gpu[i] /= (float(2) * dx);
  }

  std::cout<<"Relative error cpu gradW "<<relative_error(wsize, gradW_numerical, gradW_analytical)<<std::endl;
  std::cout<<"Relative error cpu gradX "<<relative_error(M * N, gradX_numerical, gradX_analytical)<<std::endl;

  CUDA_CHECK(cudaMemcpy(gradW_analytical, gradW_analytical_gpu, wsize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gradX_analytical, gradX_analytical_gpu, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::cout<<"Relative error gpu gradW "<<relative_error(wsize, gradW_numerical_gpu, gradW_analytical)<<std::endl;
  std::cout<<"Relative error gpu gradX "<<relative_error(M * N, gradX_numerical_gpu, gradX_analytical)<<std::endl;


  //print(M * N, gradX_numerical);
  //print(M * N, gradX_analytical);

  delete [] Y1;
  delete [] Y2;
  delete [] Wsize;
  delete [] Ysize;
  delete [] gradW_analytical;
  delete [] gradW_numerical;
  delete [] gradX_analytical;
  delete [] gradX_numerical;
  delete [] Ypdx;
  delete [] Ymdx;
  delete [] gradY;
  CUDA_CHECK(cudaFreeHost(W));
  CUDA_CHECK(cudaFreeHost(X));
  CUDA_CHECK(cudaFree(W_gpu));
  CUDA_CHECK(cudaFree(X_gpu));
  CUDA_CHECK(cudaFree(Y_gpu));

  CUDA_CHECK(cudaFree(gradW_analytical_gpu));
  CUDA_CHECK(cudaFree(gradX_analytical_gpu));
  CUDA_CHECK(cudaFreeHost(gradW_numerical_gpu));
  CUDA_CHECK(cudaFreeHost(gradX_numerical_gpu));
  CUDA_CHECK(cudaFree(Ypdx_gpu));
  CUDA_CHECK(cudaFree(Ymdx_gpu));  
}

int main() {

  forward_backward_checker(1, 16, 16, 3);
  forward_backward_checker(5, 32, 32, 4);
  forward_backward_checker(5, 32, 16, 4);
  forward_backward_checker(5, 16, 32, 4);
  forward_backward_checker(10, 32, 16, 1);
  forward_backward_checker(15, 16, 32, 3);
  forward_backward_checker(5, 16, 32, 1);
  forward_backward_checker(5, 64, 32, 1);
  forward_backward_checker(5, 64, 32, 5);
  /*
  forward_backward_checker(3, 128, 256, 5);
  forward_backward_checker(400, 4, 4, 2);
  forward_backward_checker(700, 200, 10000, 5);
  */
}
