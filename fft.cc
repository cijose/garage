#include "fft.h"
#include  <cmath>
#include <cassert>
#define TWOPI   (2.0*M_PI)

/*
 A simple inplace FFT implementation.
*/
void cpu_inplace_bitreverse(int N, float* Y) {
  for (int i = 0, j = 0; i < N; i++) {
    if (i < j) {
      float tmp = Y[i];
      Y[i] = Y[j];
      Y[j] = tmp;
    }
    int bit = ~i & (i + 1);
    int rev = (N / 2) / bit;
    j ^= (N - 1) & ~(rev - 1);
  }
}

void cpu_inplace_bitreverse(int M, int N, float *Y) {
  float *Y_m = Y;
  for(int m = 0; m < 2 * M; m++) {
    cpu_inplace_bitreverse(N, Y_m);
    Y_m += N;
  }
}

void cpu_fft_kernel1(int MN, float *Y) {
  for(int i = 0; i <  MN; i += 2) {
    float A =  Y[i];
    float B =  Y[i + 1];
    Y[i] =  A + B;
    Y[i + 1] = A - B;
  }
}

void cpu_fft_kernel2(int MN, int bit, float fft_sign,
                     float *realY, float *imagY) {
  for(int i = 0; i <  MN; i++) {
    if(!(bit & i)) {
      float theta = fft_sign * TWOPI * i / float(2 * bit);
      //The following two lines are slow
      float omega_real =  cos(theta);
      float omega_imag =  sin(theta);
      int k = i;
      int j = (bit | i);
      float AR =  realY[k];
      float AI =  imagY[k];
      float BR =  realY[j];
      float BI =  imagY[j];
      float tBR = omega_real * BR - omega_imag * BI;
      BI = omega_real * BI + omega_imag * BR;
      realY[k] += float(tBR);
      imagY[k] += float(BI);
      realY[j] = float(AR - tBR);
      imagY[j] = float(AI - BI);
    }
  }
}

void cpu_fft(int M, int N, bool isfft, float* Y) {
  assert(!(N == 0) && !(N & (N - 1)) == true);
  cpu_inplace_bitreverse(M, N, Y);
  int MN = M * N;
  float fft_sign  = isfft == 1 ? float(1) : float(-1);
  cpu_fft_kernel1(2 * MN, Y);
  for(int bit = 2; bit < N; bit <<= 1) {
    cpu_fft_kernel2(MN, bit, fft_sign, Y, Y + MN);
  }
  for(int i =0; i < 2 * MN; i++) { Y[i] = Y[i] / sqrt(float(N)); }
}
