#include <cmath>
#include <iostream>

typedef float real;

real entropy(const real *probs, int n) {
  real e = real(0);
  for (int i = 0; i < n; i++) {
    e += probs[i] * log(probs[i]);
  }
  return -e;
}

void entropyGradient(const real *probs, real *gradient, int n) {
  for (int i = 0; i < n; i++) {
    gradient[i] = -(log(probs[i]) + real(1));
  }
}

real infNorm(const real *vec, int n) {
  real nrm = real(0);
  for (int i = 0; i < n; i++) {
    nrm = std::max(nrm, abs(vec[i]));
  }
  return nrm;
}

void maximiseEntropy(const real *probs, int n) {

  real lr = 0.1;
  real *gradient = new real[n];
  real *maxProbs = new real[n];
  memcpy(maxProbs, probs, sizeof(real) * n);
  std::cout << "Initial probabilities: ";
  for (int j = 0; j < n; j++) {
    std::cout << maxProbs[j] << " ";
  }
  std::cout << std::endl;
  int iter = 0;
  while (true) {
    real e = entropy(maxProbs, n);
    entropyGradient(maxProbs, gradient, n);
    if (infNorm(gradient, n) < 1e-6) {
      break;
    }
    for (int j = 0; j < n; j++) {
      maxProbs[j] += lr * gradient[j];
    }
    if (iter % 10 == 0) {
      std::cout << "Iteration: " << iter << " Entropy: " << e << std::endl;
    }
    iter++;
  }

  real normaliser = real(0);
  for (int j = 0; j < n; j++) {
    normaliser += maxProbs[j];
  }
  std::cout << "Maximal entropy probabilites: (should be a uniform): ";
  for (int j = 0; j < n; j++) {
    std::cout << maxProbs[j] / normaliser << " ";
  }
  std::cout << std::endl;
  delete[] gradient;
  delete[] maxProbs;
}

int main() {
  int n = 20;
  real *probs = new real[n];
  real normalizer = real(0);
  for (int i = 0; i < n; i++) {
    probs[i] = 1.0 + (real)rand() / RAND_MAX;
    normalizer += probs[i];
  }
  for (int i = 0; i < n; i++) {
    probs[i] /= normalizer;
  }
  maximiseEntropy(probs, n);
}
