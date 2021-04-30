#include <iostream>
#include <random>

typedef float real;

real dotProduct(const real *x, const real *y, int d) {
  real result = 0;
  for (int i = 0; i < d; i++) {
    result += x[i] * y[i];
  }
  return result;
}
// y = ax + y
void axpy(const real *x, real *y, real a, int d) {
  for (int i = 0; i < d; i++) {
    y[i] = a * x[i] + y[i];
  }
}

void scale(real *x, real a, int d) {
  for (int i = 0; i < d; i++) {
    x[i] *= a;
  }
}

real l2Norm(const real *x, int d) {
  real result = 0;
  for (int i = 0; i < d; i++) {
    result += pow(x[i], 2);
  }
  return sqrt(result);
}
real lInfinityNorm(const real *x, int d) {
  real result = real(0);
  for (int i = 0; i < d; i++) {
    result = std::max(result, abs(x[i]));
  }
  return result;
}

real relativeError(const real *x1, const real *x2, int d) {
  real relError = 0;
  for (int i = 0; i < d; i++) {
    real a = x1[i];
    real b = x2[i];
    relError += abs(a - b) / (abs(a) + abs(b));
  }
  relError /= real(d);
  return relError;
}

class LeastSquares {
  real *weight_;
  real bias_;
  int dimension_;

public:
  LeastSquares(int dimension) {
    dimension_ = dimension;
    weight_ = new real[dimension];
  }
  ~LeastSquares() { delete[] weight_; }
  void trainWithGradientDescent(const real *X, const real *y, int numPoints,
                                real learningRate = 0.1) {
    real *gradientWeight = new real[dimension_];
    real gradientBias = 0;
    real loss;
    real epsilon = 1e-5;
    int epoch = 0;
    memset(weight_, 0, sizeof(real) * dimension_);
    bias_ = real(0);
    while (true) {
      memset(gradientWeight, 0, sizeof(real) * dimension_);
      gradientBias = 0;
      loss = 0;
      for (int i = 0; i < numPoints; i++) {
        const real *xi = X + i * dimension_;
        real prediction = dotProduct(xi, weight_, dimension_) + bias_;
        real difference = prediction - y[i];
        loss += pow(difference, 2);
        axpy(xi, gradientWeight, real(2) * difference, dimension_);
        gradientBias += real(2) * difference;
      }
      if (lInfinityNorm(gradientWeight, dimension_) < epsilon) {
        std::cout << "Gradient descent converged!, finished training\n";
        return;
      }
      scale(gradientWeight, real(1.0 / real(numPoints)), dimension_);
      gradientBias *= real(1.0 / real(numPoints));
      loss *= real(1.0 / real(numPoints));
      axpy(gradientWeight, weight_, -learningRate, dimension_);
      bias_ -= learningRate * gradientBias;
      if (epoch % 10 == 0) {
        std::cout << "Epoch: " << epoch << " MSE loss " << loss << std::endl;
      }
      epoch++;
    }
    delete[] gradientWeight;
  }
  // x \n \mathbb{R}^{dimension_}
  real predict(const real *x) {
    return dotProduct(x, weight_, dimension_) + bias_;
  }

  const real *weight() { return (const real *)weight_; }
  real bias() { return bias_; }
};

void generateData(int numPoints, int dimension, real *X, real *y,
                  real *weightStar, real *biasStar) {

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> uniformDist(0, 1);
  std::normal_distribution<real> normalDist(0, 1);

  for (int i = 0; i < dimension; i++) {
    weightStar[i] = normalDist(e2);
  }
  *biasStar = normalDist(e2);
  for (int i = 0; i < numPoints * dimension; i++) {
    X[i] = uniformDist(e2);
  }
  for (int i = 0; i < numPoints; i++) {
    real *Xi = X + i * dimension;
    y[i] = *biasStar;
    for (int j = 0; j < dimension; j++) {
      y[i] += weightStar[j] * Xi[j];
    }
  }
}

void printVector(const real *x, int d) {
  for (int i = 0; i < d; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
}
int main() {
  int numPoints = 100;
  int dimension = 25;
  real *X = new real[numPoints * dimension];
  real *y = new real[numPoints];
  real *weightStar = new real[dimension];
  real biasStar;
  generateData(numPoints, dimension, X, y, weightStar, &biasStar);

  LeastSquares model(dimension);
  model.trainWithGradientDescent(X, y, numPoints);
  const real *weight = model.weight();
  real bias = model.bias();
  real relError = relativeError(weightStar, weight, dimension);
  std::cout << "Relative error " << relError << std::endl;

  std::cout << "Original: " << std::endl;
  printVector(weightStar, dimension);
  std::cout << "Learned: " << std::endl;
  printVector(weight, dimension);

  delete[] X;
  delete[] y;
  delete[] weightStar;
  return 0;
}
