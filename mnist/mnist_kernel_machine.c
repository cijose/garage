#include "mnist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float sparse_distance(const SparseVector *x1, const SparseVector *x2) {
  float result = 0;
  int i = 0, j = 0;
  while (i < x1->size && j < x2->size) {
    if (x1->indices[i] == x2->indices[j]) {
      result += pow(((float)(x1->values[i]) / (float)(255)) -
                        (float)(x2->values[j] / (float)(255)),
                    2);
      i++;
      j++;
    } else if (x1->indices[i] < x2->indices[j]) {
      i++;
    } else {
      j++;
    }
  }
  return result;
}

float sparse_dot_product(const SparseVector *x1, const SparseVector *x2) {
  float result = 0;
  int i = 0, j = 0;
  while (i < x1->size && j < x2->size) {
    if (x1->indices[i] == x2->indices[j]) {
      result += ((float)(x1->values[i]) / (float)(255)) *
                (float)(x2->values[j] / (float)(255));
      i++;
      j++;
    } else if (x1->indices[i] < x2->indices[j]) {
      i++;
    } else {
      j++;
    }
  }
  return result;
}

float gaussian_kernel(const SparseVector *x1, const SparseVector *x2,
                      float sigma) {
  return exp(-sparse_distance(x1, x2) / (2.0 * sigma * sigma));
}
float polynomial_kernel(const SparseVector *x1, const SparseVector *x2,
                        int degree) {
  return pow(sparse_dot_product(x1, x2), degree);
}

float softmax_loss(const float *predictions, float *gradient,
                   unsigned char label, int num_classes) {
  float max_val = predictions[0];
  for (int i = 1; i < num_classes; i++) {
    if (predictions[i] > max_val) {
      max_val = predictions[i];
    }
  }
  float normalizer = 0;
  for (int i = 0; i < num_classes; i++) {
    normalizer += exp(predictions[i] - max_val);
  }
  float loss = log(normalizer) + max_val - predictions[(int)(label)];
  for (int i = 0; i < num_classes; i++) {
    float v = (int)(label) == i ? 1 : 0;
    gradient[i] = exp(predictions[i] - max_val) / normalizer - v;
  }
  return loss;
}

void get_predictions(const SparseVector *train_data,
                     const SparseVector *test_data, const float *w,
                     float *kernel_row, float *predictions, int num_train,
                     int num_classes, float sigma) {

  memset(predictions, 0, sizeof(float) * num_classes);
#pragma omp parallel for
  for (int i = 0; i < num_train; i++) {
    kernel_row[i] = gaussian_kernel(&train_data[i], test_data, sigma);
  }
  for (int i = 0; i < num_train; i++) {
    float ki = kernel_row[i];
    const float *wi = w + i * num_classes;
    for (int j = 0; j < num_classes; j++) {
      predictions[j] += wi[j] * ki;
    }
  }
}
float train_kernel_machine(const SparseVector *train_data, float *w,
                           int num_train, int num_classes, float sigma) {
  float *kernel_row = (float *)malloc(sizeof(float) * num_train);
  float *predictions = (float *)malloc(sizeof(float) * num_classes);
  float *grad = (float *)malloc(sizeof(float) * num_classes);
  float loss = 0;
  float eta = 1e-3;
  for (int iter = 0; iter < num_train; iter++) {
    int r = rand() % num_train;
    get_predictions(train_data, &train_data[r], w, kernel_row, predictions,
                    num_train, num_classes, sigma);
    float lossr =
        softmax_loss(predictions, grad, train_data[r].label, num_classes);
    for (int i = 0; i < num_train; i++) {
      float ki = kernel_row[i];
      float *wi = w + i * num_classes;
      for (int j = 0; j < num_classes; j++) {
        wi[j] -= grad[j] * ki * eta;
      }
    }
    if (iter == 0) {
      loss = lossr;
    } else {
      loss = 0.99 * loss + 0.01 * lossr;
    }
    if (iter % 10 == 0) {
      printf("Loss after iteration: %d is: %f\n", iter, loss);
    }
  }
  free(kernel_row);
  free(predictions);
  free(grad);
  return loss;
}

float test_kernel_machine(const SparseVector *train_data,
                          const SparseVector *test_data, const float *w,
                          int num_train, int num_test, int num_classes,
                          float sigma) {

  float *kernel_row = (float *)malloc(sizeof(float) * num_train);
  float *predictions = (float *)malloc(sizeof(float) * num_classes);
  float accuracy = 0;
  for (int j = 0; j < num_test; j++) {
    get_predictions(train_data, &test_data[j], w, kernel_row, predictions,
                    num_train, num_classes, sigma);
    float max_value = predictions[0];
    int max_index = 0;
    for (int i = 1; i < num_classes; i++) {
      if (predictions[i] > max_value) {
        max_value = predictions[i];
        max_index = i;
      }
    }
    accuracy += (float)(max_index == ((int)(test_data[j].label)));
  }
  free(kernel_row);
  free(predictions);
  return accuracy / (float)(num_test);
}

int main() {
  int num_train = 60000, num_test = 1000, dim = 784, num_classes = 10;
  unsigned char *train_images =
      (unsigned char *)malloc(sizeof(unsigned char) * num_train * dim);
  unsigned char *train_labels =
      (unsigned char *)malloc(sizeof(unsigned char) * num_train);
  unsigned char *test_images =
      (unsigned char *)malloc(sizeof(unsigned char) * num_test * dim);
  unsigned char *test_labels =
      (unsigned char *)malloc(sizeof(unsigned char) * num_test);

  printf("Reading training data\n");
  read_mnist_file("train-images.idx3-ubyte", train_images, 4, num_train * dim);
  read_mnist_file("train-labels.idx1-ubyte", train_labels, 2, num_train);
  printf("Converting to sparse vector\n");
  SparseVector *train_data =
      convert_to_sparse(train_images, train_labels, num_train, dim);
  free(train_images);
  free(train_labels);

  printf("Reading test data\n");
  read_mnist_file("t10k-images.idx3-ubyte", test_images, 4, num_test * dim);
  read_mnist_file("t10k-labels.idx1-ubyte", test_labels, 2, num_test);
  printf("Converting to sparse vector\n");
  SparseVector *test_data =
      convert_to_sparse(test_images, test_labels, num_test, dim);
  free(test_images);
  free(test_labels);

  float sigma = 10.0;
  printf("Training....\n");
  float *w = (float *)malloc(sizeof(float) * num_train * num_classes);
  memset(w, 0, sizeof(float) * num_train * num_classes);

  train_kernel_machine(train_data, w, num_train, num_classes, sigma);
  printf("Testing....\n");
  float accuracy_train = test_kernel_machine(
      train_data, train_data, w, num_train, num_train, num_classes, sigma);
  printf("Accuracy on the train set is %f\n", accuracy_train);
  float accuracy_test = test_kernel_machine(train_data, test_data, w, num_train,
                                            num_test, num_classes, sigma);
  printf("Accuracy on the test set is %f\n", accuracy_test);
  free_sparse_vector(train_data, num_train);
  free_sparse_vector(test_data, num_test);
  free(w);
}
