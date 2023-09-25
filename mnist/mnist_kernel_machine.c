#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct SparseVector {
  int *indices;
  unsigned char *values;
  unsigned char label;
  int size;
} SparseVector;

int reverse_int(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_mnist_file(char *filepath, unsigned char *mnist_data,
                     int len_meta_data, int size) {

  FILE *fp;
  int *meta_data = (int *)malloc(sizeof(int) * len_meta_data);
  if ((fp = fopen(filepath, "r")) == NULL) {
    fprintf(stderr, "couldn't open mnist file");
    exit(-1);
  }
  fread(meta_data, sizeof(int), len_meta_data, fp);
  for (int i = 0; i < len_meta_data; i++) {
    meta_data[i] = reverse_int(meta_data[i]);
  }
  fread(mnist_data, sizeof(unsigned char), size, fp);
  fclose(fp);
  free(meta_data);
}

SparseVector *convert_to_sparse(unsigned char *data, unsigned char *labels,
                                int num_data, int dim) {
  SparseVector *sv = (SparseVector *)malloc(num_data * sizeof(SparseVector));
  for (int i = 0; i < num_data; i++) {
    unsigned char *di = data + i * dim;
    int num_non_zero = 0;
    for (int j = 0; j < dim; j++) {
      if (di[j] != 0) {
        num_non_zero++;
      }
    }
    sv[i].label = labels[i];
    sv[i].size = num_non_zero;
    sv[i].indices = (int *)malloc(sizeof(int) * num_non_zero);
    sv[i].values =
        (unsigned char *)malloc(sizeof(unsigned char) * num_non_zero);
    int k = 0;
    for (int j = 0; j < dim; j++) {
      if (di[j] != 0) {
        sv[i].indices[k] = j;
        sv[i].values[k] = di[j];
        k++;
      }
    }
  }
  return sv;
}

void free_sparse_vector(SparseVector *sv, int num) {
  for (int i = 0; i < num; i++) {
    free(sv[i].indices);
    free(sv[i].values);
  }
}

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
                      int degree) {
  return exp(-sparse_distance(x1, x2));
}
float polynomial_kernel(const SparseVector *x1, const SparseVector *x2,
                        int degree) {
  return pow(sparse_dot_product(x1, x2), degree);
}

float softmax_loss(const float *predictions, float *gradient,
                   unsigned char label, int num_classes) {
  // printf("Label is %d\n", (int)label);
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
                     int num_test, int num_classes) {

  memset(predictions, 0, sizeof(float) * num_test * num_classes);
  for (int i = 0; i < num_test; t++) {
#pragma omp parallel for
    for (int i = 0; i < num_train; i++) {
      kernel_row[i] = gaussian_kernel(&train_data[i], &test_data[t], 2);
    }
    for (int i = 0; i < num_train; i++) {
      float ki = kernel_row[i];
      const float *wi = w + i * num_classes;
      for (int j = 0; j < num_classes; j++) {
        predictions[j] += wi[j] * ki;
      }
    }
  }
}
float train_kernel_machine(const SparseVector *train_data, float *w,
                           int num_train, int num_classes) {
  float *kernel_row = (float *)malloc(sizeof(float) * num_train);
  float *predictions = (float *)malloc(sizeof(float) * num_classes);
  float *grad = (float *)malloc(sizeof(float) * num_classes);
  float loss = 0;
  float t = 1;
  for (int iter = 0; iter < num_train; iter++) {
    int r = rand() % num_train;
    get_predictions(train_data, &train_data[r], w, kernel_row, predictions,
                    num_train, num_classes);
    float lossr =
        softmax_loss(predictions, grad, train_data[r].label, num_classes);
    for (int i = 0; i < num_train; i++) {
      float ki = kernel_row[i];
      float *wi = w + i * num_classes;
      for (int j = 0; j < num_classes; j++) {
        wi[j] -= grad[j] * ki * (float)1e-1;
      }
    }
    t += 1;
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
                          int num_train, int num_test, int num_classes) {

  float *kernel_row = (float *)malloc(sizeof(float) * num_train);
  float *predictions = (float *)malloc(sizeof(float) * num_classes);
  float accuracy = 0;
  for (int j = 0; j < num_test; j++) {
    get_predictions(train_data, &test_data[j], w, kernel_row, predictions,
                    num_train, num_classes);
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

  printf("Training....\n");
  float *w = (float *)malloc(sizeof(float) * num_train * num_classes);
  memset(w, 0, sizeof(float) * num_train * num_classes);

  train_kernel_machine(train_data, w, num_train, num_classes);
  printf("Testing....\n");
  float accuracy_train = test_kernel_machine(train_data, train_data, w,
                                             num_train, num_train, num_classes);
  printf("Accuracy on the train set is %f\n", accuracy_train);
  float accuracy_test = test_kernel_machine(train_data, test_data, w, num_train,
                                            num_test, num_classes);
  printf("Accuracy on the test set is %f\n", accuracy_test);
  free_sparse_vector(train_data, num_train);
  free_sparse_vector(test_data, num_test);
  free(w);
}
