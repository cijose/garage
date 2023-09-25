#include "math.h"
#include "mnist.h"
#include <assert.h>
#include <stdbool.h>

float std_randn() {
  float u = ((float)rand() / (RAND_MAX)) * 2 - 1;
  float v = ((float)rand() / (RAND_MAX)) * 2 - 1;
  float r = u * u + v * v;
  if (r == 0 || r > 1)
    return std_randn();
  float c = sqrt(-2 * log(r) / r);
  return u * c;
}

void vec_rademacher(int n, float *x) {
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    x[i] = std_randn() > 0 ? (float)(1) : (float)(-1);
  }
}

void vec_uniform(int n, const float r1, const float r2, float *x) {
  for (int i = 0; i < n; i++) {
    float a = ((float)rand() / (RAND_MAX));
    x[i] = a * (r2 - r1) + r1;
  }
}

void vec_cosine(int n, const float *x, float *y) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    y[i] = cos(x[i]);
}

void vec_scale(int n, float alpha, const float *x, float *y) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i];
}

// z = a * x + y

void vec_axpy(int n, float a, const float *x, const float *y, float *z) {
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
    z[j] = a * x[j] + y[j];
  }
}

// z = x * y

void vec_mul(int n, const float *x, const float *y, float *z) {
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
    z[j] = x[j] * y[j];
  }
}

// z = x * y
// x is sparse
void sparse_vec_mul(int n, const SparseVector *x, const float *y, float *z) {
#pragma omp parallel for
  for (int j = 0; j < x->size; j++) {
    int index = x->indices[j];
    float value = (float)x->values[j];
    z[index] = value * y[index] / ((float)255);
  }
}
void vec_hadamard_transform(int n, float *x) {
  bool ispow2 = !(n == 0) && !(n & (n - 1));
  assert(ispow2 == true);
  for (int j = 0; j < n; j += 2) {
    size_t k = j + 1;
    float t1 = x[j];
    float t2 = x[k];
    x[j] = (t1 + t2);
    x[k] = (t1 - t2);
  }
  for (int bit = 2; bit < n; bit <<= 1) {
    for (int j = 0; j < n; j++) {
      if ((bit & j) == 0) {
        size_t k = bit | j;
        float t1 = x[j];
        float t2 = x[k];
        x[j] = (t1 + t2);
        x[k] = (t1 - t2);
      }
    }
  }
}

typedef struct FastRandomFeaturesModel {
  int dim;
  int num_blocks;
  int num_classes;
  float sigma;
  float *d1;
  float *d2;
  float *d3;
  float *b;
  float *w;
  float *wb;
} FastRandomFeaturesModel;

void fast_sorf_transform(const SparseVector *x, FastRandomFeaturesModel *m,
                         float *y) {
  int output_dim = m->dim * m->num_blocks;
  memset(y, 0, sizeof(float) * output_dim);
  for (int i = 0; i < m->num_blocks; i++) {
    const float *d1i = m->d1 + i * m->dim;
    const float *d2i = m->d2 + i * m->dim;
    const float *d3i = m->d3 + i * m->dim;
    float *yi = y + i * m->dim;
    sparse_vec_mul(m->dim, x, d3i, yi);
    vec_hadamard_transform(m->dim, yi);
    vec_mul(m->dim, yi, d2i, yi);
    vec_hadamard_transform(m->dim, yi);
    vec_mul(m->dim, yi, d1i, yi);
    vec_hadamard_transform(m->dim, yi);
  }
  float scaler = 1.0 / (m->sigma * (float)(m->dim));
  vec_scale(output_dim, scaler, y, y);
  vec_axpy(output_dim, (float)(1), y, m->b, y);
  vec_cosine(output_dim, y, y);
  vec_scale(output_dim, sqrt((float)(2) / (float)(output_dim)), y, y);
}

void knuth_shuffle(int n, int *indices) {
  for (int i = 0; i < n; i++) {
    int r = rand() % (n - i) + i;
    indices[i] ^= indices[r];
    indices[r] ^= indices[i];
    indices[i] ^= indices[r];
  }
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

void get_predictions(FastRandomFeaturesModel *model, const SparseVector *x,
                     float *phix, float *predictions, int num_classes) {

  int feature_dim = model->dim * model->num_blocks;
  memset(phix, 0, sizeof(float) * feature_dim);
  fast_sorf_transform(x, model, phix);
  memset(predictions, 0, sizeof(float) * num_classes);
  for (int i = 0; i < num_classes; i++) {
    const float *wi = model->w + i * feature_dim;
    for (int j = 0; j < feature_dim; j++) {
      predictions[i] += wi[j] * phix[j];
    }
  }
}

FastRandomFeaturesModel *
train_fast_random_features(const SparseVector *train_data, int num_train,
                           int num_classes, int input_dim, int num_blocks) {

  FastRandomFeaturesModel *model =
      (FastRandomFeaturesModel *)malloc(sizeof(FastRandomFeaturesModel));

  int feature_dim = input_dim * num_blocks;
  model->sigma = 10.0;
  model->dim = input_dim;
  model->num_blocks = num_blocks;
  model->num_classes = num_classes;
  model->d1 = (float *)malloc(sizeof(float) *
                              (feature_dim * (5 + num_classes) + num_classes));
  model->d2 = model->d1 + feature_dim;
  model->d3 = model->d2 + feature_dim;
  model->b = model->d3 + feature_dim;
  model->w = model->b + feature_dim;
  model->wb = model->w + feature_dim;
  vec_rademacher(3 * feature_dim, model->d1);
  vec_uniform(feature_dim, 0, (float)(2.0 * M_PI), model->b);
  for (int i = 0; i < feature_dim; i++) {
    printf("%f ", model->b[i]);
  }
  memset(model->w, 0, sizeof(float) * feature_dim * num_classes);
  float *phix = (float *)malloc(sizeof(float) * feature_dim);
  float *predictions = (float *)malloc(sizeof(float) * num_classes);
  float *grad = (float *)malloc(sizeof(float) * num_classes);
  int *indices = (int *)malloc(sizeof(int) * num_train);
  for (int i = 0; i < num_train; i++) {
    indices[i] = i;
  }
  float loss = 0;
  float t = 1;
  float eta = (float)1.0;
  int steps = 0;
  for (int e = 0; e < 10; e++) {
    knuth_shuffle(num_train, indices);
    for (int iter = 0; iter < num_train; iter++) {
      int r = indices[iter];
      get_predictions(model, &train_data[r], phix, predictions, num_classes);
      float lossr =
          softmax_loss(predictions, grad, train_data[r].label, num_classes);
      for (int i = 0; i < num_classes; i++) {
        float *wi = model->w + i * feature_dim;
        for (int j = 0; j < feature_dim; j++) {
          wi[j] -= eta * grad[i] * phix[j];
        }
      }
      t += 1;
      if (steps == 0) {
        loss = lossr;
      } else {
        loss = 0.99 * loss + 0.01 * lossr;
      }
      steps += 1;
      if (iter % 1000 == 0) {
        printf("Loss after epoch: %d iteration: %d is: %f\n", e, iter, loss);
      }
    }
  }

  free(phix);
  free(predictions);
  free(grad);
  free(indices);
  return model;
}

float test_fast_random_features(FastRandomFeaturesModel *model,
                                const SparseVector *test_data, int num_test) {

  int feature_dim = model->dim * model->num_blocks;
  float *phix = (float *)malloc(sizeof(float) * feature_dim);
  float *predictions = (float *)malloc(sizeof(float) * model->num_classes);
  float accuracy = 0;
  for (int j = 0; j < num_test; j++) {
    get_predictions(model, &test_data[j], phix, predictions,
                    model->num_classes);
    float max_value = predictions[0];
    int max_index = 0;
    for (int i = 1; i < model->num_classes; i++) {
      if (predictions[i] > max_value) {
        max_value = predictions[i];
        max_index = i;
      }
    }
    accuracy += (float)(max_index == ((int)(test_data[j].label)));
  }
  free(phix);
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

  FastRandomFeaturesModel *model =
      train_fast_random_features(train_data, num_train, num_classes, 1024, 1);
  printf("Testing....\n");
  float accuracy_train =
      test_fast_random_features(model, train_data, num_train);
  printf("Accuracy on the train set is %f\n", accuracy_train);
  float accuracy_test = test_fast_random_features(model, test_data, num_test);
  printf("Accuracy on the test set is %f\n", accuracy_test);
  free_sparse_vector(train_data, num_train);
  free_sparse_vector(test_data, num_test);
  free(model->d1);
  free(model);
}
