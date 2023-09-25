#include "math.h"
#include "mnist.h"
#include <assert.h>
#include <stdbool.h>

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

typedef struct LinearModel {
  int dim;
  int num_classes;
  float *w;
} LinearModel;
void get_predictions(LinearModel *model, const SparseVector *x,
                     float *predictions, int num_classes) {

  memset(predictions, 0, sizeof(float) * num_classes);
  for (int i = 0; i < num_classes; i++) {
    const float *wi = model->w + i * model->dim;
    for (int j = 0; j < x->size; j++) {
      int index = x->indices[j];
      float value = (float)x->values[j] / ((float)255);
      predictions[i] += wi[index] * value;
    }
  }
}

LinearModel *train_linear_model(const SparseVector *train_data, int num_train,
                                int num_classes, int input_dim) {

  LinearModel *model = (LinearModel *)malloc(sizeof(LinearModel));
  model->dim = input_dim;
  model->num_classes = num_classes;

  model->w = (float *)(malloc(sizeof(float) * input_dim * num_classes));
  memset(model->w, 0, sizeof(float) * input_dim * num_classes);
  float *predictions = (float *)malloc(sizeof(float) * num_classes);
  float *grad = (float *)malloc(sizeof(float) * num_classes);
  int *indices = (int *)malloc(sizeof(int) * num_train);
  for (int i = 0; i < num_train; i++) {
    indices[i] = i;
  }
  float loss = 0;
  float eta = (float)1e-3;
  for (int e = 0; e < 10; e++) {
    knuth_shuffle(num_train, indices);
    for (int iter = 0; iter < num_train; iter++) {
      int r = indices[iter];
      get_predictions(model, &train_data[r], predictions, num_classes);
      float lossr =
          softmax_loss(predictions, grad, train_data[r].label, num_classes);
      for (int i = 0; i < num_classes; i++) {
        float *wi = model->w + i * model->dim;
        for (int j = 0; j < train_data[r].size; j++) {
          int index = train_data[r].indices[j];
          float value = (float)train_data[r].values[j] / ((float)255);
          wi[index] -= eta * grad[i] * value;
        }
      }
      if (iter == 0) {
        loss = lossr;
      } else {
        loss = 0.99 * loss + 0.01 * lossr;
      }
      if (iter % 1000 == 0) {
        printf("Loss after iteration: %d is: %f\n", iter, loss);
      }
    }
  }

  free(predictions);
  free(grad);
  free(indices);
  return model;
}

float test_linear_model(LinearModel *model, const SparseVector *test_data,
                        int num_test) {

  float *predictions = (float *)malloc(sizeof(float) * model->num_classes);
  float accuracy = 0;
  for (int j = 0; j < num_test; j++) {
    get_predictions(model, &test_data[j], predictions, model->num_classes);
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

  LinearModel *model =
      train_linear_model(train_data, num_train, num_classes, 784);
  printf("Testing....\n");
  float accuracy_train = test_linear_model(model, train_data, num_train);
  printf("Accuracy on the train set is %f\n", accuracy_train);
  float accuracy_test = test_linear_model(model, test_data, num_test);
  printf("Accuracy on the test set is %f\n", accuracy_test);
  free_sparse_vector(train_data, num_train);
  free_sparse_vector(test_data, num_test);
  free(model->w);
  free(model);
}
