#ifndef _MNIST_H_
#define _MNIST_H_
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

#endif
