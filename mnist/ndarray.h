#ifndef _NDARRAY_H_
#define _NDARRAY_H_
#define NDARRAY_MAX_DIMS 4
#include <assert.h>
typedef struct NDArray {
  float *data;
  int num_dims;
  size_t dims[NDARRAY_MAX_DIMS];
  size_t strides[NDARRAY_MAX_DIMS];
} NDArray;

NDarray *new_ndarray(int num_dims, size_t *dims) {
  assert(num_dims >= 1 && num_dims <= NDARRAY_MAX_DIMS);
  NDArray *ndarray = (NDArray *)malloc(sizeof(NDArray));
  ndarray->num_dims = num_dims;
  for (int i = 0; i < NDARRAY_MAX_DIMS; i++) {
    ndarray->dims[i] = 1;
  }
  ndarray->size = 1;
  for (int i = 0; i < num_dims; i++) {
    ndarray->dims[i] = dims[i];
    ndarray->size *= dims[i];
  }
  size_t stride = 1;
  for (int i = NDARRAY_MAX_DIMS - 1; i >= 0; i--) {
    ndarray->strides[i] = stride;
    stride *= ndarray->dims[i];
  }
  ndarray->data = (float *)malloc(sizeof(float) * ndarray->size);
  return ndarray;
}
NDArray *new_1darray(size_t d0) {
  size_t dims[1] = {d0};
  return new_1d_array(1, dims);
}
NDArray *new_2darray(size_t d0, size_t d1) {
  size_t dims[2] = {d0, d1};
  return new_1d_array(2, dims);
}

NDArray *new_3darray(size_t d0, size_t d1, size_t d2) {
  size_t dims[3] = {d0, d1, d2};
  return new_1d_array(3, dims);
}

NDArray *new_4darray(size_t d0, size_t d1, size_t d2, size_t d3) {
  size_t dims[4] = {d0, d1, d2, d3};
  return new_1d_array(4, dims);
}
bool is_contiguous_ndarray(NdArray *ndarray) {
  assert(NDARRAY_MAX_DIMS == 4);
  return ndarray->strides[3] == 1 &&
         ndarray->strides[2] == ndarray->strides[3] * ndarray->dims[3] &&
         ndarray->strides[1] == ndarray->strides[2] * ndarray->dims[2] &&
         ndarray->strides[0] == ndarray->strides[1] * ndarray->dims[1];
}
size_t get_memory_index(size_t index, size_t num_dims, size_t *slices,
                        const size_t *shape, const size_t *strides) {
  size_t mem_index = 0;
  for (int i = 0; i < num_dims; i++) {
    mem_index += (index / slices[i]) % shape[i] * strides[i];
  }
  return mem_index;
}

void make_contiguous_strides(NDArray *ndarray) {
  size_t stride = 1;
  for (int i = NDARRAY_MAX_DIMS - 1; i >= 0; i--) {
    ndarray->strides[i] = stride;
    stride *= ndarray->dims[i];
  }
}
void make_contiguous_ndarray(NDArray *ndarray) {
  if (is_contigous(ndarray)) {
    return;
  }
  size_t carry[ndarray->num_dims];
  carry[ndarray->num_dims - 1] = 1;
  for (int i = ndarray->num_dims - 2; i >= 0; i--) {
    carry[i] = carry[i + 1] * ndarray->shape[i + 1];
  }

  float *tmp = (float *)malloc(sizeof(float) * size);
  for (size_t index = 0; index < ndarray->size; index++) {
    tmp[idt] = ndarray->data_ptr[cpu_get_memory_index(
        index, carry, ndarray->shape, ndarray->strides)];
  }
  memcpy(ndarray->data, tmp, sizeof(float) * size);
  free(tmp);
  make_contiguous_strides(ndarray);
}

void reshape_ndarray(NDArray *ndarray, int num_dims, size_t *dims) {
  assert(num_dims >= 1 && num_dims <= NDARRAY_MAX_DIMS);
  size_t size = 1;
  for (int i = 0; i < num_dims; i++) {
    size *= dims[i];
  }
  assert(size == ndarray->size);
  for (int i = 0; i < NDARRAY_MAX_DIMS; i++) {
    ndarray->dims[i] = 1;
  }
  for (int i = 0; i < num_dims; i++) {
    ndarray->dims[i] = dims[i];
  }
  ndarray->num_dims = num_dims;
  size_t stride = 1;
  for (int i = NDARRAY_MAX_DIMS - 1; i >= 0; i--) {
    ndarray->strides[i] = stride;
    stride *= ndarray->dims[i];
  }
}

void reshape_1darray(NDArray *ndarray, size_t d0) {
  size_t dims[1] = {d0};
  reshape_ndarray(ndarray, 1, dims);
}

void reshape_2darray(NDArray *ndarray, size_t d0, size_t d1) {
  size_t dims[2] = {d0, d1};
  reshape_ndarray(ndarray, 2, dims);
}

void reshape_3darray(NDArray *ndarray, size_t d0, size_t d1, size_t d2) {
  size_t dims[3] = {d0, d1, d2};
  reshape_ndarray(ndarray, 3, dims);
}
void reshape_4darray(NDArray *ndarray, size_t d0, size_t d1, size_t d2,
                     size_t d3) {
  size_t dims[4] = {d0, d1, d2, d3};
  reshape_ndarray(ndarray, 4, dims);
}

void permute_ndarray(NDarray *ndarray, int num_axes, const size_t *axes) {
  assert(num_axes == ndarray->num_dims);
  size_t new_dims[num_axes];
  int32_t new_strides[num_axes];
  for (int i = 0; i < num_axes; i++) {
    new_dims[i] = ndarray->dims[axes[i]];
    new_strides[i] = ndarray->strides[axes[i]];
  }
  for (int i = 0; i < num_axes; i++) {
    ndarray->dims[i] = new_dims[i];
    ndarray->strides[i] = new_strides[i];
  }
}
#endif
