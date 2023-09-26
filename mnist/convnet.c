#include "linalg.h"
#include "ndarray.h"
typedef struct Convolution2DLayer {
  size_t num_pad;
  size_t stride;
  bool requires_grad_input;
  NDArray *kernel;
  NDArray *grad_kernel;
  NDArray *input;
  NDArray *cache;
  NDArray *output;
} Convolution2dLayer;

NDArray *get_ndarray(NDArray *array, size_t batch_size, size_t channels,
                     size_t height, size_t width) {

  size_t size = batch_size * channels * height * width;
  if (array == NULL) {
    array = new_4darray(batch_size, channels, height, width);
  } else if (array->size != size) {
    free(array->data_ptr);
    free(array);
    array = new_4darray(batch_size, channels, height, width);
  } else {
    reshape_4darray(array, batch_size, channels, height, width);
  }
  return array;
}
NDArray *convolution2D_forward(NDarray *input,
                               Convolution2DLayer *conv2d_layer) {
  assert(input != NULL);
  assert(conv2d_layer != NULL);
  assert(input->num_dims == 4);
  assert(conv2d_layer->kernel != NULL);
  NDArray *kernel = conv2d_layer->kernel;
  assert(kernel->num_dims == 4);
  const size_t batch_size = input->dims[0];
  const size_t input_channels = input->dims[1];
  const size_t input_height = input->dims[2];
  const size_t input_width = input->dims[3];

  const size_t num_kernels = kernel->dims[0];
  const size_t kernel_dim = kernel->dims[1];
  const size_t kernel_height = kernel->dims[2];
  const size_t kernel_width = kernel->dims[3];

  assert(kernel_dim == input_channels);

  size_t stride = conv2d_layer->stride;
  size_t num_pad = conv2d_layer->num_pad;

  assert((input_height + 2 * num_pad - kernel_height) % stride == 0);
  assert((input_width + 2 * num_pad - kernel_width) % stride == 0);
  size_t output_height =
      (input_height + 2 * num_pad - kernel_height) / stride + 1;
  size_t output_width = (input_width + 2 * num_pad - kernel_width) / stride + 1;
  conv2d_layer->cache =
      get_ndarray(conv2d_layer->cache, batch_size,
                  input_channels * kernel_height * kernel_width, output_height,
                  output_width);
  conv2d_layer->output = get_ndarray(conv2d_layer->output, batch_size,
                                     num_kernels, output_height, output_width);
  conv2d_layer->input = input;
  // To do: Use threads here
  for (size_t i = 0; i < batch_size; i++) {
    const float *input_datai =
        input->data + i * (input_channels * input_height * input_width);
    float *input_data_coli =
        cache->data + i * (input_channels * kernel_height * kernel_width *
                           output_height * output_width);
    im2col(input_datai, input_channels, input_height, input_width,
           kernel_height, kernel_width, stride, num_pad, input_data_coli);
    float *output_datai = conv2d_layer->output->data +
                          i * (num_kernels * output_height * output_width);
    gemm(CblasNoTrans, CblasNoTrans, num_kernels, output_height * output_width,
         input_chanels * kernel_height * kernel_width, 1, kernel->data,
         input_data_coli, 0, output_datai);
  }
  return conv2d_layer->output;
}

NDArray *convolution2D_backward(NDarray *grad_output,
                                Convolution2DLayer *conv2d_layer) {
  assert(grad_output != NULL);
  assert(grad_output->num_dims == 4);
  assert(conv2d_layer != NULL);
  assert(conv2d_layer->input != NULL);
  assert(conv2d_layer->kernel != NULL);
  assert(conv2d_layer->grad_kernel != NULL);
  NDArray *kernel = conv2d_layer->kernel;
  NDArray *grad_kernel = conv2d_layer->grad_kernel;
  assert(kernel->num_dims == 4);
  assert(grad_kernel->num_dims == 4);
  assert(kernel->dims[0] == grad_kernel->dims[0]);
  assert(kernel->dims[1] == grad_kernel->dims[1]);
  assert(kernel->dims[2] == grad_kernel->dims[2]);
  assert(kernel->dims[3] == grad_kernel->dims[3]);
  NDArray *input = conv2d_layer->input;
  const size_t batch_size = input->dims[0];
  const size_t input_channels = input->dims[1];
  const size_t input_height = input->dims[2];
  const size_t input_width = input->dims[3];

  const size_t num_kernels = kernel->dims[0];
  const size_t kernel_dim = kernel->dims[1];
  const size_t kernel_height = kernel->dims[2];
  const size_t kernel_width = kernel->dims[3];

  assert(kernel_dim == input_channels);

  size_t stride = conv2d_layer->stride;
  size_t num_pad = conv2d_layer->num_pad;

  assert((input_height + 2 * num_pad - kernel_height) % stride == 0);
  assert((input_width + 2 * num_pad - kernel_width) % stride == 0);
  size_t output_height =
      (input_height + 2 * num_pad - kernel_height) / stride + 1;
  size_t output_width = (input_width + 2 * num_pad - kernel_width) / stride + 1;
  assert(grad_output->dims[0] == batch_size);
  assert(grad_output->dims[1] == num_kernels);
  assert(grad_output->dims[2] == output_height);
  assert(grad_output->dims[3] == output_width);
  memset(grad_kernel->data, 0, sizeof(float) * grad_kernel->size);
  for (size_t i = 0; i < batch_size; i++) {
    float *grad_output_datai =
        grad_output->data + i * (num_kernels * output_height * output_width);
    float *grad_data_coli =
        cache->data + i * (input_channels * kernel_height * kernel_width *
                           output_height * output_width);
    gemm(CblasNoTrans, CblasTrans, num_kernels,
         input_chanels * kernel_height * kernel_width,
         output_height * output_width, 1, grad_output_datai, grad_data_coli, 1,
         grad_kernel->data);
    gemm(CblasTrans, CblasNoTrans, input_chanels * kernel_height * kernel_width,
         output_height * output_width, num_kernels, 1, kernel,
         grad_output_datai, 0, grad_data_coli);
    float *input_datai =
        input->data + i * (input_channels * input_height * input_width);
    if (conv2d_layer->requires_grad_input) {
      col2im(grad_data_coli, input_channels, input_height, input_width,
             kernel_height, kernel_width, stride, num_pad, input_datai);
    }
  }
  return conv2d_layer->requires_grad_input ? input : NULL;
}
