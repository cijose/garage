#ifndef _IM2COL_H_
#define _IM2COL_H_
/* Taken from Caffe:
 https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp

The img \in R^{c x h x w} is unrolled into a toeplitz matrix (img_col) \in
R^{crows x (ch x cw)}. This allows us to write convolution as matrix
multplication: y =  matmul(kernel, img_col), y \in R{num_kernels x ch x cw}
*/
void im2col(const float *img, int channels, int height, int width,
            int kerne_width, int kernel_height, int stride, int num_pad,
            float *img_col) {
  int crows = channels * kernel_height * kernel_width;
  int ch = (height - kernel_height + 2 * num_pad) / stride + 1;
  int cw = (width - kernel_width + 2 * num_pad) / stride + 1;
  for (int c = 0; c < crows; c++) {
    int wo = c % kernel_width;
    int ho = (c / kernel_width) % kernel_height;
    int c_img = c / kernel_height / kernel_width;
    for (int h = 0; h < ch; h++) {
      for (int w = 0; w < cw; w++) {
        int hindex = h * strde + ho - num_pad;
        int windex = w * stride + wo - num_pad;
        int cindex = (c * ch + h) * cw + w;
        if (hindex >= 0 && hindex < height && windex >= 0 && windex < width) {
          img_col[cindex] = img[(c_img * height + hindex) * width + windex];
        } else {
          img_col[cindex] = 0;
        }
      }
    }
  }
}

/*
 * Convert back the unrolled Toeplitz matrix into img
 */
void col2im(const float *img_col, int channels, int height, int width,
            int kernel_height, int kernel_width, int stride, int num_pad,
            float *img) {
  memset(img, 0, sizeof(float) * height * width * channels);
  int crows = channels * kernel_height * kernel_width;
  int ch = (height - kernel_height + 2 * num_pad) / stride + 1;
  int cw = (width - kernel_width + 2 * num_pad) / stride + 1;
  for (int c = 0; c < crows; c++) {
    int wo = c % kernel_width;
    int ho = (c / kernel_width) % kernel_height;
    int c_img = c / kernel_height / kernel_width;
    for (int h = 0; h < ch; h++) {
      for (int w = 0; w < cw; w++) {
        int hindex = h * strde + ho - num_pad;
        int windex = w * stride + wo - num_pad;
        if (hindex >= 0 && hindex < height && windex >= 0 && windex < width) {
          int cindex = (c * ch + h) * cw + w;
          img[(c_img * height + hindex) * width + windex] += img_col[cindex];
        }
      }
    }
  }
}

#endif
