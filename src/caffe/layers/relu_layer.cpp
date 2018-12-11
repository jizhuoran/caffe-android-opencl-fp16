#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


template <>
void ReLULayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {

  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();

  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().math_program, "ReLUForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&count));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_float), (void *)&negative_slope));  

  size_t global_size = CAFFE_GET_BLOCKS(count);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  

}

template <>
void ReLULayer<half>::Forward_gpu(const vector<Blob<half>*>& bottom,
    const vector<Blob<half>*>& top) {

  const half* bottom_data = bottom[0]->gpu_data();
  half* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  half negative_slope = this->layer_param_.relu_param().negative_slope();

  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().math_program, "ReLUForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  half_b negative_slope_half = float2half_impl(negative_slope);
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&top_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&count));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_half), (void *)&negative_slope_half));  

  size_t global_size = CAFFE_GET_BLOCKS(count);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  

}



template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Backward_cpu(top, propagate_down, bottom);
}



// #ifdef CPU_ONLY
// STUB_GPU(ReLULayer);
// #endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
