#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TileLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const TileParameter& tile_param = this->layer_param_.tile_param();
  axis_ = bottom[0]->CanonicalAxisIndex(tile_param.axis());
  CHECK(tile_param.has_tiles()) << "Number of tiles must be specified";
  tiles_ = tile_param.tiles();
  CHECK_GT(tiles_, 0) << "Number of tiles must be positive.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[axis_] = bottom[0]->shape(axis_) * tiles_;
  top[0]->Reshape(top_shape);
  outer_dim_ = bottom[0]->count(0, axis_);
  inner_dim_ = bottom[0]->count(axis_);
}

template <typename Dtype>
void TileLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < outer_dim_; ++i) {
    for (int t = 0; t < tiles_; ++t) {
      caffe_copy(inner_dim_, bottom_data, top_data);
      top_data += inner_dim_;
    }
    bottom_data += inner_dim_;
  }
}

template <typename Dtype>
void TileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < outer_dim_; ++i) {
    caffe_copy(inner_dim_, top_diff, bottom_diff);
    top_diff += inner_dim_;
    for (int t = 1; t < tiles_; ++t) {
      caffe_axpy(inner_dim_, Dtype(1), top_diff, bottom_diff);
      top_diff += inner_dim_;
    }
    bottom_diff += inner_dim_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TileLayer);
#elif USE_OPENCL



// template <typename Dtype>
// void TileLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {

 
//   const Dtype* bottom_data = bottom[0]->gpu_data();
//   Dtype* top_data = top[0]->mutable_gpu_data();
//   const int count = bottom[0]->count();
//   const int dim = bottom[0]->count(2);
//   const int channels = bottom[0]->channels();
//   const Dtype* slope_data = this->blobs_[0]->gpu_data();
//   const int div_factor = channel_shared_ ? channels : 1;

// #ifndef FORWARD_ONLY 
//   // For in-place computation
//   if (top[0] == bottom[0]) {
//     caffe_cl_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
//   }
// #endif


//   cl_int ret;

//   cl_kernel kernel = clCreateKernel(Caffe::Get().program, "PReLUForward", &ret);
//   OPENCL_CHECK(ret);

//   // Set arguments for kernel
//   OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
//   OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&slope_data));  
//   OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_data));  
//   OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&count));
//   OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&channels));
//   OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&dim));
//   OPENCL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&div_factor));

//   size_t global_size = CAFFE_GET_BLOCKS(count);
  
//   OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  





// }


// template <typename Dtype>
// void TileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down,
//     const vector<Blob<Dtype>*>& bottom) {

//   Backward_cpu(top, propagate_down, bottom);
// }



#endif

INSTANTIATE_CLASS(TileLayer);
REGISTER_LAYER_CLASS(Tile);

}  // namespace caffe
