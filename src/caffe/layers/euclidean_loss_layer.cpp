#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}




#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#elif USE_OPENCL
TEMP_GPU(EuclideanLossLayer);
// template <typename Dtype>
// void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   int count = bottom[0]->count();
//   caffe_gpu_sub(
//       count,
//       bottom[0]->gpu_data(),
//       bottom[1]->gpu_data(),
//       diff_.mutable_gpu_data());
//   Dtype dot;
//   caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
//   Dtype loss = dot / bottom[0]->num() / Dtype(2);
//   top[0]->mutable_cpu_data()[0] = loss;
// }

// template <typename Dtype>
// void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

//   Backward_cpu(top, propagate_down, bottom);
// }

#endif


INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
