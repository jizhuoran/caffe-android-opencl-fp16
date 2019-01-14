#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void BatchReindexLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom[1]->num_axes());
  vector<int> newshape;
  newshape.push_back(bottom[1]->shape(0));
  for (int i = 1; i < bottom[0]->shape().size(); ++i) {
    newshape.push_back(bottom[0]->shape()[i]);
  }
  top[0]->Reshape(newshape);
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::check_batch_reindex(int initial_num,
                                                   int final_num,
                                                   const Dtype* ridx_data) {
  for (int i = 0; i < final_num; ++i) {
    CHECK_GE(ridx_data[i], 0)
        << "Index specified for reindex layer was negative.";
    CHECK_LT(ridx_data[i], initial_num)
        << "Index specified for reindex layer was greater than batch size.";
  }
}

template<typename Dtype>
void BatchReindexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }
  int inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  const Dtype* in = bottom[0]->cpu_data();
  const Dtype* permut = bottom[1]->cpu_data();
  Dtype* out = top[0]->mutable_cpu_data();
  for (int index = 0; index < top[0]->count(); ++index) {
    int n = index / (inner_dim);
    int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}


#ifdef CPU_ONLY
STUB_GPU(BatchReindexLayer);
#elif USE_OPENCL


template <typename Dtype>
void BatchReindexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }

  int nthreads = top[0]->count();
  int inner_dim = bottom[0]->count() / bottom[0]->shape(0);

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* permut_data = bottom[1]->gpu_data();
  const Dtype* top_data = top[0]->mutable_gpu_data();


  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().math_program, "BRForward", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&permut_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_data));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&inner_dim)); 
  OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&nthreads)); 

  size_t global_size = CAFFE_GET_BLOCKS(nthreads);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}



#endif


INSTANTIATE_CLASS(BatchReindexLayer);
REGISTER_LAYER_CLASS(BatchReindex);

}  // namespace caffe
