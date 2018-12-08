#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_
 
#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda, cl_mem *gpu_ptr) {
#ifdef USE_OPENCL

  if (Caffe::mode() == Caffe::GPU) {

#ifdef ZERO_COPY

    cl_int ret;

    *gpu_ptr = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &ret);
    OPENCL_CHECK(ret);

    *ptr = clEnqueueMapBuffer(Caffe::Get().commandQueue, *gpu_ptr, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);
    OPENCL_CHECK(ret);
    *use_cuda = true;
#else
    *ptr = malloc(size);
    
#endif

    return;
  }

#endif


#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifdef USE_OPENCL
  if (use_cuda) {
    free(ptr);

    // CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory { 
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifdef FORWARD_LESS_MEM
  void default_reference();
  void increase_reference();
  void decrease_reference();
  void zhihan_release();
#endif

// #ifndef CPU_ONLY
//   void async_gpu_push(const cudaStream_t& stream);
// #endif

 private:
  void check_device();
#ifdef FORWARD_LESS_MEM
  int refer_num;
#endif
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  cl_mem gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;

#ifdef USE_OPENCL
  cl_int ret;
#endif 

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
