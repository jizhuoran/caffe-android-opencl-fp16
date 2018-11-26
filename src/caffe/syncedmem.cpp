#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  // CUDA_CHECK(cudaGetDevice(&device_)); TODOTODOO
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  // CUDA_CHECK(cudaGetDevice(&device_)); TODOTODOO
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifdef USE_OPENCL
  if (gpu_ptr_ && own_gpu_data_) {
    OPENCL_CHECK(clReleaseMemObject(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

#ifdef FORWARD_LESS_MEM
void SyncedMemory::default_reference() {
    refer_num = 0;
}

void SyncedMemory::increase_reference() {
    refer_num++;
}

void SyncedMemory::decrease_reference() {
    refer_num--;
    if (refer_num == 0) {
        zhihan_release();
    }
}

void SyncedMemory::zhihan_release() {
    if (cpu_ptr_ && own_cpu_data_) {
        CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
    }
    
    cpu_ptr_ = NULL;
    own_cpu_data_ = false;
    
    head_ = UNINITIALIZED;
}

#endif


inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifdef USE_OPENCL
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    
    OPENCL_CHECK(clEnqueueReadBuffer(Caffe::Get().commandQueue, gpu_ptr_, CL_TRUE, 0, size_, cpu_ptr_, 0, NULL, NULL));

    // caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();

// #ifndef CPU_ONLY
#ifdef USE_OPENCL

  /*std::cout << "memory come to" << head_ << std::endl;

  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;

  to_print_callstack();
  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;
  std::cout << "------------------"<< std::endl;


    float* tmp = (float*)malloc(size_);*/


  switch (head_) {
  case UNINITIALIZED:
    
    gpu_ptr_ = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, size_, NULL, NULL);
    caffe_gpu_memset(size_, 0, (void *)gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;

  case HEAD_AT_CPU:

    // std::cout << "HEAD_AT_CPU is " << head_ << std::endl;


    if (gpu_ptr_ == NULL) {
      gpu_ptr_ = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, size_, NULL, NULL);
      own_gpu_data_ = true;
    }
    
    // std::cout << "very very interesting, the size is is " << size_ << std::endl;



    OPENCL_CHECK(clEnqueueWriteBuffer(Caffe::Get().commandQueue, gpu_ptr_, CL_TRUE, 0, size_, cpu_ptr_, 0, NULL, NULL));
    // caffe_gpu_memcpy(size_, cpu_ptr_, (void *)gpu_ptr_);
    

    // clEnqueueReadBuffer(Caffe::Get().commandQueue, gpu_ptr_, CL_TRUE, 0, size_, tmp, 0, NULL, NULL);

    // for (int i = 0; i < 100; ++i) {
    //   std::cout << tmp[i] << " ";
    // }


    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
// #endif
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifdef USE_OPENCL
  to_gpu();
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifdef USE_OPENCL
  // CHECK(data);
  if (own_gpu_data_) {
    clReleaseMemObject(gpu_ptr_);
  }
  gpu_ptr_ = (cl_mem) data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifdef USE_OPENCL
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
// void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
//   check_device();
//   CHECK(head_ == HEAD_AT_CPU);
//   if (gpu_ptr_ == NULL) {
//     CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//     own_gpu_data_ = true;
//   }
//   const cudaMemcpyKind put = cudaMemcpyHostToDevice;
//   CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
//   // Assume caller will synchronize on the stream before use
//   head_ = SYNCED;
// } TODOTODOO
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  // int device;
  // cudaGetDevice(&device);
  // CHECK(device == device_);
  // if (gpu_ptr_ && own_gpu_data_) {
  //   cudaPointerAttributes attributes;
  //   CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
  //   CHECK(attributes.device == device_);
  // } TODOTODOO
#endif
#endif
}

}  // namespace caffe

