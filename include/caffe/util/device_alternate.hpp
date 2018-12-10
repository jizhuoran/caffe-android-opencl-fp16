#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_
#include <cxxabi.h>

#ifdef USE_OPENCL


#ifdef __ANDROID__



#define OPENCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if(error != CL_SUCCESS) { \
      LOG(ERROR) << "This is a error for OpenCL " << error; \
      exit(0); \
    } \
  } while (0)


#define CLBLAST_CHECK(condition) \
  do { \
    CLBlastStatusCode status = condition; \
    if(status != CLBlastSuccess) { \
      LOG(ERROR) << "This is a error for CLBlast " << status; \
      exit(0); \
    } \
  } while (0)

#else
#include <execinfo.h>

#define OPENCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if(error != CL_SUCCESS) { \
      std::cerr << "This is a error for OpenCL "<< error << " in " << __LINE__ << " in " << __FILE__ << std::endl;\
      void *buffer[100];\
      int n = backtrace(buffer,10);\
      char **str = backtrace_symbols(buffer, n);\
      for (int i = 0; i < n; i++) {printf("%d:  %s\n", i, str[i]);}\
      exit(0); \
    } \
  } while (0)




#define CLBLAST_CHECK(condition) \
  do { \
    CLBlastStatusCode status = condition; \
    if(status != CL_SUCCESS) { \
      std::cerr << "This is a error for CLBlast "<< status << " in " << __LINE__ << " in " << __FILE__ << std::endl;\
      void *buffer[100];\
      int n = backtrace(buffer,10);\
      char **str = backtrace_symbols(buffer, n);\
      for (int i = 0; i < n; i++) {printf("%d:  %s\n", i, str[i]);}\
      exit(0); \
    } \
  } while (0)

#define CLBUILD_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if (ret != CL_SUCCESS) { \
      char *buff_erro; \
      cl_int errcode; \
      size_t build_log_len; \
      errcode = clGetProgramBuildInfo(program, Caffe::Get().deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len); \
      if (errcode) { \
        LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__; \
        exit(-1); \
      } \
      buff_erro = (char *)malloc(build_log_len); \
      if (!buff_erro) { \
          printf("malloc failed at line %d\n", __LINE__); \
          exit(-2); \
      } \
      errcode = clGetProgramBuildInfo(program, Caffe::Get().deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL); \
      if (errcode) { \
          LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__; \
          exit(-3); \
      } \
      LOG(ERROR) << "Build log: " << buff_erro; \
      free(buff_erro); \
      LOG(ERROR) << "clBuildProgram failed"; \
      exit(EXIT_FAILURE); \
    } \
  } while (0)



  
#endif


#endif

#define NOT_IMPLEMENT LOG(FATAL) << "This function has not been implemented yet!"



#ifdef CPU_ONLY  // CPU-only Caffe.

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.

// #include <cublas_v2.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <curand.h>
// #include <driver_types.h>  // cuda driver types
// #ifdef USE_CUDNN  // cuDNN acceleration library.
// #include "caffe/util/cudnn.hpp"
// #endif

//
// CUDA macros
//


#define TEMP_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NOT_IMPLEMENT; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NOT_IMPLEMENT; } \



// #define TEMP_GPU(classname) \
// template <typename Dtype> \
// void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
//     const vector<Blob<Dtype>*>& top) { Forward_cpu(bottom, top); } \
// template <typename Dtype> \
// void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
//     const vector<bool>& propagate_down, \
//     const vector<Blob<Dtype>*>& bottom) { Backward_cpu(top, propagate_down, bottom); } \

#define TEMP_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { funcname##_##cpu(bottom, top); } \

#define TEMP_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { funcname##_##cpu(top, propagate_down, bottom); } \


/*
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
*/
namespace caffe {

// CUDA: library error reporting.
// const char* cublasGetErrorString(cublasStatus_t error);
// const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const size_t CAFFE_CUDA_NUM_THREADS = 128;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS * CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe

#endif  // CPU_ONLY

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
