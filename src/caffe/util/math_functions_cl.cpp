#ifdef USE_BOOST
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#else
#include <math.h>
#endif // USE_BOOST

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <clblast_c.h>

#ifdef WITH_HALF
#include "caffe/util/half.hpp"
#endif


#if defined(__APPLE__) && defined(__MACH__)
#include <vecLib.h>
#elif defined(USE_NEON_MATH) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  


  size_t lda = (TransA == CblasNoTrans) ? K : M;
  size_t ldb = (TransB == CblasNoTrans) ? N : K;
  size_t ldc = N;

  CLBlastTranspose_ blastTransA =
      (TransA == CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;
  CLBlastTranspose_ blastTransB =
      (TransB == CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;

#ifdef WITH_HALF

    half_b alpha_half = float2half_impl(alpha);
    half_b beta_half = float2half_impl(beta);

    CLBLAST_CHECK(CLBlastHgemm(CLBlastLayoutRowMajor,
                                        blastTransA, blastTransB,
                                        M, N, K,
                                        alpha_half,
                                        (cl_mem) A, 0, lda,
                                        (cl_mem) B, 0, ldb,
                                        beta_half,
                                        (cl_mem) C, 0, ldc,
                                        &Caffe::Get().commandQueue, NULL));


#else


  CLBLAST_CHECK(CLBlastSgemm(CLBlastLayoutRowMajor,
                                          blastTransA, blastTransB,
                                          M, N, K,
                                          alpha,
                                          (cl_mem) A, 0, lda,
                                          (cl_mem) B, 0, ldb,
                                          beta,
                                          (cl_mem) C, 0, ldc,
                                          &Caffe::Get().commandQueue, NULL));

#endif

}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y) {

    CLBlastTranspose_ blastTransA =
      (TransA != CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;


#ifdef WITH_HALF

    half_b alpha_half = float2half_impl(alpha);
    half_b beta_half = float2half_impl(beta);

    CLBLAST_CHECK(CLBlastHgemv(CLBlastLayoutColMajor, 
                                            blastTransA, 
                                            N, M,
                                            alpha_half,
                                            (cl_mem) A, 0, N,
                                            (cl_mem) x, 0, 1,
                                            beta_half,
                                            (cl_mem) y, 0, 1,
                                            &Caffe::Get().commandQueue, NULL));

#else

    CLBLAST_CHECK(CLBlastSgemv(CLBlastLayoutColMajor, 
                                            blastTransA, 
                                            N, M,
                                            alpha,
                                            (cl_mem) A, 0, N,
                                            (cl_mem) x, 0, 1,
                                            beta,
                                            (cl_mem) y, 0, 1,
                                            &Caffe::Get().commandQueue, NULL));
#endif
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
      
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "axpy_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Y));

#ifdef WITH_HALF
  half_b alpha_half = float2half_impl(alpha);
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_half), (void *)&alpha_half));
#else
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_float), (void *)&alpha));
#endif

  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}




template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {

#ifdef WITH_HALF
  OPENCL_CHECK(clEnqueueFillBuffer(Caffe::Get().commandQueue, (cl_mem) Y, &alpha, 2, 0, N * 2, 0, NULL, NULL));
#else
  OPENCL_CHECK(clEnqueueFillBuffer(Caffe::Get().commandQueue, (cl_mem) Y, &alpha, sizeof(Dtype), 0, N * sizeof(Dtype), 0, NULL, NULL));
#endif

}


template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);
template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);


template <>
void caffe_gpu_add_scalar<float>(const int N, const float alpha, float *X) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "add_scalar_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  

#ifdef WITH_HALF
  half_b alpha_half = float2half_impl(alpha);
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_half), (void *)&alpha_half));
#else
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_float), (void *)&alpha));
#endif

  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "scal_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  

#ifdef WITH_HALF
  half_b alpha_half = float2half_impl(alpha);
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_half), (void *)&alpha_half));
#else
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_float), (void *)&alpha));
#endif

  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);

  std::cout << "The N size is " << N << std::endl;
  std::cout << "The global size is " << global_size << std::endl;
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
 
}


template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}


template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b, float* y){
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "add_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b, float* y){
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "sub_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a, const float* b, float* y){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "mul_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void caffe_gpu_div<float>(const int N, const float* a, const float* b, float* y){
    
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "div_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void caffe_gpu_abs<float>(const int n, const float* a, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_exp<float>(const int n, const float* a, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_log<float>(const int n, const float* a, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_powx<float>(const int n, const float* a, const float b, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_sqrt<float>(const int n, const float* a, float* y){
      
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "sqrt_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y));  
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&n));  

  size_t global_size = CAFFE_GET_BLOCKS(n);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].


template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b, float* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_rng_gaussian<float>(const int n, const float mu, const float sigma,
                            float* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_rng_bernoulli<float>(const int n, const float p, int* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y, float* out){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y){
  NOT_IMPLEMENT;
}

template<>
void caffe_gpu_sign<float>(const int n, const float* x, float* y){
  NOT_IMPLEMENT;
}

template<>
void caffe_gpu_sgnbit<float>(const int n, const float* x, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_fabs<float>(const int n, const float* x, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x, float* y){
  NOT_IMPLEMENT;
}



template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const double alpha, const double* A, const double* x, const double beta,
    double* y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_add_scalar<double>(const int N, const double alpha, double *X) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a, const double* b, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_div<double>(const int N, const double* a, const double* b, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_abs<double>(const int n, const double* a, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_exp<double>(const int n, const double* a, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_log<double>(const int n, const double* a, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_powx<double>(const int n, const double* a, const double b, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_sqrt<double>(const int n, const double* a, double* y){
  NOT_IMPLEMENT;
}

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b, double* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_rng_gaussian<double>(const int n, const double mu, const double sigma,
                            double* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_rng_bernoulli<double>(const int n, const double p, int* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y, double* out){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y){
  NOT_IMPLEMENT;
}

template<>
void caffe_gpu_sign<double>(const int n, const double* x, double* y){
  NOT_IMPLEMENT;
}

template<>
void caffe_gpu_sgnbit<double>(const int n, const double* x, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_fabs<double>(const int n, const double* x, double* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x, double* y){
  NOT_IMPLEMENT;
}



void caffe_gpu_rng_uniform(const int n, unsigned int* r){
  NOT_IMPLEMENT;
}


void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {

  if (X != Y) {
    /*std::cout << "Copy memory" << std::endl;
    std::cout << "MEM X: " << x.get_ocl_mem() << std::endl;
    std::cout << "OFF X: " << x.get_ocl_off() << std::endl;
    std::cout << "MEM Y: " << y.get_ocl_mem() << std::endl;
    std::cout << "OFF Y: " << y.get_ocl_off() << std::endl;*/
    cl_int err = clEnqueueCopyBuffer(Caffe::Get().commandQueue, (cl_mem) X, (cl_mem) Y, 0, 0, N, 0,  NULL, NULL);
    // OCL_CHECK(err);
  }


}






}  // namespace caffe
