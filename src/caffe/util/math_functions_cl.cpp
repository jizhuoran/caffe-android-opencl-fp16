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
void caffe_gpu_gemm<half>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const half* A, const half* B, const float beta,
    half* C) {
  
  size_t lda = (TransA == CblasNoTrans) ? K : M;
  size_t ldb = (TransB == CblasNoTrans) ? N : K;
  size_t ldc = N;

  CLBlastTranspose_ blastTransA =
      (TransA == CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;
  CLBlastTranspose_ blastTransB =
      (TransB == CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;


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

}


template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y) {

    CLBlastTranspose_ blastTransA =
      (TransA != CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;

    CLBLAST_CHECK(CLBlastSgemv(CLBlastLayoutColMajor, 
                                            blastTransA, 
                                            N, M,
                                            alpha,
                                            (cl_mem) A, 0, N,
                                            (cl_mem) x, 0, 1,
                                            beta,
                                            (cl_mem) y, 0, 1,
                                            &Caffe::Get().commandQueue, NULL));
}

template <>
void caffe_gpu_gemv<half>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const half* A, const half* x, const float beta,
    half* y) {
      CLBlastTranspose_ blastTransA =
      (TransA != CblasNoTrans) ? CLBlastTransposeNo : CLBlastTransposeYes;



    half_b alpha_half = float2half_impl(alpha);
    half_b beta_half = float2half_impl(beta);

    CLBLAST_CHECK(CLBlastHgemv(CLBlastLayoutColMajor, 
                                            blastTransA, 
                                            N, M,
                                            (cl_half) alpha_half,
                                            (cl_mem) A, 0, N,
                                            (cl_mem) x, 0, 1,
                                            (cl_half) beta_half,
                                            (cl_mem) y, 0, 1,
                                            &Caffe::Get().commandQueue, NULL));

}

template <>
void caffe_gpu_bsum<float>(const int m, const int n, const float* X, const float alpha, const float beta,
                            float* y, const int x_inc) {

  cl_int ret;

  cl_kernel kernel1 = clCreateKernel(Caffe::Get().program, "Xasum", &ret);
  OPENCL_CHECK(ret);
  cl_kernel kernel2 = clCreateKernel(Caffe::Get().program, "XasumEpilogue", &ret);
  OPENCL_CHECK(ret);


  size_t temp_size = 2*64;

  cl_mem temp_buffer = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, m * temp_size * sizeof(half), NULL, NULL);

  OPENCL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_int), (void *)&n));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_int), (void *)&x_inc));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&temp_buffer));
  OPENCL_CHECK(clSetKernelArg(kernel1, 4, sizeof(cl_float), (void *)&alpha));  



  size_t* local_size = new size_t[2];
  local_size[0] = static_cast<size_t>(64);
  local_size[1] = static_cast<size_t>(1);

  size_t* global_size = new size_t[2];
  global_size[0] = static_cast<size_t>(temp_size * 64);
  global_size[1] = static_cast<size_t>(m);



  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel1, 2, NULL, global_size, local_size, 0, NULL, NULL));  


  OPENCL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&temp_buffer));  
  OPENCL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&y));
  OPENCL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_float), (void *)&beta));

  global_size[0] = static_cast<size_t>(64);

  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel2, 2, NULL, global_size, local_size, 0, NULL, NULL));  
 
  // OPENCL_CHECK(clReleaseMemObject(temp_buffer));


}


template <>
void caffe_gpu_bsum<half>(const int m, const int n, const half* X, const float alpha,  const float beta,
                            half* y, const int x_inc) {
  
  cl_int ret;
  cl_kernel kernel1 = clCreateKernel(Caffe::Get().program, "Xasum", &ret);
  OPENCL_CHECK(ret);
  cl_kernel kernel2 = clCreateKernel(Caffe::Get().program, "XasumEpilogue", &ret);
  OPENCL_CHECK(ret);

  half_b alpha_half = float2half_impl(alpha);


  size_t temp_size = 2*64;

  cl_mem temp_buffer = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, m * temp_size * sizeof(half), NULL, NULL);

  OPENCL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_int), (void *)&n));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_int), (void *)&x_inc));  
  OPENCL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&temp_buffer));
  OPENCL_CHECK(clSetKernelArg(kernel1, 4, sizeof(cl_half), (void *)&alpha_half));  


  size_t* local_size = new size_t[2];
  local_size[0] = static_cast<size_t>(64);
  local_size[1] = static_cast<size_t>(1);

  size_t* global_size = new size_t[2];
  global_size[0] = static_cast<size_t>(temp_size * 64);
  global_size[1] = static_cast<size_t>(m);



  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel1, 2, NULL, global_size, local_size, 0, NULL, NULL));  


  half_b beta_half = float2half_impl(beta);
  OPENCL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&temp_buffer));  
  OPENCL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&y));
  OPENCL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_half), (void *)&beta_half));

  global_size[0] = static_cast<size_t>(64);

  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel2, 2, NULL, global_size, local_size, 0, NULL, NULL));  
 

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
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_float), (void *)&alpha));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}

template <>
void caffe_gpu_axpy<half>(const int N, const float alpha, const half* X,
    half* Y) {
  
  cl_int ret;
  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "axpy_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  half alpha_half = float2half_impl(alpha);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Y));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_half), (void *)&alpha_half));
  OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}


template <>
void caffe_gpu_set<float>(const int N, const float alpha, float* Y) {
  OPENCL_CHECK(clEnqueueFillBuffer(Caffe::Get().commandQueue, (cl_mem) Y, &alpha, sizeof(float), 0, N * sizeof(float), 0, NULL, NULL));
}

template <>
void caffe_gpu_set<half>(const int N, const float alpha, half* Y) {

  half_b alpha_half = float2half_impl(alpha);
  OPENCL_CHECK(clEnqueueFillBuffer(Caffe::Get().commandQueue, (cl_mem) Y, &alpha_half, sizeof(half), 0, N * sizeof(half), 0, NULL, NULL));

}

void caffe_gpu_set(const int N, const int alpha, int *Y) {
  OPENCL_CHECK(clEnqueueFillBuffer(Caffe::Get().commandQueue, (cl_mem) Y, &alpha, sizeof(int), 0, N * sizeof(int), 0, NULL, NULL));
}


template <>
void caffe_gpu_add_scalar<float>(const int N, const float alpha, float *X) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "add_scalar_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_float), (void *)&alpha));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);
  
  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
  
}


template <>
void caffe_gpu_add_scalar<half>(const int N, const float alpha, half *X) {
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "add_scalar_kernel", &ret);
  OPENCL_CHECK(ret);

  half_b alpha_half = float2half_impl(alpha);

  // Set arguments for kernel
  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_half), (void *)&alpha_half));
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
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_float), (void *)&alpha));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);

  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
 
}


template <>
void caffe_gpu_scal<half>(const int N, const float alpha, half* X){
  
  cl_int ret;

  cl_kernel kernel = clCreateKernel(Caffe::Get().program, "scal_kernel", &ret);
  OPENCL_CHECK(ret);

  // Set arguments for kernel
  half_b alpha_half = float2half_impl(alpha);

  OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&X));  
  OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_half), (void *)&alpha_half));
  OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&N));  

  size_t global_size = CAFFE_GET_BLOCKS(N);

  OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &CAFFE_CUDA_NUM_THREADS, 0, NULL, NULL));  
 
}



template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<half>(const int N, const float alpha, const half* X,
    const float beta, half* Y) {
  caffe_gpu_scal<half>(N, beta, Y);
  caffe_gpu_axpy<half>(N, alpha, X, Y);
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
void caffe_gpu_add<half>(const int N, const half* a, const half* b, half* y){
  
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
void caffe_gpu_sub<half>(const int N, const half* a, const half* b, half* y){
  
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
void caffe_gpu_mul<half>(const int N, const half* a, const half* b, half* y){
  
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
void caffe_gpu_div<half>(const int N, const half* a, const half* b, half* y){
  
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
void caffe_gpu_abs<half>(const int n, const half* a, half* y){
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
void caffe_gpu_exp<half>(const int n, const half* a, half* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_log<half>(const int n, const half* a, half* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_powx<half>(const int n, const half* a, const float b, half* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_sqrt<half>(const int n, const half* a, half* y){
  NOT_IMPLEMENT;
}

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].

template <>
void caffe_gpu_rng_uniform<half>(const int n, const half a, const half b, half* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_rng_gaussian<half>(const int n, const half mu, const half sigma,
                            half* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_rng_bernoulli<half>(const int n, const half p, int* r){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_dot<half>(const int n, const half* x, const half* y, half* out){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_asum<half>(const int n, const half* x, half* y){
  NOT_IMPLEMENT;
}

template<>
void caffe_gpu_sign<half>(const int n, const half* x, half* y){
  NOT_IMPLEMENT;
}

template<>
void caffe_gpu_sgnbit<half>(const int n, const half* x, half* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_fabs<half>(const int n, const half* x, half* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_scale<half>(const int n, const float alpha, const half *x, half* y){
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


template <typename Dtype>
void caffe_cl_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    OPENCL_CHECK(clEnqueueCopyBuffer(Caffe::Get().commandQueue, (cl_mem) X, (cl_mem) Y, 0, 0, sizeof(Dtype) * N, 0, NULL, NULL));
  }
}


template void caffe_cl_copy<int>(const int N, const int* X, int* Y);
template void caffe_cl_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_cl_copy<half>(const int N, const half* X, half* Y);
template void caffe_cl_copy<float>(const int N, const float* X, float* Y);





}  // namespace caffe
