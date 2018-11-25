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
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  NOT_IMPLEMENT;
}


template <>
void caffe_gpu_set<float>(const int N, const float alpha, float* Y) {
  NOT_IMPLEMENT;
}


template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a, const float* b, float* y){
  NOT_IMPLEMENT;
}

template <>
void caffe_gpu_div<float>(const int N, const float* a, const float* b, float* y){
  NOT_IMPLEMENT;
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
  NOT_IMPLEMENT;
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
void caffe_gpu_set<double>(const int N, const double alpha, double* Y) {
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
