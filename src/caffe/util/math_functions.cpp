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


template<>
void caffe_cpu_gemm<half>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const half* A, const half* B, const float beta,
    half* C) {
  NOT_IMPLEMENT;
}


template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}


template <>
void caffe_cpu_gemv<half>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const half* A, const half* x,
    const float beta, half* y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}


template <>
void caffe_axpy<half>(const int N, const float alpha, const half* X,
    half* Y) { NOT_IMPLEMENT; }

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }


template <>
void caffe_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <>
void caffe_set(const int N, const float alpha, half* Y) {

  half alpha_half = float2half_impl(alpha);

  if (alpha == 0) {
    memset(Y, 0, sizeof(half) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha_half;
  }
}

#ifdef __VECLIB__
template <>
void caffe_set(const int N, const float alpha, float* Y) {
  vDSP_vfill(&alpha, Y, 1, N);
}


#elif defined(__ARM_NEON_H)
template <>
void caffe_set(const int N, const float alpha, float* Y) {
  int tail_frames = N % 4;
  const float* end = Y + N - tail_frames;
  while (Y < end) {
    float32x4_t alpha_dup = vld1q_dup_f32(&alpha);
    vst1q_f32(Y, alpha_dup);
    Y += 4;
  }
  for (int i = 0; i < tail_frames; ++i) {
    Y[i] = alpha;
  }
}
#endif


// template void caffe_set<half>(const int N, const float alpha, half* Y);
// template void caffe_set<float>(const int N, const float alpha, float* Y);

void caffe_set(const int N, const int alpha, int* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(int) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}


template <>
void caffe_add_scalar(const int N, const float alpha, half* Y) { NOT_IMPLEMENT; }

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
#ifdef __VECLIB__
  vDSP_vsadd(Y, 1, &alpha, Y, 1, N);
#elif defined(__ARM_NEON_H)
  int tail_frames = N % 4;
  const float* end = Y + N - tail_frames;
  while (Y < end) {
    float32x4_t a_frame = vld1q_f32(Y);
    float32x4_t alpha_dup = vld1q_dup_f32(&alpha);
    vst1q_f32(Y, vaddq_f32(a_frame, alpha_dup));
    Y += 4;
  }
  for (int i = 0; i < tail_frames; ++i) {
    Y[i] += alpha;
  }
#else
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
#endif
}



template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    //Different with CAFFE
    
    memcpy(Y, X, sizeof(Dtype) * N);
  }
}






template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);


template void caffe_copy<half>(const int N, const half* X, half* Y);

#if defined(__ARM_NEON_H)
template <>
void caffe_copy<float>(const int N, const float* X, float* Y) {
  int tail_frames = N % 4;
  const float* end = Y + N - tail_frames;
  while (Y < end) {
    float32x4_t x_frame = vld1q_f32(X);
    vst1q_f32(Y, x_frame);
    X += 4;
    Y += 4;
  }
  for (int i = 0; i < tail_frames; ++i) {
    Y[i] = X[i];
  }
}
#else
template void caffe_copy<float>(const int N, const float* X, float* Y);
#endif



template <>
void caffe_scal<half>(const int N, const float alpha, half *X) {
  NOT_IMPLEMENT;
}

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}




template <>
void caffe_cpu_axpby<half>(const int N, const float alpha, const half* X,
                            const float beta, half* Y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}



template <>
void caffe_add<half>(const int n, const half* a, const half* b,
    half* y) { NOT_IMPLEMENT; }

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vadd(a, 1, b, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vaddq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsAdd(tail_frames, a, b, y);
#else
  vsAdd(n, a, b, y);
#endif
}



template <>
void caffe_sub<half>(const int n, const half* a, const half* b,
    half* y) { NOT_IMPLEMENT; }


template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vsub(a, 1, b, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vsubq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsSub(tail_frames, a, b, y);
#else
  vsSub(n, a, b, y);
#endif
}



template <>
void caffe_mul<half>(const int n, const half* a, const half* b,
    half* y) { NOT_IMPLEMENT; }


template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vmul(a, 1, b, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vmulq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsMul(tail_frames, a, b, y);
#else
  vsMul(n, a, b, y);
#endif
}



template <>
void caffe_div<half>(const int n, const half* a, const half* b,
    half* y) { NOT_IMPLEMENT; }

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef __VECLIB__
  vDSP_vdiv(b, 1, a, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    float32x4_t b_frame = vld1q_f32(b);
    vst1q_f32(y, vdivq_f32(a_frame, b_frame));
    a += 4;
    b += 4;
    y += 4;
  }
  vsDiv(tail_frames, a, b, y);
#else
  vsDiv(n, a, b, y);
#endif
}



template <>
void caffe_powx<half>(const int n, const half* a, const float b,
    half* y) { NOT_IMPLEMENT; }

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}




template <>
void caffe_sqr<half>(const int n, const half* a, half* y) { NOT_IMPLEMENT; }

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vDSP_vsq(a, 1, y, 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    vst1q_f32(y, vmulq_f32(a_frame, a_frame));
    a += 4;
    y += 4;
  }
  vsSqr(tail_frames, a, y);
#else
  vsSqr(n, a, y);
#endif
}



template <>
void caffe_sqrt<half>(const int n, const half* a, half* y) { NOT_IMPLEMENT; }



template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vvsqrtf(y, a, &n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    vst1q_f32(y, vsqrtq_f32(a_frame));
    a += 4;
    y += 4;
  }
  vsSqrt(tail_frames, a, y);
#else
  vsSqrt(n, a, y);
#endif
}



template <>
void caffe_exp<half>(const int n, const half* a, half* y) { NOT_IMPLEMENT; }



template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vvexpf(y, a, &n);
#else
  vsExp(n, a, y);
#endif
}


template <>
void caffe_log<half>(const int n, const half* a, half* y) { NOT_IMPLEMENT; }


template <>
void caffe_log<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vvlogf(y, a, &n);
#else
  vsLn(n, a, y);
#endif
}



template <>
void caffe_abs<half>(const int n, const half* a, half* y) { NOT_IMPLEMENT; }


template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
#ifdef __VECLIB__
  vDSP_vabs(a, 1, y , 1, n);
#elif defined(__ARM_NEON_H)
  int tail_frames = n % 4;
  const float* end = y + n - tail_frames;
  while (y < end) {
    float32x4_t a_frame = vld1q_f32(a);
    vst1q_f32(y, vabsq_f32(a_frame));
    a += 4;
    y += 4;
  }
  vsAbs(tail_frames, a, y);
#else
  vsAbs(n, a, y);
#endif
}



unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

#ifdef USE_BOOST
template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);


#else
// std::nextafter has some problems with tr1 & _GLIBCXX_USE_C99_MATH_TR1
// when using android ndk

half caffe_nextafter(const half b) {
  NOT_IMPLEMENT;
}

float caffe_nextafter(const float b) {
    return ::nextafterf(b, std::numeric_limits<float>::max());
}

#endif

template <typename Dtype>
void caffe_rng_uniform(const int n, const float a, const float b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
#ifdef USE_BOOST
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
#else
  std::uniform_real_distribution<Dtype> random_distribution(a, caffe_nextafter(b));
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng());
  }
#endif
}

template <>
void caffe_rng_uniform<half>(const int n, const float a, const float b,
                              half* r) {
  float* convertor = (float*) malloc(n * sizeof(float));
  caffe_rng_uniform<float>(n, a, b, convertor);
  float2half(n, convertor, r);
  free(convertor);
  // NOT_IMPLEMENT;
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);


// template <typename Dtype>
// void caffe_rng_gaussian(const int n, const float a,
//                         const float sigma, Dtype* r) {
//   CHECK_GE(n, 0);
//   CHECK(r);
//   CHECK_GT(sigma, 0);
// #ifdef USE_BOOST
//   boost::normal_distribution<Dtype> random_distribution(a, sigma);
//   boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
//       variate_generator(caffe_rng(), random_distribution);
//   for (int i = 0; i < n; ++i) {
//     r[i] = variate_generator();
//   }
// #else
//   std::normal_distribution<Dtype> random_distribution(a, sigma);
//   for (int i = 0; i < n; ++i) {
//     r[i] = random_distribution(*caffe_rng());
//   }
// #endif
// }

template <>
void caffe_rng_gaussian<half>(const int n, const float a,
                               const float sigma, half* r) {
  
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);

  std::normal_distribution<float> random_distribution(a, sigma);
  for (int i = 0; i < n; ++i) {
    r[i] = float2half_impl(random_distribution(*caffe_rng()));
  }

}

template <>
void caffe_rng_gaussian<float>(const int n, const float a,
                               const float sigma, float* r) {

  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);


  std::normal_distribution<float> random_distribution(a, sigma);
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng());
  }

}


template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
#ifdef USE_BOOST 
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
#else
  std::bernoulli_distribution random_distribution(p);
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng());
  }
#endif
}

template <>
void caffe_rng_bernoulli<half>(const int n, const half p, int* r) {
  NOT_IMPLEMENT;
}


template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
#ifdef USE_BOOST
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
#else
  std::bernoulli_distribution random_distribution(p);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(random_distribution(*caffe_rng()));
  }
#endif
}

// template <>
// void caffe_rng_bernoulli<half>(const int n, const half x, unsigned int* r) {
  // NOT_IMPLEMENT;
// }


template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);


template <>
half caffe_cpu_strided_dot<half>(const int n, const half* x, const int incx,
    const half* y, const int incy) {
  NOT_IMPLEMENT;
}

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}


template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
half caffe_cpu_dot<half>(const int n, const half* x, const half* y);

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);



template <>
half caffe_cpu_asum<half>(const int n, const half* x) {
  NOT_IMPLEMENT;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
void caffe_cpu_scale<half>(const int n, const float alpha, const half *x,
                            half* y) {
  NOT_IMPLEMENT;
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

}  // namespace caffe
