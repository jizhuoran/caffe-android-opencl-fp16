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
void caffe_gpu_set(const int N, const float alpha, float* Y) {

  std::cout << "zhegehaoyongma1" << std::endl;
  std::cout << "zhegehaoyongma1" << std::endl;
  std::cout << "zhegehaoyongma1" << std::endl;
  std::cout << "zhegehaoyongma1" << std::endl;


}


void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {

  std::cout << "zhegehaoyongma" << std::endl;
  std::cout << "zhegehaoyongma" << std::endl;
  std::cout << "zhegehaoyongma" << std::endl;
  std::cout << "zhegehaoyongma" << std::endl;
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
