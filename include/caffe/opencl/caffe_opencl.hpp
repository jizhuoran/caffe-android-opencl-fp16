#ifndef CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_
#define CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_

// #ifdef CMAKE_BUILD
// #include "caffe_config.h"
// #endif

// #include <vector>

#ifdef USE_OPENCL

#ifndef __APPLE__
#include "CL/cl.h"
#else
#include "OpenCL/cl.h"
#endif

// #include "viennacl/backend/opencl.hpp"
// #include "viennacl/ocl/backend.hpp"
// #include "viennacl/ocl/context.hpp"
// #include "viennacl/ocl/device.hpp"
// #include "viennacl/ocl/platform.hpp"
// #include "viennacl/vector.hpp"

namespace caffe {

#ifndef OPENCL_QUEUE_COUNT
#define OPENCL_QUEUE_COUNT 8
#endif  // OPENCL_QUEUE_COUNT


const char* clGetErrorString(cl_int error);

#define OCL_CHECK(condition) \
  do { \
    cl_int error = (condition); \
    CHECK_EQ(error, CL_SUCCESS) << " " << clGetErrorString(error); \
  } while (0)

#define OCL_CHECK_MESSAGE(condition, message) \
  do { \
    cl_int error = (condition); \
    CHECK_EQ(error, CL_SUCCESS) << " " << clGetErrorString(error) \
                                       << " (" << message << ")"; \
  } while (0)


}  // namespace caffe

#endif  // USE_OPENCL

#endif  // CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_
