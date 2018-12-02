#ifndef CAFFE_UTIL_OPENCL_KERNEL_H_
#define CAFFE_UTIL_OPENCL_KERNEL_H_

#include <iostream>
#include <string.h>
#include <sstream>

namespace caffe {
  
  	std::string generate_opencl_math();
	std::string general_gemm_kernel();
}  // namespace caffe

#endif   // CAFFE_UTIL_OPENCL_KERNEL_H_
