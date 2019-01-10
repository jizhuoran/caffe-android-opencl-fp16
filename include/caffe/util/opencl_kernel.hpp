#ifndef CAFFE_UTIL_OPENCL_KERNEL_H_
#define CAFFE_UTIL_OPENCL_KERNEL_H_

#include <iostream>
#include <string.h>
#include <sstream>

namespace caffe {
  	std::string generate_opencl_defs(bool is_half);
  	std::string generate_opencl_math(bool is_half);
}  // namespace caffe

#endif   // CAFFE_UTIL_OPENCL_KERNEL_H_
