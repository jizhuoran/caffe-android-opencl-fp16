#include "caffe/util/opencl_kernel.hpp"

namespace caffe {
  
std::string generate_opencl_math() {
	
	std::stringstream ss;

	ss << "#define OPENCL_KERNEL_LOOP(i, n) \\" << std::endl;
	ss << "for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\" << std::endl;
	ss << "i < (n); \\" << std::endl;
	ss << "i += get_num_groups(0)*get_local_size(0))" << std::endl;

	ss << std::endl;
	ss << std::endl;
	ss << std::endl;

	ss << "__kernel void ReLUForward(__global float *in," << std::endl;
	ss << "__global float *out," << std::endl;
	ss << "int N, float negative_slope) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;

  	return ss.str();
}

}  // namespace caffe
