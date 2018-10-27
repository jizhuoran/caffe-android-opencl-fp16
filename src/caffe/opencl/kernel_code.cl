#define OPENCL_KERNEL_LOOP(i, n) \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
       i < (n); \
       i += get_num_groups(0)*get_local_size(0))





__kernel void ReLUForward(__global float *in,
 						  __global float *out, 
 						  int N, float negative_slope) {
    OPENCL_KERNEL_LOOP(index, N) {
    	out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  	}
}

