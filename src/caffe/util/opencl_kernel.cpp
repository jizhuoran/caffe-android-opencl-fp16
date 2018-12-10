#include "caffe/util/opencl_kernel.hpp"

namespace caffe {
  

std::string generate_opencl_defs(bool is_half) {

	std::stringstream ss;

	if (is_half) {
		ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
		ss << std::endl;
		ss << "#define Dtype half" << std::endl;
		ss << "#define Dtype1 half" << std::endl;
		ss << "#define Dtype2 half2" << std::endl;
		ss << "#define Dtype4 half4" << std::endl;
		ss << "#define Dtype8 half8" << std::endl;
		ss << "#define Dtype16 half16" << std::endl;
	} else {
		ss << "#define Dtype float" << std::endl;
		ss << "#define Dtype1 float" << std::endl;
		ss << "#define Dtype2 float2" << std::endl;
		ss << "#define Dtype4 float4" << std::endl;
		ss << "#define Dtype8 float8" << std::endl;
		ss << "#define Dtype16 float16" << std::endl;
	}

	ss << std::endl;
	ss << std::endl;
	ss << std::endl;

  	return ss.str();
	
}






std::string generate_opencl_math() {
	
	std::stringstream ss;



	ss << "#define OPENCL_KERNEL_LOOP(i, n) \\" << std::endl;
	ss << "for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\" << std::endl;
	ss << "i < (n); \\" << std::endl;
	ss << "i += get_num_groups(0)*get_local_size(0))" << std::endl;

	ss << std::endl;
	ss << std::endl;
	ss << std::endl;


	ss << "__kernel void null_kernel_float(int alpha) {" << std::endl;
	ss << "int a = get_local_id(0);" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void ReLUForward(__global Dtype *in," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N, Dtype negative_slope) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void PReLUForward(__global Dtype *in, __global Dtype *slope_data," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N, int channels, int dim, int div_factor) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " int c = (index / dim) % channels / div_factor;" << std::endl;
	ss << " out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;



	ss << "__kernel void ELUForward(__global Dtype *in," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N, Dtype alpha) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "out[index] = in[index] > 0 ? in[index] : alpha * (exp(in[index]) - 1);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;

	ss << "__kernel void ScaleForward(__global Dtype *in," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N, __global Dtype *scale, int scale_dim, int inner_dim) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "const int scale_index = (index / inner_dim) % scale_dim;" << std::endl;
	ss << "out[index] = in[index] * scale[scale_index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void ScaleBiasForward(__global Dtype *in," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N, __global Dtype *scale, __global Dtype *bias, int scale_dim, int inner_dim) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "const int scale_index = (index / inner_dim) % scale_dim;" << std::endl;
	ss << "out[index] = in[index] * scale[scale_index] + bias[scale_index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void TanHForward(__global Dtype *in," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "out[index] = tanh(in[index]);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;

	ss << "__kernel void mul_kernel(__global Dtype *a, __global Dtype *b," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = a[index] * b[index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;

	ss << "__kernel void div_kernel(__global Dtype *a, __global Dtype *b," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = a[index] / b[index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void sqrt_kernel(__global Dtype *x," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = sqrt(x[index]);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void log_kernel(__global Dtype *a," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = log(a[index]);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void exp_kernel(__global Dtype *a," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = exp(a[index]);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;

	ss << "__kernel void abs_kernel(__global Dtype *a," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = fabs(a[index]);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;




	ss << "__kernel void powx_kernel(__global Dtype *a," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "Dtype alpha, int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = pow(a[index], alpha);" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;



	ss << "__kernel void sub_kernel(__global Dtype *a, __global Dtype *b," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = a[index] - b[index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;

	ss << "__kernel void add_kernel(__global Dtype *a, __global Dtype *b," << std::endl;
	ss << "__global Dtype *y," << std::endl;
	ss << "int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << " y[index] = a[index] + b[index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;



	ss << "__kernel void MaxForward(__global Dtype *bottom_data_a, __global Dtype *bottom_data_b," << std::endl;
	ss << "__global Dtype *top_data, __global int *mask," << std::endl;
	ss << "int nthreads, int blob_idx) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, nthreads) {" << std::endl;
	ss << "Dtype maxval = -FLT_MAX;" << std::endl;
	ss << "int maxidx = -1;" << std::endl;
	ss << "if (bottom_data_a[index] > bottom_data_b[index]) {" << std::endl;
	ss << "if (blob_idx == 0) {" << std::endl;
	ss << "maxval = bottom_data_a[index];" << std::endl;
	ss << "top_data[index] = maxval;" << std::endl;
	ss << "maxidx = blob_idx;" << std::endl;
	ss << "mask[index] = maxidx;" << std::endl;
	ss << "}" << std::endl;
	ss << "} else {" << std::endl;
	ss << "maxval = bottom_data_a[index];" << std::endl;
	ss << "top_data[index] = maxval;" << std::endl;
	ss << "maxidx = blob_idx + 1;" << std::endl;
	ss << "mask[index] = maxidx;" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void axpy_kernel(__global Dtype *X," << std::endl;
	ss << "__global Dtype *Y," << std::endl;
	ss << "Dtype alpha, int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "Y[index] = X[index] * alpha + Y[index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void scal_kernel(__global Dtype *X," << std::endl;
	ss << "Dtype alpha, int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "X[index] *= alpha;" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void add_scalar_kernel(__global Dtype *y," << std::endl;
	ss << "Dtype alpha, int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "y[index] += alpha;" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void BiasForward(__global Dtype *in, __global Dtype *bias," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int bias_dim, int inner_dim, int N) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "const int bias_index = (index / inner_dim) % bias_dim;" << std::endl;
	ss << "out[index] = in[index] + bias[bias_index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "int compute_uncropped_index(int index, const int ndims," << std::endl;
	ss << "__global int *src_strides, __global int *dest_strides," << std::endl;
	ss << "__global int *offsets) {" << std::endl;
	ss << "int dest_index = index;" << std::endl;
	ss << "int src_index = 0;" << std::endl;
	ss << "for (int i = 0; i < ndims; ++i) {" << std::endl;
	ss << "int coord = dest_index / dest_strides[i];" << std::endl;
	ss << "dest_index -= coord * dest_strides[i];" << std::endl;
	ss << "src_index += src_strides[i] * (coord + offsets[i]);" << std::endl;
	ss << "}" << std::endl;
	ss << "return src_index;" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void crop_kernel_forward(__global Dtype *src," << std::endl;
	ss << "__global Dtype *dest," << std::endl;
	ss << "__global int *src_strides," << std::endl;
	ss << "__global int *dest_strides," << std::endl;
	ss << "__global int *offsets," << std::endl;
	ss << "int ndims, int nthreads) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, nthreads) {" << std::endl;
	ss << " int src_index = compute_uncropped_index(index, ndims, src_strides, dest_strides, offsets);" << std::endl;
	ss << " dest[index] = src[src_index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel void Concat(__global Dtype *in_data," << std::endl;
	ss << "__global Dtype *out_data," << std::endl;
	ss << "const int concat_size," << std::endl;
	ss << "const int top_concat_axis, const int bottom_concat_axis," << std::endl;
	ss << "const int offset_concat_axis, int nthreads) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, nthreads) {" << std::endl;
	ss << " const int total_concat_size = concat_size * bottom_concat_axis;" << std::endl;
	ss << " const int concat_num = index / total_concat_size;" << std::endl;
	ss << " const int concat_index = index % total_concat_size;" << std::endl;
	ss << " const int top_index = concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;" << std::endl;
	ss << " out_data[top_index] = in_data[index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;





	ss << "__kernel void Slice(__global Dtype *in_data," << std::endl;
	ss << "__global Dtype *out_data," << std::endl;
	ss << "const int slice_size," << std::endl;
	ss << "const int bottom_slice_axis, const int top_slice_axis," << std::endl;
	ss << "const int offset_slice_axis, int nthreads) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, nthreads) {" << std::endl;
	ss << " const int total_slice_size = slice_size * top_slice_axis;" << std::endl;
	ss << " const int slice_num = index / total_slice_size;" << std::endl;
	ss << " const int slice_index = index % total_slice_size;" << std::endl;
	ss << " const int bottom_index = slice_index + " << std::endl;
	ss << "(slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;" << std::endl;
	ss << " out_data[index] = in_data[bottom_index];" << std::endl;
	ss << "}" << std::endl;
	ss << "}" << std::endl;



	ss << "__kernel void MaxPoolForward(int nthreads, " << std::endl;
	ss << "    __global Dtype* bottom_data, int num, int channels, " << std::endl;
	ss << "    int height, int width, int pooled_height, " << std::endl;
	ss << "    int pooled_width, int kernel_h, int kernel_w, " << std::endl;
	ss << "    int stride_h, int stride_w, __global Dtype* top_data, int pad_h, int pad_w, " << std::endl;
	ss << "    __global int* mask, __global Dtype* top_mask) { " << std::endl;
	ss << "  OPENCL_KERNEL_LOOP(index, nthreads) { " << std::endl;
	ss << "    const int pw = index % pooled_width; " << std::endl;
	ss << "    const int ph = (index / pooled_width) % pooled_height; " << std::endl;
	ss << "    const int c = (index / pooled_width / pooled_height) % channels; " << std::endl;
	ss << "    const int n = index / pooled_width / pooled_height / channels; " << std::endl;
	ss << "    int hstart = ph * stride_h - pad_h; " << std::endl;
	ss << "    int wstart = pw * stride_w - pad_w; " << std::endl;
	ss << "    const int hend = min(hstart + kernel_h, height); " << std::endl;
	ss << "    const int wend = min(wstart + kernel_w, width); " << std::endl;
	ss << "    hstart = max(hstart, 0); " << std::endl;
	ss << "    wstart = max(wstart, 0); " << std::endl;
	ss << "    Dtype maxval = -FLT_MAX; " << std::endl;
	ss << "    int maxidx = -1; " << std::endl;
	ss << "    __global Dtype* bottom_slice = " << std::endl;
	ss << "        bottom_data + (n * channels + c) * height * width; " << std::endl;
	ss << "    for (int h = hstart; h < hend; ++h) { " << std::endl;
	ss << "      for (int w = wstart; w < wend; ++w) { " << std::endl;
	ss << "        if (bottom_slice[h * width + w] > maxval) { " << std::endl;
	ss << "          maxidx = h * width + w; " << std::endl;
	ss << "          maxval = bottom_slice[maxidx]; " << std::endl;
	ss << "        } " << std::endl;
	ss << "      } " << std::endl;
	ss << "    } " << std::endl;
	ss << "    top_data[index] = maxval; " << std::endl;
	ss << "    if (mask) { " << std::endl;
	ss << "      mask[index] = maxidx; " << std::endl;
	ss << "    } else { " << std::endl;
	ss << "      top_mask[index] = maxidx; " << std::endl;
	ss << "    } " << std::endl;
	ss << "  } " << std::endl;
	ss << "} " << std::endl;

	ss << "__kernel void AvePoolForward(const int nthreads, " << std::endl;
	ss << "    __global Dtype* bottom_data, const int num, const int channels, " << std::endl;
	ss << "    const int height, const int width, const int pooled_height, " << std::endl;
	ss << "    const int pooled_width, const int kernel_h, const int kernel_w, " << std::endl;
	ss << "    const int stride_h, const int stride_w, __global Dtype* top_data, const int pad_h, const int pad_w) {" << std::endl;
	ss << "  OPENCL_KERNEL_LOOP(index, nthreads) { " << std::endl;
	ss << "    const int pw = index % pooled_width; " << std::endl;
	ss << "    const int ph = (index / pooled_width) % pooled_height; " << std::endl;
	ss << "    const int c = (index / pooled_width / pooled_height) % channels; " << std::endl;
	ss << "    const int n = index / pooled_width / pooled_height / channels; " << std::endl;
	ss << "    int hstart = ph * stride_h - pad_h; " << std::endl;
	ss << "    int wstart = pw * stride_w - pad_w; " << std::endl;
	ss << "    int hend = min(hstart + kernel_h, height + pad_h); " << std::endl;
	ss << "    int wend = min(wstart + kernel_w, width + pad_w); " << std::endl;
	ss << "    const int pool_size = (hend - hstart) * (wend - wstart); " << std::endl;
	ss << "    hstart = max(hstart, 0); " << std::endl;
	ss << "    wstart = max(wstart, 0); " << std::endl;
	ss << "    hend = min(hend, height); " << std::endl;
	ss << "    wend = min(wend, width); " << std::endl;
	ss << "    Dtype aveval = 0; " << std::endl;
	ss << "    __global Dtype* const bottom_slice = " << std::endl;
	ss << "        bottom_data + (n * channels + c) * height * width; " << std::endl;
	ss << "    for (int h = hstart; h < hend; ++h) { " << std::endl;
	ss << "      for (int w = wstart; w < wend; ++w) { " << std::endl;
	ss << "        aveval += bottom_slice[h * width + w]; " << std::endl;
	ss << "      } " << std::endl;
	ss << "    } " << std::endl;
	ss << "    top_data[index] = aveval / pool_size; " << std::endl;
	ss << "  } " << std::endl;
	ss << "} " << std::endl;


	ss << "__kernel void StoPoolForwardTest(const int nthreads, " << std::endl;
	ss << "    __global Dtype* bottom_data, " << std::endl;
	ss << "    const int num, const int channels, const int height, " << std::endl;
	ss << "    const int width, const int pooled_height, const int pooled_width, " << std::endl;
	ss << "    const int kernel_h, const int kernel_w, const int stride_h, " << std::endl;
	ss << "    const int stride_w, __global Dtype* top_data) { " << std::endl;
	ss << "    OPENCL_KERNEL_LOOP(index, nthreads) { " << std::endl;
	ss << "    const int pw = index % pooled_width; " << std::endl;
	ss << "    const int ph = (index / pooled_width) % pooled_height; " << std::endl;
	ss << "    const int c = (index / pooled_width / pooled_height) % channels; " << std::endl;
	ss << "    const int n = index / pooled_width / pooled_height / channels; " << std::endl;
	ss << "    const int hstart = ph * stride_h; " << std::endl;
	ss << "    const int hend = min(hstart + kernel_h, height); " << std::endl;
	ss << "    const int wstart = pw * stride_w; " << std::endl;
	ss << "    const int wend = min(wstart + kernel_w, width); " << std::endl;
	ss << "    // We set cumsum to be 0 to avoid divide-by-zero problems " << std::endl;
	ss << "    Dtype cumsum = 0.; " << std::endl;
	ss << "    Dtype cumvalues = 0.; " << std::endl;
	ss << "    __global Dtype* const bottom_slice = " << std::endl;
	ss << "        bottom_data + (n * channels + c) * height * width; " << std::endl;
	ss << "    // First pass: get sum " << std::endl;
	ss << "    for (int h = hstart; h < hend; ++h) { " << std::endl;
	ss << "      for (int w = wstart; w < wend; ++w) { " << std::endl;
	ss << "        cumsum += bottom_slice[h * width + w]; " << std::endl;
	ss << "        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w]; " << std::endl;
	ss << "      } " << std::endl;
	ss << "    } " << std::endl;
	ss << "    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.; " << std::endl;
	ss << "  } " << std::endl;
	ss << "} " << std::endl;












	// ss << "#ifndef WGS1" << std::endl;
	ss << "  #define WGS1 64" << std::endl;
	// ss << "#endif" << std::endl;
	// ss << "#ifndef WGS2" << std::endl;
	ss << "  #define WGS2 64" << std::endl;
	// ss << "#endif" << std::endl;

	ss << "__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))" << std::endl;
	ss << "void Xasum(const int n," << std::endl;
	ss << "           const __global Dtype* restrict xgm, const int x_inc," << std::endl;
	ss << "           __global Dtype* output, Dtype alpha) {" << std::endl;
	      
	ss << "  __local Dtype lm[WGS1];" << std::endl;
	ss << "  const int lid = get_local_id(0);" << std::endl;
	ss << "  const int wgid = get_group_id(0);" << std::endl;
	ss << "  const int num_groups = get_num_groups(0);" << std::endl;

	ss << "  // Performs loading and the first steps of the reduction" << std::endl;
	ss << "  Dtype acc = 0;" << std::endl;

	ss << "  int id = wgid*WGS1 + lid;" << std::endl;

	ss << "  while (id*x_inc < n) {" << std::endl;
	ss << "    Dtype x = xgm[id*x_inc + get_group_id(1) * n];" << std::endl;
	ss << "    acc += x * alpha;" << std::endl;
	ss << "    id += WGS1*num_groups;" << std::endl;
	ss << "  }" << std::endl;
	ss << "  lm[lid] = acc * alpha;" << std::endl;
	ss << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	ss << "  // Performs reduction in local memory" << std::endl;
	ss << "  for (int s=WGS1/2; s>0; s=s>>1) {" << std::endl;
	ss << "    if (lid < s) {" << std::endl;
	ss << "      lm[lid] += lm[lid + s];" << std::endl;
	ss << "    }" << std::endl;
	ss << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
	ss << "  }" << std::endl;

	ss << "  // Stores the per-workgroup result" << std::endl;
	ss << "  if (lid == 0) {" << std::endl;
	ss << "    output[wgid + get_group_id(1) * num_groups] = lm[0];" << std::endl;
	ss << "  }" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))" << std::endl;
	ss << "void XasumEpilogue(const __global Dtype* restrict input," << std::endl;
	ss << "                   __global Dtype* asum, Dtype beta) {" << std::endl;
	      
	ss << "  __local Dtype lm[WGS2];" << std::endl;
	ss << "  const int lid = get_local_id(0);" << std::endl;

	ss << "  // Performs the first step of the reduction while loading the data" << std::endl;
	ss << "  lm[lid] = (input[get_group_id(1) * WGS2 * 2 + lid] + input[get_group_id(1) * WGS2 * 2 + lid + WGS2]) * beta;" << std::endl;
	ss << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	ss << "  // Performs reduction in local memory" << std::endl;
	ss << "  for (int s=WGS2/2; s>0; s=s>>1) {" << std::endl;
	ss << "    if (lid < s) {" << std::endl;
	ss << "      lm[lid] += lm[lid + s];" << std::endl;
	ss << "    }" << std::endl;
	ss << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
	ss << "  }" << std::endl;

	ss << "  // Computes the absolute value and stores the final result" << std::endl;
	ss << "  if (lid == 0) {" << std::endl;
	ss << "    asum[get_group_id(1)] = lm[0];" << std::endl;
	ss << "  }" << std::endl;
	ss << "}" << std::endl;




  	return ss.str();
}

}  // namespace caffe
