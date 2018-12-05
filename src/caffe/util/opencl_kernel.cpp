#include "caffe/util/opencl_kernel.hpp"

namespace caffe {
  
std::string generate_opencl_math() {
	
	std::stringstream ss;

#ifdef WITH_HALF		
	ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
	ss << std::endl;
	ss << "#define Dtype half" << std::endl;
	ss << "#define Dtype1 half" << std::endl;
	ss << "#define Dtype2 half2" << std::endl;
	ss << "#define Dtype4 half4" << std::endl;
	ss << "#define Dtype8 half8" << std::endl;
	ss << "#define Dtype16 half16" << std::endl;
#else
	ss << "#define Dtype float" << std::endl;
	ss << "#define Dtype1 float" << std::endl;
	ss << "#define Dtype2 float2" << std::endl;
	ss << "#define Dtype4 float4" << std::endl;
	ss << "#define Dtype8 float8" << std::endl;
	ss << "#define Dtype16 float16" << std::endl;
#endif

	

	ss << std::endl;
	ss << std::endl;
	ss << std::endl;

	ss << "#define OPENCL_KERNEL_LOOP(i, n) \\" << std::endl;
	ss << "for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\" << std::endl;
	ss << "i < (n); \\" << std::endl;
	ss << "i += get_num_groups(0)*get_local_size(0))" << std::endl;

	ss << std::endl;
	ss << std::endl;
	ss << std::endl;

	ss << "__kernel void ReLUForward(__global Dtype *in," << std::endl;
	ss << "__global Dtype *out," << std::endl;
	ss << "int N, Dtype negative_slope) {" << std::endl;
	ss << "OPENCL_KERNEL_LOOP(index, N) {" << std::endl;
	ss << "out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;" << std::endl;
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
	      
	ss << "  __local float lm[WGS1];" << std::endl;
	ss << "  const int lid = get_local_id(0);" << std::endl;
	ss << "  const int wgid = get_group_id(0);" << std::endl;
	ss << "  const int num_groups = get_num_groups(0);" << std::endl;

	ss << "  // Performs loading and the first steps of the reduction" << std::endl;
	ss << "  float acc = 0;" << std::endl;

	ss << "  int id = wgid*WGS1 + lid;" << std::endl;

	ss << "  while (id*x_inc < n) {" << std::endl;
	ss << "    Dtype x = xgm[id*x_inc + get_group_id(1) * n];" << std::endl;
	ss << "    acc += x;" << std::endl;
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
	ss << "    output[wgid + get_group_id(1) * num_groups] = lm[0] * alpha;" << std::endl;
	ss << "  }" << std::endl;
	ss << "}" << std::endl;


	ss << "__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))" << std::endl;
	ss << "void XasumEpilogue(const __global Dtype* restrict input," << std::endl;
	ss << "                   __global Dtype* asum, Dtype beta) {" << std::endl;
	      
	ss << "  __local float lm[WGS2];" << std::endl;
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


std::string general_gemm_kernel() {

	std::stringstream ss;

	
	ss << "#define TSM 16                      // The tile-size in dimension M" << std::endl;
	ss << "#define TSN 16                      // The tile-size in dimension N" << std::endl;
	ss << "#define TSK 4                       // The tile-size in dimension K" << std::endl;
	ss << "#define WPTM 4                       // The amount of work-per-thread in dimension M" << std::endl;
	ss << "#define WPTN 4                       // The amount of work-per-thread in dimension N" << std::endl;
	ss << "#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)" << std::endl;
	ss << "#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)" << std::endl;
	ss << "#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A" << std::endl;
	ss << "#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B" << std::endl;


	ss << "__kernel void myGEMM6(const int M, const int N, const int K," << std::endl;
	ss << "                      const __global float* A," << std::endl;
	ss << "                      const __global float* B," << std::endl;
	ss << "                      __global float* C) {" << std::endl;

	ss << "    // Thread identifiers" << std::endl;
	ss << "    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)" << std::endl;
	ss << "    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)" << std::endl;
	ss << "    const int offsetM = TSM*get_group_id(0); // Work-group offset" << std::endl;
	ss << "    const int offsetN = TSN*get_group_id(1); // Work-group offset" << std::endl;

	ss << "    // Local memory to fit a tile of A and B" << std::endl;
	ss << "    __local float Asub[TSK][TSM];" << std::endl;
	ss << "    __local float Bsub[TSN][TSK+2];" << std::endl;

	ss << "    // Allocate register space" << std::endl;
	ss << "    float Areg;" << std::endl;
	ss << "    float Breg[WPTN];" << std::endl;
	ss << "    float acc[WPTM][WPTN];" << std::endl;

	ss << "    // Initialise the accumulation registers" << std::endl;
	ss << "    #pragma unroll" << std::endl;
	ss << "    for (int wm=0; wm<WPTM; wm++) {" << std::endl;
	ss << "        #pragma unroll" << std::endl;
	ss << "        for (int wn=0; wn<WPTN; wn++) {" << std::endl;
	ss << "            acc[wm][wn] = 0.0f;" << std::endl;
	ss << "        }" << std::endl;
	ss << "    }" << std::endl;
	                                
	ss << "    // Loop over all tiles" << std::endl;
	ss << "    const int numTiles = K/TSK;" << std::endl;
	ss << "    int t=0;" << std::endl;
	ss << "    do {" << std::endl;

	ss << "        // Load one tile of A and B into local memory" << std::endl;
	ss << "        #pragma unroll" << std::endl;
	ss << "        for (int la=0; la<LPTA; la++) {" << std::endl;
	ss << "            int tid = tidn*RTSM + tidm;" << std::endl;
	ss << "            int id = la*RTSN*RTSM + tid;" << std::endl;
	ss << "            int row = MOD2(id,TSM);" << std::endl;
	ss << "            int col = DIV2(id,TSM);" << std::endl;
	ss << "            int tiledIndex = TSK*t + col;" << std::endl;
	ss << "            Asub[col][row] = A[tiledIndex*M + offsetM + row];" << std::endl;
	ss << "            Bsub[row][col] = B[tiledIndex*N + offsetN + row];" << std::endl;
	ss << "        }" << std::endl;

	ss << "        // Synchronise to make sure the tile is loaded" << std::endl;
	ss << "        barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	ss << "        // Loop over the values of a single tile" << std::endl;
	ss << "        for (int k=0; k<TSK; k++) {" << std::endl;

	ss << "            // Cache the values of Bsub in registers" << std::endl;
	ss << "            #pragma unroll" << std::endl;
	ss << "            for (int wn=0; wn<WPTN; wn++) {" << std::endl;
	ss << "                int col = tidn + wn*RTSN;" << std::endl;
	ss << "                Breg[wn] = Bsub[col][k];" << std::endl;
	ss << "            }" << std::endl;

	ss << "            // Perform the computation" << std::endl;
	ss << "            #pragma unroll" << std::endl;
	ss << "            for (int wm=0; wm<WPTM; wm++) {" << std::endl;
	ss << "                int row = tidm + wm*RTSM;" << std::endl;
	ss << "                Areg = Asub[k][row];" << std::endl;
	ss << "                #pragma unroll" << std::endl;
	ss << "                for (int wn=0; wn<WPTN; wn++) {" << std::endl;
	ss << "                    acc[wm][wn] += Areg * Breg[wn];" << std::endl;
	ss << "                }" << std::endl;
	ss << "            }" << std::endl;
	ss << "        }" << std::endl;

	ss << "        // Synchronise before loading the next tile" << std::endl;
	ss << "        barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	ss << "        // Next tile" << std::endl;
	ss << "        t++;" << std::endl;
	ss << "    } while (t<numTiles);" << std::endl;

	ss << "    // Store the final results in C" << std::endl;
	ss << "    #pragma unroll" << std::endl;
	ss << "    for (int wm=0; wm<WPTM; wm++) {" << std::endl;
	ss << "        int globalRow = offsetM + tidm + wm*RTSM;" << std::endl;
	ss << "        #pragma unroll" << std::endl;
	ss << "        for (int wn=0; wn<WPTN; wn++) {" << std::endl;
	ss << "            int globalCol = offsetN + tidn + wn*RTSN;" << std::endl;
	ss << "            C[globalCol*M + globalRow] = acc[wm][wn];" << std::endl;
	ss << "        }" << std::endl;
	ss << "    }" << std::endl;
	ss << "}" << std::endl;
}



}  // namespace caffe
