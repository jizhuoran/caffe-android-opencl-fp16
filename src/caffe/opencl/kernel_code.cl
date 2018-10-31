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

#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w


#define v_pad_A 0
#define v_pad_B 0
#define TSM 64
#define TSN 64
#define TSK 8
#define TSK_UNROLL 1
#define WPTM 4
#define VWM 4
#define WPTN 4
#define VWN 4
#define RTSM 16
#define RTSN 16
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))



__kernel void conv_forward(
	__global const Dtype* __restrict im_in,
	__global const Dtype* __restrict wg,
	__global Dtype* __restrict im_out,
	int v_B_off,
	int v_C_off,
	int v_imsi_0,
	int v_imso_0,
	int v_imsi_1,
	int v_imso_1,
	int v_imsi,
	int v_imso,
	int v_k_0,
	int v_k_1,
	int v_p_0,
	int v_p_1,
	int v_s_0,
	int v_s_1,
	int v_d_0,
	int v_d_1,
	int v_fin,
	int v_fout,
	int MG,
	int M,
	int N,
	int KG,
	int K) {

	int v_num_tiles = (((K - 1)/(TSK*2) + 1)*2);
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[64][8 + v_pad_A];
volatile __local Dtype Bsub[8][64 + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 16][k];
VEC_4_2(Areg) = Asub[row + 32][k];
VEC_4_3(Areg) = Asub[row + 48][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}




__kernel void conv_forward_with_bias(
	__global const Dtype* im_in,
	__global const Dtype* wg,
	__global Dtype* im_out,
	int v_B_off,
	int v_C_off,
	int v_imsi_0,
	int v_imso_0,
	int v_imsi_1,
	int v_imso_1,
	int v_imsi,
	int v_imso,
	int v_k_0,
	int v_k_1,
	int v_p_0,
	int v_p_1,
	int v_s_0,
	int v_s_1,
	int v_d_0,
	int v_d_1,
	int v_fin,
	int v_fout,
	int MG,
	int M,
	int N,
	int KG,
	int K,
	__global const Dtype* bias) {

	int v_num_tiles = (((K - 1)/(TSK*2) + 1)*2);
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[64][8 + v_pad_A];
volatile __local Dtype Bsub[8][64 + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 16][k];
VEC_4_2(Areg) = Asub[row + 32][k];
VEC_4_3(Areg) = Asub[row + 48][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
