
#define MItype float
#define MItype1 float
#define MItype2 float2
#define MItype4 float4
#define MItype8 float8
#define MItype16 float16

#define MOtype float
#define MOtype1 float
#define MOtype2 float2
#define MOtype4 float4
#define MOtype8 float8
#define MOtype16 float16

#define Acctype float
#define Acctype1 float
#define Acctype2 float2
#define Acctype4 float4
#define Acctype8 float8
#define Acctype16 float16

#define VEC_1_0(ELEM) ELEM
#define VEC_2_0(ELEM) ELEM.x
#define VEC_2_1(ELEM) ELEM.y
#define VEC_4_0(ELEM) ELEM.x
#define VEC_4_1(ELEM) ELEM.y
#define VEC_4_2(ELEM) ELEM.z
#define VEC_4_3(ELEM) ELEM.w
/*
#define VEC_8_0(ELEM) ELEM.s0
#define VEC_8_1(ELEM) ELEM.s1
#define VEC_8_2(ELEM) ELEM.s2
#define VEC_8_3(ELEM) ELEM.s3
#define VEC_8_4(ELEM) ELEM.s4
#define VEC_8_5(ELEM) ELEM.s5
#define VEC_8_6(ELEM) ELEM.s6
#define VEC_8_7(ELEM) ELEM.s7
#define VEC_16_0(ELEM) ELEM.s0
#define VEC_16_1(ELEM) ELEM.s1
#define VEC_16_2(ELEM) ELEM.s2
#define VEC_16_3(ELEM) ELEM.s3
#define VEC_16_4(ELEM) ELEM.s4
#define VEC_16_5(ELEM) ELEM.s5
#define VEC_16_6(ELEM) ELEM.s6
#define VEC_16_7(ELEM) ELEM.s7
#define VEC_16_8(ELEM) ELEM.s8
#define VEC_16_9(ELEM) ELEM.s9
#define VEC_16_10(ELEM) ELEM.sA
#define VEC_16_11(ELEM) ELEM.sB
#define VEC_16_12(ELEM) ELEM.sC
#define VEC_16_13(ELEM) ELEM.sD
#define VEC_16_14(ELEM) ELEM.sE
#define VEC_16_15(ELEM) ELEM.sF*/



#ifdef v_pad_A
#undef v_pad_A
#endif  //v_pad_A
#define v_pad_A 0
#ifdef v_pad_B
#undef v_pad_B
#endif  //v_pad_B
#define v_pad_B 0
#ifdef TSM
#undef TSM
#endif  //TSM
#define TSM 128
#ifdef TSN
#undef TSN
#endif  //TSN
#define TSN 128
#ifdef TSK
#undef TSK
#endif  //TSK
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif  //TSK_UNROLL
#define TSK_UNROLL 1
#ifdef WPTM
#undef WPTM
#endif  //WPTM
#define WPTM 8
#ifdef VWM
#undef VWM
#endif  //VWM
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif  //WPTN
#define WPTN 8
#ifdef VWN
#undef VWN
#endif  //VWN
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif  //RTSM
#define RTSM 16
#ifdef RTSN
#undef RTSN
#endif  //RTSN
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif  //LPTA
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif  //LPTB
#define LPTB ((TSK*TSN)/(RTSM*RTSN))



__kernel void conv_forward(__global const float* im_in,
				    __global const float* wg,
				    __global float* im_out,
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
					int K
					) {
	
	int v_num_tiles = (((K - 1)/(TSK*2) + 1)*2);

	const int tidn = get_local_id(0);
	const int tidm = get_local_id(1);
	const int offN = TSN * get_group_id(0);
	const int offM = TSM * get_group_id(1);
	__local MItype Asub[128][8 + v_pad_A];
	__local MItype Bsub[8][128 + v_pad_B];
	int batch = get_global_id(2);
	__global const MItype* Aptr = wg;
	__global const MItype* Bptr = im_in + v_B_off * batch;
	__global MItype* Cptr = im_out + v_C_off * batch;
	{
		Acctype4 Creg[WPTM][WPTN / VWN];
		#pragma unroll
		for (int wm = 0; wm < WPTM; ++wm) {
			#pragma unroll
			for (int wn = 0; wn < WPTN; ++wn) {
				((Acctype*)(&(Creg[wm][wn / VWN])))[wn % VWN] = (Acctype)0;
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
							Asub[row][col] = (MItype)0.0;
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
							int d_iter_im;
							d_iter_im = d_temp_0 + d_iter_0;
							tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
							d_iter_im = d_temp_1 + d_iter_1;
							tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
							Bsub[row][col] = Bptr[tiledIndex];
						} else {
							Bsub[row][col] = (MItype)0.0;
						}
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				MItype4 Areg;
				MItype4 Breg[TSK_UNROLL*WPTN/VWN];
				#pragma unroll 1
				for (int k = 0; k < TSK; k += TSK_UNROLL) {
					#pragma unroll
					for (int wn = 0; wn < WPTN / (VWN / TSK_UNROLL); ++wn) {
						int col = tidn + wn * (VWN / TSK_UNROLL) * RTSN;
						VEC_4_0(Breg[wn]) = Bsub[k + 0][col + 0];
						VEC_4_1(Breg[wn]) = Bsub[k + 0][col + 16];
						VEC_4_2(Breg[wn]) = Bsub[k + 0][col + 32];
						VEC_4_3(Breg[wn]) = Bsub[k + 0][col + 48];
					}
					#pragma unroll
					for (int wm = 0; wm < WPTM / (VWM / TSK_UNROLL); ++wm) {
						int row = tidm + wm * (VWM / TSK_UNROLL) * RTSM;
						VEC_4_0(Areg) = Asub[row + 0][k + 0];
						VEC_4_1(Areg) = Asub[row + 16][k + 0];
						VEC_4_2(Areg) = Asub[row + 32][k + 0];
						VEC_4_3(Areg) = Asub[row + 48][k + 0];
						#pragma unroll
						for (int wn = 0; wn < WPTN / VWN; ++wn) {
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
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
					Cptr[globalRow * N + globalCol] = (MOtype)(((Acctype*)(&(Creg[wm][wn/VWN])))[wn % VWN]);
				}
			}
		}
	}
}

__kernel void conv_forward_with_bias(__global const float* im_in,
				    __global const float* wg, 
				    __global float* im_out,
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
				    __global const float* bias
					) {
	
	int v_num_tiles = (((K - 1)/(TSK*2) + 1)*2);

	const int tidn = get_local_id(0);
	const int tidm = get_local_id(1);
	const int offN = TSN * get_group_id(0);
	const int offM = TSM * get_group_id(1);
	__local MItype Asub[128][8 + v_pad_A];
	__local MItype Bsub[8][128 + v_pad_B];
	int batch = get_global_id(2);
	__global const MItype* Aptr = wg;
	__global const MItype* Bptr = im_in + v_B_off * batch;
	__global MItype* Cptr = im_out + v_C_off * batch;
	__global const MItype* Dptr = bias;
	
	{
		Acctype4 Creg[WPTM][WPTN / VWN];
		#pragma unroll
		for (int wm = 0; wm < WPTM; ++wm) {
			#pragma unroll
			for (int wn = 0; wn < WPTN; ++wn) {
				((Acctype*)(&(Creg[wm][wn / VWN])))[wn % VWN] = (Acctype)0;
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
							Asub[row][col] = (MItype)0.0;
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
							int d_iter_im;
							d_iter_im = d_temp_0 + d_iter_0;
							tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
							d_iter_im = d_temp_1 + d_iter_1;
							tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
							Bsub[row][col] = Bptr[tiledIndex];
						} else {
							Bsub[row][col] = (MItype)0.0;
						}
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				MItype4 Areg;
				MItype4 Breg[TSK_UNROLL*WPTN/VWN];
				#pragma unroll 1
				for (int k = 0; k < TSK; k += TSK_UNROLL) {
					#pragma unroll
					for (int wn = 0; wn < WPTN / (VWN / TSK_UNROLL); ++wn) {
						int col = tidn + wn * (VWN / TSK_UNROLL) * RTSN;
						VEC_4_0(Breg[wn]) = Bsub[k + 0][col + 0];
						VEC_4_1(Breg[wn]) = Bsub[k + 0][col + 16];
						VEC_4_2(Breg[wn]) = Bsub[k + 0][col + 32];
						VEC_4_3(Breg[wn]) = Bsub[k + 0][col + 48];
					}
					#pragma unroll
					for (int wm = 0; wm < WPTM / (VWM / TSK_UNROLL); ++wm) {
						int row = tidm + wm * (VWM / TSK_UNROLL) * RTSM;
						VEC_4_0(Areg) = Asub[row + 0][k + 0];
						VEC_4_1(Areg) = Asub[row + 16][k + 0];
						VEC_4_2(Areg) = Asub[row + 32][k + 0];
						VEC_4_3(Areg) = Asub[row + 48][k + 0];
						#pragma unroll
						for (int wn = 0; wn < WPTN / VWN; ++wn) {
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
							VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
						}
					}
				}

				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}

		#pragma unroll
		for (int wm=0; wm<WPTM; ++wm) {
			int globalRow = offM + tidm + wm * RTSM;
			MItype biasval = Dptr[globalRow];
			#pragma unroll
			for (int wn=0; wn<WPTN; ++wn) {
				int globalCol = offN + tidn + wn * RTSN;
				if (globalRow < M && globalCol < N) {
					Cptr[globalRow * N + globalCol] = (MOtype)(((Acctype*)(&(Creg[wm][wn/VWN])))[wn % VWN] + biasval);
				}
			}
		}
	}
}
