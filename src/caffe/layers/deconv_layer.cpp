#include <vector>

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

template <typename Dtype>
void DeconvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = stride_data[i] * (input_dim - 1)
        + kernel_extent - 2 * pad_data[i];
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}




template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  cl_int ret = -1;

  cl_kernel kernel = clCreateKernel(this->program, (this->layer_param_.name() + "_forward").c_str(), &ret);

  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();


    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
    OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&weight));  
    OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_data));

    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bias));
    }

    size_t* local_size = new size_t[3];
    local_size[0] = static_cast<size_t>(this->rtsn_);
    local_size[1] = static_cast<size_t>(this->rtsm_);
    local_size[2] = static_cast<size_t>(1);

    size_t* global_size = new size_t[3];
    global_size[0] = static_cast<size_t>((((top[i]->shape(2) * top[i]->shape(3)) - 1) / this->tsn_ + 1)*this->rtsn_);
    global_size[1] = static_cast<size_t>((((top[i]->shape(1) / this->group_) - 1) / this->tsm_ + 1)*this->rtsm_);
    global_size[2] = static_cast<size_t>(bottom[i]->shape()[0] * 1);

    OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL));  

#ifndef __ANDROID__
    int skip_bi = this->bottom_shape_[0].size() - this->output_shape_.size();
    int fmaps_in_ = this->bottom_shape_[0][skip_bi-1];
    int fmaps_out_ = this->num_output_;
    int MG_FW_ = fmaps_out_;
    int M_FW_ = fmaps_out_ / this->group_;
    int N_FW_ = 1;
    int KG_FW_ = fmaps_in_;
    int K_FW_ = fmaps_in_ / this->group_;

    
    for (int i = 0; i < this->output_shape_.size(); ++i) {
      K_FW_ *= this->kernel_shape_.cpu_data()[i];
      KG_FW_ *= this->kernel_shape_.cpu_data()[i];
      N_FW_ *= this->output_shape_[i];
    }

    LOG(INFO) << "The size of deconv are " << M_FW_ << " and "<< N_FW_ << " and " << K_FW_;
#endif
  }

}

template<typename Dtype>
std::string DeconvolutionLayer<Dtype>::generate_fw_defs() {
  
  std::stringstream ss;


  this->add_def(ss, "v_g", this->group_);

  int skip_bi = this->bottom_shape_[0].size() - this->output_shape_.size();
  int fmaps_in_ = this->bottom_shape_[0][skip_bi-1];
  int fmaps_out_ = this->num_output_;

  int A_off = fmaps_in_ * fmaps_out_;
  int B_off = fmaps_in_;
  int C_off = fmaps_out_;

  for (int i = 0; i < this->output_shape_.size(); ++i) {
    A_off *= this->kernel_shape_.cpu_data()[i];
    B_off *= this->bottom_shape_[0][skip_bi+i];
    C_off *= this->output_shape_[i];
  }

  this->add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  this->add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  this->add_def(ss, "v_C_off", C_off);

  int imsi = 1;
  int imso = 1;


  for (int i = 0; i < this->output_shape_.size(); ++i) {
    this->add_def(ss, "v_imsi_" + std::to_string(i), this->bottom_shape_[0][skip_bi+i]);
    imsi *= this->bottom_shape_[0][skip_bi+i];
    this->add_def(ss, "v_imso_" + std::to_string(i), this->output_shape_[i]);
    imso *= this->output_shape_[i];
  }
  this->add_def(ss, "v_imsi", imsi);
  this->add_def(ss, "v_imso", imso);

  int v_ks = 1;

  for (int i = 0; i < this->kernel_shape_.count(); ++i) {
    this->add_def(ss, "v_k_" + std::to_string(i), this->kernel_shape_.cpu_data()[i]);
    v_ks *= this->kernel_shape_.cpu_data()[i];
  }

  this->add_def(ss, "v_ks", v_ks);


  for (int i = 0; i < this->pad_.count(); ++i) {
    this->add_def(ss, "v_p_" + std::to_string(i), (this->kernel_shape_.cpu_data()[i] - 1) * this->dilation_.cpu_data()[i] - this->pad_.cpu_data()[i]);
  }

  for (int i = 0; i < this->stride_.count(); ++i) {
    this->add_def(ss, "v_s_" + std::to_string(i), this->stride_.cpu_data()[i]);
  }

  for (int i = 0; i < this->dilation_.count(); ++i) {
    this->add_def(ss, "v_d_" + std::to_string(i), this->dilation_.cpu_data()[i]);
  }

  this->add_def(ss, "v_fin", fmaps_in_);
  this->add_def(ss, "v_fout", fmaps_out_);

  // if (bias_term_) {
  //   this->add_def(ss, "v_bmul", bias_multiplier_);
  // }
  int MG_FW_ = fmaps_out_;
  int M_FW_ = fmaps_out_ / this->group_;
  int N_FW_ = 1;
  int KG_FW_ = fmaps_in_;
  int K_FW_ = fmaps_in_ / this->group_;

  for (int i = 0; i < this->output_shape_.size(); ++i) {
    K_FW_ *= this->kernel_shape_.cpu_data()[i];
    KG_FW_ *= this->kernel_shape_.cpu_data()[i];
    N_FW_ *= this->output_shape_[i];
  }

  // GEMM definitions
  this->add_def(ss, "MG", MG_FW_);
  this->add_def(ss, "M", M_FW_);
  this->add_def(ss, "N", N_FW_);
  this->add_def(ss, "KG", KG_FW_);
  this->add_def(ss, "K", K_FW_);

    // Local memory padding
  this->add_def(ss, "v_pad_A", 1);
  this->add_def(ss, "v_pad_B", 1);

  // The tile-size in dimension M
  this->add_def(ss, "TSM", this->tsm_);
  // The tile-size in dimension N
  this->add_def(ss, "TSN", this->tsn_);
  // The tile-size in dimension K
  this->add_def(ss, "TSK", this->tsk_);
  // TSK unrolling
  this->add_def(ss, "TSK_UNROLL", this->tsk_unroll_);
  // The work-per-thread in dimension M
  this->add_def(ss, "WPTM", this->wptm_);
  this->add_def(ss, "VWM", this->vwm_);
  // The work-per-thread in dimension N
  this->add_def(ss, "WPTN", this->wptn_);
  this->add_def(ss, "VWN", this->vwn_);
  // The reduced tile-size in dimension M
  this->add_def(ss, "RTSM", this->rtsm_);
  // The reduced tile-size in dimension N
  this->add_def(ss, "RTSN", this->rtsn_);
  // Loads-per-thread for A
  this->add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  this->add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  this->add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}





template <typename Dtype>
std::string DeconvolutionLayer<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;

  bool skip_range_check_ = true;

  for (int i = 0; i < this->pad_.count(); ++i) {
    if (this->pad_.cpu_data()[i] > 0) {
      skip_range_check_ = false;
    }
  }

 
  // Backward kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << this->rtsn_ << ", " << this->rtsm_ << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(this->vwm_, this->vwn_) << ")))" << std::endl;
  ss << "void " + name + "(";
  ss << "__global const Dtype* __restrict im_out, ";
  ss << "__global const Dtype* __restrict wg, ";
  ss << "__global Dtype* __restrict im_in";
    if (this->bias_term_) {
    ss << ", __global const Dtype* __restrict bias";
  }
  ss << ") {" << std::endl;

  // Thread identifiers
  // Local row ID (max: TSM/WPTM)
  ss << "const int tidn = get_local_id(0);" << std::endl;
  // Local col ID (max: TSN/WPTN)
  ss << "const int tidm = get_local_id(1);" << std::endl;
  // Work-group offset
  ss << "const int offN = TSN*get_group_id(0);" << std::endl;
  // Work-group offset
  ss << "const int offM = TSM*get_group_id(1);" << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << "volatile __local Dtype Asub[" << this->tsm_ << "][" << this->tsk_ << " + v_pad_A];"
     << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << "volatile __local Dtype Bsub[" << this->tsk_ << "][" << this->tsn_ << " + v_pad_B];"
     << std::endl;

  // Batch and group
  if (this->group_ > 1) {
    ss << "int group = get_global_id(2) % v_g;" << std::endl;
    ss << "int batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int batch = get_global_id(2);" << std::endl;
  }

  if (this->group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (v_A_off / (v_g * v_g));"
       << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch "
       << "+ group * (v_C_off / v_g);" << std::endl;
    if (this->bias_term_) {
      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);"
          << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch;" << std::endl;
    if (this->bias_term_) {
      ss << "__global const Dtype* Dptr = bias;" << std::endl;
    }
  }


  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(false, false);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  ss << "for (int la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int row = id / TSK;" << std::endl;
  ss << "int col = id % TSK;" << std::endl;
  ss << "int tiledIndex = TSK * t + col;" << std::endl;

    // Load weights (wg) into Asub, flip fin/fout and inverse spatially
    // Compute kidx and midx, the column and row index of the
    // weights in the original A (weights) matrix
    ss << "int kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;"
       << std::endl;
    ss << "int midx = tiledIndex / v_ks;" << std::endl;
    // Check range of the spatially flipped, fin/fout inverted weights
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    // Access weights with the original (translated) weight indices
    ss << "Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];"
       << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;

  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading A

  // Load one tile of B into local memory
  ss << "{" << std::endl;  // Scoping for loading B
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int id = lb * RTSN * RTSM + tid;" << std::endl;
  ss << "int col = id % TSN;" << std::endl;
  ss << "int row = id / TSN;" << std::endl;
  ss << "int tiledIndex = TSK * t + row;" << std::endl;

  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;

    // Load from B with im2col transformation

    // Define temporary registers
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      ss << "int d_iter_" << i << ";" << std::endl;
      ss << "int d_temp_" << i << ";" << std::endl;
    }

    // Compute in-range
    ss << "bool in_range = true;" << std::endl;

    ss << "int imageIndex = offN + col;" << std::endl;
    for (int i = this->num_spatial_axes_ - 1; i >= 0; --i) {
      // Compute d_iter, final tiledIndex becomes input feature map ID
      // Scale d_iter by the dilation factor
      ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

      // Compute d_temp
      // Subtract the padding from d_temp, note v_p_i can be negative
      ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ")"
         << " - v_p_" << i << ";" << std::endl;
      ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
    }

    ss << "int d_iter_im;" << std::endl;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      // Here, d_temp_ represents the column shift,
      // while d_iter_ is the kernel shift
      ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
      ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im / v_s_"
         << i << ";" << std::endl;
      // In range: Not before or after actual image data
      // and not between image strides
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i
         << " * v_s_" << i << " && d_iter_im % v_s_" << i << " == 0;"
         << std::endl;
    }

    ss << "if (in_range) {" << std::endl;
    // tiledIndex now holds the memory offset for the input image
    ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
    ss << "} else {" << std::endl;
    // Out of B's image dimensions
    ss << "Bsub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;


  ss << "} else {" << std::endl;
  // Out of B's matrix dimensions
  ss << "Bsub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading B

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << this->generate_gemm_core(false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int globalRow = offM + tidm + wm * RTSM;" <<std::endl;
  if (this->bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int globalCol = offN + tidn + wn * RTSN;" << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  ss << "Cptr[globalRow * N + globalCol] = ";
  if (this->bias_term_) {
    ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]"
       << " + biasval;" << std::endl;
  } else {
    ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  }
  ss << "}" << std::endl;


  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}






// #ifdef CPU_ONLY
// STUB_GPU(DeconvolutionLayer);
// #endif

INSTANTIATE_CLASS(DeconvolutionLayer);
REGISTER_LAYER_CLASS(Deconvolution);

}  // namespace caffe
