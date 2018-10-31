#include <algorithm>
#include <vector>
#include <iomanip>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"



template<class T>
inline void add_def(std::stringstream& ss,  // NOLINT
    const char* name, T value) {
  ss << "#ifdef " << name << std::endl;
  ss << "#undef " << name << std::endl;
  ss << "#endif" << std::endl;
  if (std::is_same<T, float>::value) {
    ss << "#define " << name << " (float) " << std::setprecision(32) << value
        << std::endl;
  } else if (std::is_same<T, double>::value) {
    ss << "#define " << name << " (double) " << std::setprecision(32) << value
        << std::endl;
  } else {
    ss << "#define " << name << " " << value << std::endl;
  }
}

template<class T>
inline void add_def(std::stringstream& ss,  // NOLINT
    const std::string name, T value) {
  add_def(ss, name.c_str(), value);
}


namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  
}


template<typename Dtype>
std::string BaseConvolutionLayer<Dtype>::generate_header() {
  std::stringstream ss;

  if (std::is_same<Dtype, double>::value) {
    ss << "#define Dtype double" << std::endl;
    ss << "#define Dtype1 double" << std::endl;
    // double2, double4, double8, double16
    for (int i = 2; i <= 16; i *= 2) {
      ss << "#define Dtype" << i << " double" << i << std::endl;
    }
  } else {
    ss << "#define Dtype float" << std::endl;
    ss << "#define Dtype1 float" << std::endl;
    // float2, float4, float8, float16
    for (int i = 2; i <= 16; i *= 2) {
      ss << "#define Dtype" << i << " float" << i << std::endl;
    }
  }

  std::vector<std::string> elems4({
      "x", "y", "z", "w" });
  std::vector<std::string> elems16({
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8", "s9", "sA", "sB", "sC", "sD", "sE", "sF" });

  for (int i = 1; i <= 16; i *= 2) {
    for (int j = 0; j < i; ++j) {
      if (i == 1) {
        ss << "#define VEC_" << i << "_" << j << "(X)" << " X" << std::endl;
      } else if (i < 8) {
        ss << "#define VEC_" << i << "_" << j << "(X)" << " X." << elems4[j]
           << std::endl;
      } else {
        ss << "#define VEC_" << i << "_" << j << "(X)" << " X." << elems16[j]
           << std::endl;
      }
    }
  }

  return ss.str();
}


template<typename Dtype>
std::string BaseConvolutionLayer<Dtype>::generate_fw_defs() {
  
  std::stringstream ss;

  int skip_bi = this->bottom_shape_[0].size() - output_shape_.size();
  int fmaps_in_ = this->bottom_shape_[0][skip_bi-1];
  int fmaps_out_ = num_output_;

  int B_off = fmaps_in_;
  int C_off = fmaps_out_;

  for (int i = 0; i < output_shape_.size(); ++i) {
    B_off *= this->bottom_shape_[0][skip_bi+i];
    C_off *= this->output_shape_[i];
  }


  // Input image batch offset
  add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  add_def(ss, "v_C_off", C_off);

  int imsi = 1;
  int imso = 1;


  for (int i = 0; i < output_shape_.size(); ++i) {
    add_def(ss, "v_imsi_" + std::to_string(i), this->bottom_shape_[0][skip_bi+i]);
    imsi *= this->bottom_shape_[0][skip_bi+i];
    add_def(ss, "v_imso_" + std::to_string(i), this->output_shape_[i]);
    imso *= this->output_shape_[i];
  }
  add_def(ss, "v_imsi", imsi);
  add_def(ss, "v_imso", imso);

  for (int i = 0; i < kernel_shape_.count(); ++i) {
    add_def(ss, "v_k_" + std::to_string(i), kernel_shape_.cpu_data()[i]);
  }


  for (int i = 0; i < pad_.count(); ++i) {
    add_def(ss, "v_p_" + std::to_string(i), pad_.cpu_data()[i]);
  }

  for (int i = 0; i < stride_.count(); ++i) {
    add_def(ss, "v_s_" + std::to_string(i), stride_.cpu_data()[i]);
  }

  for (int i = 0; i < dilation_.count(); ++i) {
    add_def(ss, "v_d_" + std::to_string(i), dilation_.cpu_data()[i]);
  }

  add_def(ss, "v_fin", fmaps_in_);
  add_def(ss, "v_fout", fmaps_out_);

  // if (bias_term_) {
  //   add_def(ss, "v_bmul", bias_multiplier_);
  // }
  int MG_FW_ = fmaps_out_;
  int M_FW_ = fmaps_out_ / group_;
  int N_FW_ = 1;
  int KG_FW_ = fmaps_in_;
  int K_FW_ = fmaps_in_ / group_;

  for (int i = 0; i < output_shape_.size(); ++i) {
    K_FW_ *= kernel_shape_.cpu_data()[i];
    KG_FW_ *= kernel_shape_.cpu_data()[i];
    N_FW_ *= output_shape_[i];
  }

  // GEMM definitions
  add_def(ss, "MG", MG_FW_);
  add_def(ss, "M", M_FW_);
  add_def(ss, "N", N_FW_);
  add_def(ss, "KG", KG_FW_);
  add_def(ss, "K", K_FW_);

    // Local memory padding
  add_def(ss, "v_pad_A", 1);
  add_def(ss, "v_pad_B", 1);

  // The tile-size in dimension M
  add_def(ss, "TSM", 64);
  // The tile-size in dimension N
  add_def(ss, "TSN", 64);
  // The tile-size in dimension K
  add_def(ss, "TSK", 8);
  // TSK unrolling
  add_def(ss, "TSK_UNROLL", 1);
  // The work-per-thread in dimension M
  add_def(ss, "WPTM", 4);
  add_def(ss, "VWM", 4);
  // The work-per-thread in dimension N
  add_def(ss, "WPTN", 4);
  add_def(ss, "VWN", 4);
  // The reduced tile-size in dimension M
  add_def(ss, "RTSM", 16);
  // The reduced tile-size in dimension N
  add_def(ss, "RTSN", 16);
  // Loads-per-thread for A
  add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}


template<typename Dtype>
std::string BaseConvolutionLayer<Dtype>::generate_gemm_core(bool dterm) {
  std::stringstream ss;
  int vwm = 4;
  int vwn = 4;
  int rtsn = 16;
  int rtsm = 16;
  bool unroll = true;

  // Temporary registers for A and B
  ss << "Dtype" << vwm << " Areg;" << std::endl;
  ss << "Dtype" << vwn << " Breg[WPTN/VWN];" << std::endl;

  // Loop over the values of a single tile
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {" << std::endl;
  ss << "#pragma unroll " << 1 << std::endl;
  ss << "for (int ku=0; ku<TSK_UNROLL; ++ku) {" << std::endl;
  ss << "int k = kt + ku;" << std::endl;

  // Cache the values of Bsub in registers
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  ss << "int col = tidn + wn*VWN*RTSN;" << std::endl;
  for (int i = 0; i < vwn; ++i) {
    ss << "VEC_" << vwn << "_" << i << "(Breg[wn])"
       << " = Bsub[k][col + " << (i*rtsn)
       << "];" << std::endl;
  }
  ss << "}" << std::endl;

  // Perform the computation
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
  ss << "int row = tidm + wm*VWM*RTSM;" << std::endl;
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Areg)" << " = Asub[row + " << (i*rtsm)
       << "][k];" << std::endl;
  }
  if (dterm) {
    if (unroll) {
      for (int i = 0; i < vwm; ++i) {
        ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) " << "+= VEC_" << vwm
           << "_" << i << "(Areg) * v_bmul;" << std::endl;
      }
    } else {
      ss << "Dreg[wm] += Areg * v_bmul;" << std::endl;
    }
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  if (unroll) {
    for (int n = 0; n < vwn; ++n) {
      for (int m = 0; m < vwm; ++m) {
        ss << "VEC_" << vwn << "_" << n << "(Creg[wm * VWM + " << m << "][wn])"
           << " += VEC_" << vwm << "_" << m << "(Areg)" << " * VEC_" << vwn
           << "_" << n << "(Breg[wn]);" << std::endl;
      }
    }
  } else {
    for (int m = 0; m < vwm; ++m) {
      ss << "Creg[wm * VWM + " << m << "][wn]"
         << " += VEC_"<< vwm << "_" << m << "(Areg)" << " * (Breg[wn]);"
         << std::endl;
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
std::string BaseConvolutionLayer<Dtype>::generate_accreg_init(bool dterm, bool load) {
  std::stringstream ss;

  int vwm = 4;
  int vwn = 4;
  bool unroll = true;

  if (dterm) {
    ss << "Dtype" << vwm << " Dreg[WPTM/VWM];" << std::endl;
  }
  ss << "Dtype" << vwn << " Creg[WPTM][WPTN/VWN];" << std::endl;

  // Initialize the accumulation registers
  if (load) {
    // Load
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
      ss << "int globalRow = offM + tidm + wm * RTSM;"
         << std::endl;
      ss << "((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM] = Dptr[globalRow];"
         << std::endl;
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "int globalRow = offM + tidm + wm * RTSM;"
       << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wn=0; wn<WPTN; ++wn) {" << std::endl;
    ss << "int globalCol = offN + tidn + wn * RTSN;"
       << std::endl;
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] = "
       << "Cptr[globalRow * N + globalCol];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  } else {
    // Zero init
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
      if (unroll) {
        for (int i = 0; i < vwm; ++i) {
          ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) = 0.0;" << std::endl;
        }
      } else {
        ss << "Dreg[wm] = 0.0;" << std::endl;
      }
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
    if (unroll) {
      for (int i = 0; i < vwn; ++i) {
        ss << "VEC_" << vwn << "_" << i << "(Creg[wm][wn]) = 0.0;" << std::endl;
      }
    } else {
      ss << "Creg[wm][wn] = 0.0;" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}



template <typename Dtype>
std::string BaseConvolutionLayer<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;

  int wptn = 4;
  int wptm = 4;
  int tsk = 8;
  int rtsn = 16;
  int rtsm = 16;
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = 4;
  int vwn = 4;
  int lpta = (tsm * tsk) / (rtsm * rtsn);
  int lptb = (tsn * tsk) / (rtsm * rtsn);

  bool skip_range_check_ = true;

  for (int i = 0; i < pad_.count(); ++i) {
    if (pad_.cpu_data()[i] > 0) {
      skip_range_check_ = false;
    }
  }

  // Forward kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << rtsn << ", " << rtsm << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(vwm, vwn) << ")))" << std::endl;
  ss << "void " + name + "(";
  ss << "__global const Dtype* __restrict im_in, ";
  ss << "__global const Dtype* __restrict wg, ";
  if (bias_term_) {
    ss << "__global const Dtype* __restrict bias, ";
  }
  ss << "__global Dtype* __restrict im_out";
  ss << ") {" << std::endl;

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int tidn = get_local_id(0);" << std::endl;
  // Local col ID (max: RTSN=TSN/WPTN)
  ss << "const int tidm = get_local_id(1);" << std::endl;
  // Work-group offset
  ss << "const int offN = TSN*get_group_id(0);" << std::endl;
  // Work-group offset
  ss << "const int offM = TSM*get_group_id(1);" << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << "volatile __local Dtype Asub[" << tsm << "][" << tsk << " + v_pad_A];"
     << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << "volatile __local Dtype Bsub[" << tsk << "][" << tsn << " + v_pad_B];"
     << std::endl;

  // Batch and group
  if (group_ > 1) {
    ss << "int group = get_global_id(2) % v_g;" << std::endl;
    ss << "int batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int batch = get_global_id(2);" << std::endl;
  }

  if (group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (M * K);" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch + group * (M * N);"
       << std::endl;
    if (bias_term_) {
      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);"
         << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch;" << std::endl;
    if (bias_term_) {
      ss << "__global const Dtype* Dptr = bias;" << std::endl;
    }
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << generate_accreg_init(false, false);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  /*if (rtsn * rtsm % tsk == 0) {
    ss << "int tid = tidm * RTSN + tidn;" << std::endl;
    ss << "int row = tid / TSK;" << std::endl;
    ss << "int col = tid % TSK;" << std::endl;
    ss << "int tiledIndex = TSK * t + col;" << std::endl;
    int rowstep = (rtsn * rtsm) / tsk;
    for (int i = 0; i < lpta; ++i) {
      ss << "if ((offM + row + " << i * rowstep << ") < M && tiledIndex < K) {"
         << std::endl;
      ss << "Asub[row+" << i * rowstep << "][col] = Aptr[(offM + row + "
         << i * rowstep << ") * K + tiledIndex];" << std::endl;
      ss << "} else {" << std::endl;  // M-K-Guard
      ss << "Asub[row+" << i * rowstep << "][col] = 0.0;" << std::endl;
      ss << "}";
    }
  } else {*/
    ss << "#pragma unroll 4" << std::endl;
    ss << "for (int la = 0; la < LPTA; ++la) {" << std::endl;
    ss << "int tid = tidm * RTSN + tidn;" << std::endl;
    ss << "int id = la * RTSN * RTSM + tid;" << std::endl;
    ss << "int row = id / TSK;" << std::endl;
    ss << "int col = id % TSK;" << std::endl;
    ss << "int tiledIndex = TSK * t + col;" << std::endl;
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    ss << "Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];" << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;  // LPTA
  //  }
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
  // Define temporary registers
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    ss << "int d_iter_" << i << ";" << std::endl;
    ss << "int d_temp_" << i << ";" << std::endl;
  }

  ss << "int imageIndex = offN + col;" << std::endl;
  for (int i = this->num_spatial_axes_ - 1; i >= 0; --i) {
    // Compute d_iter, final tiledIndex becomes input feature map ID
    // Scale d_iter by the dilation factor
    ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
       << ";" << std::endl;
    ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

    // Compute d_temp
    // Scale d_temp by the stride and subtract the padding
    ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ") * v_s_" << i
       << " - v_p_" << i << ";" << std::endl;
    ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
  }

  // Recombine final index, compute in-range
  if (!skip_range_check_) {
    ss << "bool in_range = true;" << std::endl;
  }
  ss << "int d_iter_im;" << std::endl;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // Here, d_temp_ represents the column shift,
    // while d_iter_ is the kernel shift
    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
    ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im;"
       << std::endl;
    if (!skip_range_check_) {
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i << ";"
         << std::endl;
    }
  }

  if (!skip_range_check_) {
    ss << "if (in_range) {" << std::endl;
  }
  // tiledIndex now holds the memory offset for the input image
  ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
  if (!skip_range_check_) {
    ss << "} else {" << std::endl;
    ss << "Bsub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading B

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << generate_gemm_core(false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block


  // Store the final results in C
  /*ss << "#pragma unroll 1" << std::endl;
  ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
  for (int j = 0; j < vwn; ++j) {
    for (int i = 0; i < vwm; ++i) {
      ss << "Asub[(tidn+wn*RTSN)*VWN + " << j << "][(tidm + wn*RTSN)*VWM + " << i << "] = VEC_" << vwm << "_" << i << "(Creg[wn + " << j << "][wm]);" << std::endl;
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for C registers

  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Store the final results in C
  ss << "{" << std::endl; // Scoping for storing C
  ss << "Dtype" << vwm << " Creg;" << std::endl;
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int lc = 0; lc < ((TSM*TSN-1)/(RTSM*RTSN))/VWM+1; ++lc) {" << std::endl;
  ss << "int tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int id = lc * RTSN * RTSM + tid;" << std::endl;
  ss << "int row = (id / TSN) * VWM;" << std::endl;
  ss << "int col = id % TSN;" << std::endl;
  ss << "int globalRow = offM + row;" << std::endl;
  ss << "int globalCol = offN + col;" << std::endl;
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Creg) = Asub[col][row + " << i << "];" << std::endl;
    ss << "if ((globalRow +" << i << ") < M && globalCol < N) {" << std::endl;
    if (bias_term_) {
      ss << "Cptr[(globalRow +" << i << ") * N + globalCol] = VEC_" << vwm << "_" << i << "(Creg) + Dptr[globalRow +" << i << "];" << std::endl;
    } else {
      ss << "Cptr[(globalRow +" << i << ") * N + globalCol] = VEC_" << vwm << "_" << i << "(Creg);" << std::endl;
    }
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl; // Scoping for storing C*/

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  if (bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  if (bias_term_) {
    ss << "Cptr[globalRow * N + globalCol] = "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;"
       << std::endl;
  } else {
    ss << "Cptr[globalRow * N + globalCol] = "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  }
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}



template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

  std::stringstream ss;

  ss << this->generate_header();
  ss << this->generate_fw_defs();
  ss << this->generate_fw_kernels(this->layer_param_.name() + "_forward");

  std::string conv_kernel = ss.str();
  size_t kernel_size = conv_kernel.size();

  cl_int ret = -1; 
  program = clCreateProgramWithSource(Caffe::Get().context, 1, (const char **)&conv_kernel, (const size_t *)&kernel_size, &ret); 
  OPENCL_CHECK(ret);
  
  // fprintf(stderr, "Come to Here!!! 2\n");

  ret = clBuildProgram(program, 1, &Caffe::Get().deviceID, NULL, NULL, NULL);


}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
