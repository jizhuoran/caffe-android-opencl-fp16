#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // this->bias_term_ = false;

  // if (!this->bias_term_ || true) {
  //   const Dtype* weight = this->blobs_[0]->cpu_data();
  //   for (int i = 0; i < bottom.size(); ++i) {
  //     const Dtype* bottom_data = bottom[i]->cpu_data();
  //     Dtype* top_data = top[i]->mutable_cpu_data();
  //     for (int n = 0; n < this->num_; ++n) {
  //       this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
  //           top_data + n * this->top_dim_);
  //       if (this->bias_term_) {
  //         const Dtype* bias = this->blobs_[1]->cpu_data();
  //         this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
  //       }
  //     }
  //   }

  //   for (int i = 0; i < 100; ++i) {
  //     std::cout << top[0]->cpu_data()[i] << " ";
  //   }
  //   // exit(0);
  //   // return;

  // }

  // const Dtype* weight = this->blobs_[0]->cpu_data();
  // for (int i = 0; i < bottom.size(); ++i) {
  //   const Dtype* bottom_data = bottom[i]->cpu_data();
  //   Dtype* top_data = top[i]->mutable_cpu_data();
  //   for (int n = 0; n < this->num_; ++n) {
  //     this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
  //         top_data + n * this->top_dim_);
  //     if (this->bias_term_) {
  //       const Dtype* bias = this->blobs_[1]->cpu_data();
  //       this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
  //     }
  //   }
  // }
  ConvolutionParameter conv_param = this->layer_param().convolution_param();
    const int channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
    
    const int first_spatial_axis = channel_axis_ + 1;
    const int num_axes = bottom[0]->num_axes();
    const int num_spatial_axes_ = num_axes - first_spatial_axis;
    int* kernel_shape_data = new int[2];
    // now we only deal with the data with kernel dimension equal to 2
    
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
    
    int* pad_data = new int[2];
    // now we only deal with the data with pad dimension equal to 2
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
    
    int* stride_data = new int[2];
    // now we only deal with the data with stride dimension equal to 2
    
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
    
    int* dilation_data = new int[num_spatial_axes_];
    const int kDefaultDilation = 1;
    const int num_dilation_dims = conv_param.dilation_size();
    for (int i = 0; i < num_spatial_axes_; ++i) {
        dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
        conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
    }


  cl_int ret = -1;

  cl_kernel kernel;

  if (this->bias_term_) {
    std::cout << "come to 1" << std::endl;
    kernel = clCreateKernel(Caffe::Get().program, "conv_forward_with_bias", &ret);
  } else {
    std::cout << "come to 2" << std::endl;

    kernel = clCreateKernel(Caffe::Get().program, "conv_forward", &ret);
  }

  OPENCL_CHECK(ret);


  for (int i = 0; i < bottom.size(); ++i) {
    
    std::cout << "This is the " << i << " time come to here!" << std::endl;


    const Dtype* weight = this->blobs_[0]->gpu_data();
    // std::cout << "before this" << std::endl;

    const Dtype* bottom_data = bottom[i]->gpu_data();

    // cl_mem weight_gpu = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, this->blobs_[0]->count() * 4, NULL, NULL);
    // OPENCL_CHECK(clEnqueueWriteBuffer(Caffe::Get().commandQueue, weight_gpu, CL_TRUE, 0, this->blobs_[0]->count() * 4, weight, 0, NULL, NULL));
    
    // cl_mem input_gpu = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, bottom[0]->count() * 4, NULL, NULL);
    // OPENCL_CHECK(clEnqueueWriteBuffer(Caffe::Get().commandQueue, input_gpu, CL_TRUE, 0, bottom[0]->count() * 4, bottom_data, 0, NULL, NULL));

    // cl_mem output_gpu = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, top[0]->count() * 4, NULL, NULL);


    // std::cout << "after this" << std::endl;

    Dtype* top_data = top[i]->mutable_gpu_data();

    // float* cpu_ptr_ = (float*)malloc(bottom[0]->count() * 4);
    // float* cpu_ptr_1 = (float*)malloc(this->blobs_[0]->count() * 4);
    // cl_mem gpu_ptr_ = clCreateBuffer(Caffe::Get().context, CL_MEM_READ_WRITE, bottom[0]->count() * 4, NULL, NULL);
    
    // OPENCL_CHECK(clEnqueueWriteBuffer(Caffe::Get().commandQueue, gpu_ptr_, CL_TRUE, 0, bottom[0]->count() * 4, bottom[0]->cpu_data(), 0, NULL, NULL));

    // clEnqueueReadBuffer(Caffe::Get().commandQueue, (cl_mem)bottom_data, CL_TRUE, 0, bottom[0]->count() * 4, cpu_ptr_, 0, NULL, NULL);
    // clEnqueueReadBuffer(Caffe::Get().commandQueue, (cl_mem)weight, CL_TRUE, 0, this->blobs_[0]->count() * 4, cpu_ptr_1, 0, NULL, NULL);

    // for (int i = 0; i < 100; ++i) {
    //   std::cout << cpu_ptr_[i] << " ";
    // }

    // for (int i = 0; i < 100; ++i) {
    //   std::cout << cpu_ptr_1[i] << " ";
    // }



    int v_B_off = bottom[i]->count();
    int v_C_off = top[i]->count();
    int v_imsi_0 = bottom[i]->shape(2);
    int v_imso_0 = top[i]->shape(2);
    int v_imsi_1 = bottom[i]->shape(3);
    int v_imso_1 = top[i]->shape(2);
    int v_imsi = bottom[i]->shape(2) * bottom[i]->shape(3);
    int v_imso = top[i]->shape(2) * top[i]->shape(3);
    int v_k_0 = kernel_shape_data[0];
    int v_k_1 = kernel_shape_data[1];
    int v_p_0 = pad_data[0];
    int v_p_1 = pad_data[1];
    int v_s_0 = stride_data[0];
    int v_s_1 = stride_data[1];
    int v_d_0 = dilation_data[0];
    int v_d_1 = dilation_data[1];
    int v_fin = bottom[i]->shape(1);
    int v_fout = top[i]->shape(1);
    int MG = top[i]->shape(1);
    int M = top[i]->shape(1) / conv_param.group();
    int N = top[i]->shape(2) * top[i]->shape(3);
    int KG = bottom[i]->shape(1)*kernel_shape_data[0] * kernel_shape_data[1];
    int K = bottom[i]->shape(1)*kernel_shape_data[0] * kernel_shape_data[1]/conv_param.group();



    std::cout << "v_B_off" << v_B_off << std::endl;
    std::cout << "v_C_off" << v_C_off << std::endl;
    std::cout << "v_imsi_0" << v_imsi_0 << std::endl;
    std::cout << "v_imso_0" << v_imso_0 << std::endl;
    std::cout << "v_imsi_1" << v_imsi_1 << std::endl;
    std::cout << "v_imso_1" << v_imso_1 << std::endl;
    std::cout << "v_imsi" << v_imsi << std::endl;
    std::cout << "v_imso" << v_imso << std::endl;
    std::cout << "v_k_0" << v_k_0 << std::endl;
    std::cout << "v_k_1" << v_k_1 << std::endl;
    std::cout << "v_p_0" << v_p_0 << std::endl;
    std::cout << "v_p_1" << v_p_1 << std::endl;
    std::cout << "v_s_0" << v_s_0 << std::endl;
    std::cout << "v_s_1" << v_s_1 << std::endl;
    std::cout << "v_d_0" << v_d_0 << std::endl;
    std::cout << "v_d_1" << v_d_1 << std::endl;
    std::cout << "v_fin" << v_fin << std::endl;
    std::cout << "v_fout" << v_fout << std::endl;
    std::cout << "MG" << MG << std::endl;
    std::cout << "M" << M << std::endl;
    std::cout << "N" << N << std::endl;
    std::cout << "KG" << KG << std::endl;
    std::cout << "K" << K << std::endl;



    OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bottom_data));  
    OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&weight));  
    OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&top_data));  
    OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&v_B_off));  
    OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&v_C_off));  
    OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&v_imsi_0));  
    OPENCL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&v_imso_0));  
    OPENCL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&v_imsi_1));  
    OPENCL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&v_imso_1));  
    OPENCL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&v_imsi));  
    OPENCL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&v_imso));  
    OPENCL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_int), (void *)&v_k_0));  
    OPENCL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_int), (void *)&v_k_1));  
    OPENCL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_int), (void *)&v_p_0));  
    OPENCL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_int), (void *)&v_p_1));  
    OPENCL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_int), (void *)&v_s_0));  
    OPENCL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_int), (void *)&v_s_1));  
    OPENCL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_int), (void *)&v_d_0));  
    OPENCL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_int), (void *)&v_d_1));  
    OPENCL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_int), (void *)&v_fin));  
    OPENCL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_int), (void *)&v_fout));  
    OPENCL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_int), (void *)&MG));  
    OPENCL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_int), (void *)&M));  
    OPENCL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_int), (void *)&N));  
    OPENCL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_int), (void *)&KG));  
    OPENCL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_int), (void *)&K));  
    

    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();

      OPENCL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_mem), (void *)&bias));
    }


    size_t* local_size = new size_t[3];
    local_size[0] = static_cast<size_t>(16);
    local_size[1] = static_cast<size_t>(16);
    local_size[2] = static_cast<size_t>(1);

    size_t* global_size = new size_t[3];
    global_size[0] = static_cast<size_t>(((N - 1) / 64 + 1)*32);
    global_size[1] = static_cast<size_t>(((M - 1) / 64 + 1)*32);
    global_size[2] = static_cast<size_t>(bottom[i]->shape()[0] * 1);

    std::cout << "The batch size is" << global_size[2] << "and the totoal is " << global_size[0] << " and " << global_size[1] << std::endl;

    OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL));  

  //   float* cpu_ptr_2 = (float*)malloc(top[0]->count() * 4);
  //   clEnqueueReadBuffer(Caffe::Get().commandQueue, output_gpu, CL_TRUE, 0, top[0]->count() * 4, cpu_ptr_2, 0, NULL, NULL);

  //   for (int i = 0; i < 100; ++i) {
  //     std::cout << cpu_ptr_2[i] << " ";
  //   }


  }
      std::cout << "--------------------- ";

  for (int i = 0; i < 100; ++i) {
      std::cout << top[0]->cpu_data()[i] << " ";
    }


  //   for (int i = 0; i < 100; ++i) {
  //     std::cout << bottom[0]->cpu_data()[i] << " ";
  //   }

  //   exit(0);





}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}


// #ifdef CPU_ONLY
// STUB_GPU(ConvolutionLayer);
// #endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
