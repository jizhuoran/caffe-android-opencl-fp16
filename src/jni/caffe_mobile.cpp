/**
 * Original version of this file is provided in https://github.com/sh1r0/caffe,
 * which is part of https://github.com/sh1r0/caffe-android-lib.
 * Thanks to github user "sh1r0" for sharing this.
 */

#include "caffe_mobile.hpp"

namespace caffe {

CaffeMobilef *CaffeMobilef::caffe_mobilef_ = NULL;

CaffeMobilef *CaffeMobilef::get() {
  return caffe_mobilef_;
}

CaffeMobilef *CaffeMobilef::get(const string &param_file,
                              const string &trained_file, int engine) {
  if (!caffe_mobilef_) {
    try {
      caffe_mobilef_ = new CaffeMobilef(param_file, trained_file, engine);
    } catch (std::invalid_argument &e) {
      // TODO
    }
  }
  return caffe_mobilef_;
}

CaffeMobilef::CaffeMobilef(const string &param_file, const string &trained_file, int engine) {
  // Load Caffe model
  
  if (engine == 0) {
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
  }

  CPUTimer timer;
  timer.Start();
  
  net_.reset(new Net<float>(param_file, caffe::TEST));


  if (net_ == NULL) {
    throw std::invalid_argument("Invalid arg: param_file=" + param_file);
  }


  net_->CopyTrainedLayersFrom(trained_file);
  timer.Stop();
  LOG(INFO) << "Load (" << param_file << "," << trained_file << "), time:"
            << timer.MilliSeconds() << " ms.";

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  // Get input_layer info
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_channels_ = input_layer->channels();
  CHECK(input_channels_ == 3 || input_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_width_  = input_layer->width();
  input_height_ = input_layer->height();
}

CaffeMobilef::~CaffeMobilef() {
  // delete net_;
  net_.reset();
}

bool CaffeMobilef::predictImage(const uint8_t* rgba,
                               int channels,
                               const std::vector<float> &mean,
                               std::vector<float> &result) {



  if ((rgba == NULL) || net_ == NULL) {
    LOG(ERROR) << "Invalid arguments: rgba=" << rgba
        << ",net_=" << net_;
    return false;
  }


  // CPUTimer timer;
  // timer.Start();

  // Write input
  Blob<float> *input_layer = net_->input_blobs()[0];
  float *input_data = input_layer->mutable_cpu_data();
  size_t plane_size = input_height() * input_width();


  if (input_channels() == 1 && channels == 1) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = static_cast<float>(rgba[i]);  // Gray
      if (mean.size() == 1) {
        input_data[i] -= mean[0];
      }
    }
  } else if (input_channels() == 1 && channels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]; // RGB2Gray
      if (mean.size() == 1) {
        input_data[i] -= mean[0];
      }
    }
  } else if (input_channels() == 3 && channels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = static_cast<float>(rgba[i * 4 + 2]);                   // B
      input_data[plane_size + i] = static_cast<float>(rgba[i * 4 + 1]);      // G
      input_data[2 * plane_size + i] = static_cast<float>(rgba[i * 4]);      // R
      // Alpha is discarded
      if (mean.size() == 3) {
        input_data[i] -= mean[0];
        input_data[plane_size + i] -= mean[1];
        input_data[2 * plane_size + i] -= mean[2];
      }
    }
  } else {
    LOG(ERROR) << "image_channels input_channels not match.";
    return false;
  }
  // Do Inference

    
  LOG(ERROR) << "Before this";

  net_->Forward();

  LOG(ERROR) << "After this";

  

  // timer.Stop();
  // LOG(INFO) << "Inference use " << timer.MilliSeconds() << " ms.";
  Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->count();
  result.assign(begin, end);
  return true;
}







CaffeMobileh *CaffeMobileh::caffe_mobileh_ = NULL;

CaffeMobileh *CaffeMobileh::get() {
  return caffe_mobileh_;
}

CaffeMobileh *CaffeMobileh::get(const string &param_file,
                              const string &trained_file, int engine) {
  if (!caffe_mobileh_) {
    try {
      caffe_mobileh_ = new CaffeMobileh(param_file, trained_file, engine);
    } catch (std::invalid_argument &e) {
      // TODO
    }
  }
  return caffe_mobileh_;
}

CaffeMobileh::CaffeMobileh(const string &param_file, const string &trained_file, int engine) {
  // Load Caffe model
  
  if (engine == 0) {
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
  }

  CPUTimer timer;
  timer.Start();
  
  net_.reset(new Net<half>(param_file, caffe::TEST));


  if (net_ == NULL) {
    throw std::invalid_argument("Invalid arg: param_file=" + param_file);
  }


  net_->CopyTrainedLayersFrom(trained_file);
  timer.Stop();
  LOG(INFO) << "Load (" << param_file << "," << trained_file << "), time:"
            << timer.MilliSeconds() << " ms.";

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  // Get input_layer info

  LOG(INFO) << "jni debug 1.5";

  vector<Blob<half>*> net_input_blobs_ = net_->input_blobs();
 
  LOG(INFO) << "jni debug 1.55";

  Blob<half> *input_layer = net_input_blobs_[0];
  
  LOG(INFO) << "jni debug 1.6";


  input_channels_ = input_layer->channels();
  CHECK(input_channels_ == 3 || input_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";

  LOG(INFO) << "jni debug 1.7";


  input_width_  = input_layer->width();
  input_height_ = input_layer->height();

  LOG(INFO) << "jni debug 2";

}

CaffeMobileh::~CaffeMobileh() {
  // delete net_;
  net_.reset();
}

bool CaffeMobileh::predictImage(const uint8_t* rgba,
                               int channels,
                               const std::vector<float> &mean,
                               std::vector<float> &result) {



  if ((rgba == NULL) || net_ == NULL) {
    LOG(ERROR) << "Invalid arguments: rgba=" << rgba
        << ",net_=" << net_;
    return false;
  }


  // CPUTimer timer;
  // timer.Start();

  // Write input
  Blob<half> *input_layer = net_->input_blobs()[0];

  float* input_convertor = (float* )malloc(input_layer->count() * sizeof(float));

  size_t plane_size = input_height() * input_width();


  if (input_channels() == 1 && channels == 1) {
    for (size_t i = 0; i < plane_size; i++) {
      input_convertor[i] = static_cast<float>(rgba[i]);  // Gray
      if (mean.size() == 1) {
        input_convertor[i] -= mean[0];
      }
    }
  } else if (input_channels() == 1 && channels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_convertor[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]; // RGB2Gray
      if (mean.size() == 1) {
        input_convertor[i] -= mean[0];
      }
    }
  } else if (input_channels() == 3 && channels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_convertor[i] = static_cast<float>(rgba[i * 4 + 2]);                   // B
      input_convertor[plane_size + i] = static_cast<float>(rgba[i * 4 + 1]);      // G
      input_convertor[2 * plane_size + i] = static_cast<float>(rgba[i * 4]);      // R
      // Alpha is discarded
      if (mean.size() == 3) {
        input_convertor[i] -= mean[0];
        input_convertor[plane_size + i] -= mean[1];
        input_convertor[2 * plane_size + i] -= mean[2];
      }
    }
  } else {
    LOG(ERROR) << "image_channels input_channels not match.";
    return false;
  }
  // Do Inference

  float2half(input_layer->count(), input_convertor, input_layer->mutable_cpu_data());
  free(input_convertor);
    
  net_->Forward();

  

  // timer.Stop();
  // LOG(INFO) << "Inference use " << timer.MilliSeconds() << " ms.";
  Blob<half> *output_layer = net_->output_blobs()[0];

  result.resize(output_layer->count());

  half2float(output_layer->count(), output_layer->cpu_data(), &result[0]);

  // const float *begin = output_layer->cpu_data();
  // const float *end = begin + output_layer->count();
  // result.assign(begin, end);
  return true;
}

} // namespace caffe
