/**
 * Original version of this file is provided in https://github.com/sh1r0/caffe,
 * which is part of https://github.com/sh1r0/caffe-android-lib.
 * Thanks to github user "sh1r0" for sharing this.
 */

#include <jni.h>

// #include "caffe_mobile.hpp"
#include "caffe/caffe.hpp"

#ifdef __cplusplus
extern "C" {
#endif


static caffe::Net<float> *net_;

JNIEXPORT void JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_setBlasThreadNum(JNIEnv *env, jobject instance,
                                                            jint numThreads) {
  openblas_set_num_threads(numThreads);
}

JNIEXPORT jboolean JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_loadModel(JNIEnv *env, jobject instance,
                                                     jstring modelPath_, jstring weightPath_) {

    

    jboolean ret = true;

    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);
    const char *weightPath = env->GetStringUTFChars(weightPath_, 0);


    net_ = new caffe::Net<float>(modelPath, caffe::TEST);
    net_->CopyTrainedLayersFrom(weightPath);

    env->ReleaseStringUTFChars(modelPath_, modelPath);
    env->ReleaseStringUTFChars(weightPath_, weightPath);
    return ret;


    // LOG(INFO) << "debug twice1 1";

    // caffe::Caffe::Get();
    // LOG(INFO) << "debug twice1 1.1";

    // caffe::Caffe::Get();
    // LOG(INFO) << "debug twice1 1.2";
    // caffe::Caffe::Get();

    // if (caffe::CaffeMobile::get(modelPath, weightPath) == NULL) {
    //     ret = false;
    // }
    // LOG(INFO) << "debug twice1 2";

    
}


JNIEXPORT jboolean JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_copyparam(JNIEnv *env, jobject instance, jstring weightPath_) {
    jboolean ret = true;

    // const char *weightPath = env->GetStringUTFChars(weightPath_, 0);

    // caffe::CaffeMobile::get()->net_->CopyTrainedLayersFrom(weightPath);

    // env->ReleaseStringUTFChars(weightPath_, weightPath);
    return ret;
}


JNIEXPORT jfloatArray JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_predict(JNIEnv *env, jobject instance,
                                                   jbyteArray jrgba, jint jchannels, jfloatArray jmean) {
  uint8_t *rgba = NULL;
  // Get matrix pointer
  if (NULL != jrgba) {
    rgba = (uint8_t *)env->GetByteArrayElements(jrgba, 0);
  } else {
    LOG(ERROR) << "caffe-jni predict(): invalid args: jrgba(NULL)";
    return NULL;
  }
  std::vector<float> mean;
  if (NULL != jmean) {
    float * mean_arr = (float *)env->GetFloatArrayElements(jmean, 0);
    int mean_size = env->GetArrayLength(jmean);
    mean.assign(mean_arr, mean_arr+mean_size);
  } else {
    LOG(INFO) << "caffe-jni predict(): args: jmean(NULL)";
  }
  // Predict
  

  // caffe::CaffeMobile *caffe_mobile = caffe::CaffeMobile::get();
    

  
  // if (NULL == caffe_mobile) {
  //   LOG(ERROR) << "caffe-jni predict(): CaffeMobile failed to initialize";
  //   return NULL;  // not initialized
  // }
  int rgba_len = env->GetArrayLength(jrgba);
  // if (rgba_len != jchannels * caffe_mobile->input_width() * caffe_mobile->input_height()) {
  //   LOG(WARNING) << "caffe-jni predict(): invalid rgba length(" << rgba_len << ") expect(" <<
  //                   jchannels * caffe_mobile->input_width() * caffe_mobile->input_height() << ")";
  //   return NULL;  // not initialized
  // }
  std::vector<float> predict;



  caffe::Blob<float> *input_layer = net_->input_blobs()[0];
  float *input_data = input_layer->mutable_cpu_data();
  size_t plane_size = input_layer->height() * input_layer->width();

  int input_channels_ = input_layer->channels();


  if (input_channels_ == 1 && jchannels == 1) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = static_cast<float>(rgba[i]);  // Gray
      if (mean.size() == 1) {
        input_data[i] -= mean[0];
      }
    }
  } else if (input_channels_ == 1 && jchannels == 4) {
    for (size_t i = 0; i < plane_size; i++) {
      input_data[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]; // RGB2Gray
      if (mean.size() == 1) {
        input_data[i] -= mean[0];
      }
    }
  } else if (input_channels_ == 3 && jchannels == 4) {
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
    // return false;
  }
  // Do Inference


  net_->Forward();

  

  // timer.Stop();
  // LOG(INFO) << "Inference use " << timer.MilliSeconds() << " ms.";
  caffe::Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->count();
  predict.assign(begin, end);




  // if (!caffe_mobile->predictImage(rgba, jchannels, mean, predict)) {
  //   LOG(WARNING) << "caffe-jni predict(): CaffeMobile failed to predict";
  //   return NULL; // predict error
  // }


  // Handle result
  jfloatArray result = env->NewFloatArray(predict.size());
  if (result == NULL) {
    return NULL; // out of memory error thrown
  }
  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, predict.size(), predict.data());
  return result;
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputChannels(JNIEnv *env, jobject instance) {
  // Predict
  // caffe::CaffeMobile *caffe_mobile = caffe::CaffeMobile::get();
  // if (NULL == caffe_mobile) {
  //     return -1;  // not initialized
  // }
  // return caffe_mobile->input_channels();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputWidth(JNIEnv *env, jobject instance) {
  // Predict
  // caffe::CaffeMobile *caffe_mobile = caffe::CaffeMobile::get();
  // if (NULL == caffe_mobile) {
  //     return -1;  // not initialized
  // }
  // return caffe_mobile->input_width();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputHeight(JNIEnv *env, jobject instance) {
  // Predict
  // caffe::CaffeMobile *caffe_mobile = caffe::CaffeMobile::get();
  // if (NULL == caffe_mobile) {
  //   return -1;  // not initialized
  // }
  // return caffe_mobile->input_height();
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }
  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
