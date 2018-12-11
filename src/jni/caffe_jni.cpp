/**
 * Original version of this file is provided in https://github.com/sh1r0/caffe,
 * which is part of https://github.com/sh1r0/caffe-android-lib.
 * Thanks to github user "sh1r0" for sharing this.
 */

#include <jni.h>

#include "caffe_mobile.hpp"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_setBlasThreadNum(JNIEnv *env, jobject instance,
                                                            jint numThreads) {
  openblas_set_num_threads(numThreads);
}

JNIEXPORT jboolean JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_loadModelh(JNIEnv *env, jobject instance,
                                                     jstring modelPath_, jstring weightPath_, int engine) {
    jboolean ret = true;

    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);
    const char *weightPath = env->GetStringUTFChars(weightPath_, 0);

    LOG(INFO) << "jni debug 1";

    if (caffe::CaffeMobileh::get(modelPath, weightPath, engine) == NULL) {
        ret = false;
    }

    LOG(INFO) << "jni debug 3";


    env->ReleaseStringUTFChars(modelPath_, modelPath);
    env->ReleaseStringUTFChars(weightPath_, weightPath);
    return ret;
}


JNIEXPORT jboolean JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_loadModelf(JNIEnv *env, jobject instance,
                                                     jstring modelPath_, jstring weightPath_, int engine) {
    jboolean ret = true;

    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);
    const char *weightPath = env->GetStringUTFChars(weightPath_, 0);

    if (caffe::CaffeMobilef::get(modelPath, weightPath, engine) == NULL) {
        ret = false;
    }

    env->ReleaseStringUTFChars(modelPath_, modelPath);
    env->ReleaseStringUTFChars(weightPath_, weightPath);
    return ret;
}


JNIEXPORT jfloatArray JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_predicth(JNIEnv *env, jobject instance,
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
  

  caffe::CaffeMobileh *caffe_mobile = caffe::CaffeMobileh::get();
    

  if (NULL == caffe_mobile) {
    LOG(ERROR) << "caffe-jni predict(): CaffeMobile failed to initialize";
    return NULL;  // not initialized
  }
  int rgba_len = env->GetArrayLength(jrgba);
  if (rgba_len != jchannels * caffe_mobile->input_width() * caffe_mobile->input_height()) {
    LOG(WARNING) << "caffe-jni predict(): invalid rgba length(" << rgba_len << ") expect(" <<
                    jchannels * caffe_mobile->input_width() * caffe_mobile->input_height() << ")";
    return NULL;  // not initialized
  }
  std::vector<float> predict;


  if (!caffe_mobile->predictImage(rgba, jchannels, mean, predict)) {
    LOG(WARNING) << "caffe-jni predict(): CaffeMobile failed to predict";
    return NULL; // predict error
  }


  // Handle result
  jfloatArray result = env->NewFloatArray(predict.size());
  if (result == NULL) {
    return NULL; // out of memory error thrown
  }
  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, predict.size(), predict.data());
  return result;
}


JNIEXPORT jfloatArray JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_predictf(JNIEnv *env, jobject instance,
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
  

  caffe::CaffeMobilef *caffe_mobile = caffe::CaffeMobilef::get();
    

  if (NULL == caffe_mobile) {
    LOG(ERROR) << "caffe-jni predict(): CaffeMobile failed to initialize";
    return NULL;  // not initialized
  }
  int rgba_len = env->GetArrayLength(jrgba);
  if (rgba_len != jchannels * caffe_mobile->input_width() * caffe_mobile->input_height()) {
    LOG(WARNING) << "caffe-jni predict(): invalid rgba length(" << rgba_len << ") expect(" <<
                    jchannels * caffe_mobile->input_width() * caffe_mobile->input_height() << ")";
    return NULL;  // not initialized
  }
  std::vector<float> predict;


  if (!caffe_mobile->predictImage(rgba, jchannels, mean, predict)) {
    LOG(WARNING) << "caffe-jni predict(): CaffeMobile failed to predict";
    return NULL; // predict error
  }


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
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputChannelsh(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeMobileh *caffe_mobile = caffe::CaffeMobileh::get();
  if (NULL == caffe_mobile) {
      return -1;  // not initialized
  }
  return caffe_mobile->input_channels();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputChannelsf(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeMobilef *caffe_mobile = caffe::CaffeMobilef::get();
  if (NULL == caffe_mobile) {
      return -1;  // not initialized
  }
  return caffe_mobile->input_channels();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputWidthh(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeMobileh *caffe_mobile = caffe::CaffeMobileh::get();
  if (NULL == caffe_mobile) {
      return -1;  // not initialized
  }
  return caffe_mobile->input_width();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputWidthf(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeMobilef *caffe_mobile = caffe::CaffeMobilef::get();
  if (NULL == caffe_mobile) {
      return -1;  // not initialized
  }
  return caffe_mobile->input_width();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputHeighth(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeMobileh *caffe_mobile = caffe::CaffeMobileh::get();
  if (NULL == caffe_mobile) {
    return -1;  // not initialized
  }
  return caffe_mobile->input_height();
}

JNIEXPORT jint JNICALL
Java_com_example_gsq_caffe_1android_1project_CaffeMobile_inputHeightf(JNIEnv *env, jobject instance) {
  // Predict
  caffe::CaffeMobilef *caffe_mobile = caffe::CaffeMobilef::get();
  if (NULL == caffe_mobile) {
    return -1;  // not initialized
  }
  return caffe_mobile->input_height();
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