/**
 * Original version of this file is provided in https://github.com/sh1r0/caffe,
 * which is part of https://github.com/sh1r0/caffe-android-lib.
 * Thanks to github user "sh1r0" for sharing this.
 */

#ifndef JNI_CAFFE_MOBILE_HPP_
#define JNI_CAFFE_MOBILE_HPP_

#include <string>
#include <vector>
#include "caffe/caffe.hpp"

namespace caffe {

/**
 * @brief Wrap caffe::net to a simpler interface; singleton
 */
class CaffeMobilef {
public:
  /**
   * @brief Destructor
   */
  ~CaffeMobilef();

  /**
   * @brief Get the CaffeMobilef singleton from the param file (*.prototxt)
   * and the tained file (*.caffemodel)
   * @return NULL: failed; not NULL: succeeded
   */
  static CaffeMobilef *get(const std::string &param_file,
                          const std::string &trained_file, int engine);

  /**
   * @brief Get the exist CaffeMobilef singleton pointer
   * @return NULL: no exist; not NULL: exist
   */
  static CaffeMobilef *get();

  void benchmark();

  /**
   * @brief Use loaded model to classify a Image
   * @param rgba: Grayscale(1 channel) or BGR(3 channels) pixels array
   */
  bool predictImage(const uint8_t *rgba,
                    int channels,
                    const std::vector<float> &mean,
                    std::vector<float> &result);

  int input_channels() {
    return input_channels_;
  }

  int input_width() {
    return input_width_;
  }

  int input_height() {
      return input_height_;
  }
private:
  /**
   * @brief Construct a caffe net from the param file (*.prototxt)
   * and the tained file (*.caffemodel)
   */
  CaffeMobilef(const string &param_file, const string &trained_file, int engine);


  /// @brief
  static CaffeMobilef *caffe_mobilef_;
  /// @brief
  // Net<float> *net_;
  shared_ptr<Net<float>> net_;
  /// @brief
  int input_channels_;
  int input_width_;
  int input_height_;
};




class CaffeMobileh {
public:
  /**
   * @brief Destructor
   */
  ~CaffeMobileh();

  /**
   * @brief Get the CaffeMobileh singleton from the param file (*.prototxt)
   * and the tained file (*.caffemodel)
   * @return NULL: failed; not NULL: succeeded
   */
  static CaffeMobileh *get(const std::string &param_file,
                          const std::string &trained_file, int engine);

  /**
   * @brief Get the exist CaffeMobileh singleton pointer
   * @return NULL: no exist; not NULL: exist
   */
  static CaffeMobileh *get();

  void benchmark();

  
  /**
   * @brief Use loaded model to classify a Image
   * @param rgba: Grayscale(1 channel) or BGR(3 channels) pixels array
   */
  bool predictImage(const uint8_t *rgba,
                    int channels,
                    const std::vector<float> &mean,
                    std::vector<float> &result);

  int input_channels() {
    return input_channels_;
  }

  int input_width() {
    return input_width_;
  }

  int input_height() {
      return input_height_;
  }
private:
  /**
   * @brief Construct a caffe net from the param file (*.prototxt)
   * and the tained file (*.caffemodel)
   */
  CaffeMobileh(const string &param_file, const string &trained_file, int engine);


  /// @brief
  static CaffeMobileh *caffe_mobileh_;
  /// @brief
  // Net<half> *net_;
  shared_ptr<Net<half>> net_;
  /// @brief
  int input_channels_;
  int input_width_;
  int input_height_;
};

} // namespace caffe

#endif
