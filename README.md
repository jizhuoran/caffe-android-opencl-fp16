Caffe on Mobile Devices (Still under developing)
==================

Optimized (for memory usage, speed and enegry efficiency) Caffe with OpenCL supporting for less powerful devices such as mobile phone (NO_BACKWARD, NO_BOOST, NO_HDF5, NO_LEVELDB). 

**I am developing this project. You can watch this project if you want to get the latest news**

# Features
- [x] **double data type is removied, scala data store in float, others store in Dtype (half or float)**
- [x] **OpenCL supporting (mobile GPU)** (Partially finished)
- [x] **FP16 Inference support**
  - [x] BatchNorm shifting to avoid overflow and underflow
  - [x] All Layer support
  - [x] FP16 caffemodel load and save
  - [x] model convertor (From FP32 to FP16)
- [x] As few dependencies as possible (Protobuf, OpenBLAS, CLBlast)
- [x] Optimized memory usage
- [x] Forward Only (I just noticed that in the original implementation, forward only also do unnecessary copy)
- [x] Zero Copy (Shared memory between Host and GPU)
- [ ] Backward (I change my mind, Pure Forward Only library will be kept)


## Peak Memory Usage Reduction

### Testing on going, I am waitting for a device with large enough memory to get the peak memory usage with the memory usage optimization.

## Layers with OpenCL:

 - [x] Convolution Layer (libdnn)
 - [x] Deconvolution Layer (libdnn)
 - [x] Batch Norm Layer (with shift)
 - [x] Others

# On-going

1. Modify the test cases to support half testing
2. Check unnecessary data copy in Forward Only mode
4. Tune for android devices
5. Change the structure of the project (move test out of the src)
6. Refactor: OpenCL kernls launch method, redundant code in math_fuctions_cl.cpp
7. Doc


# For Android

**The project is test on:**. 

- [x] Snapdragon 820 development board
- [x] HUAWEI P9
- [x] Hikey 970


## Build libcaffe.so

```
$ modify the NDK_HOME path in ./tools/build_android.sh to your NDK_HOME
$ modify the DEVICE_OPENCL_DIR path in ./tools/build_android.sh to the directory contains include/CL/cl.h and lib64/libOpencl.so
$ ./tools/build_android.sh
$ (You may want to choose your own make -j)

```
## Build Android App with Android Studio



### Make a directory in your devices.

```
$ adb shell
$ cd /sdcard/caffe

```

### Similar as Caffe, you need the proto-file and weights. Follow the below instructions to push the needed file to your devices

```
$ adb push $CAFFE/examples/style_transfer/style.protobin
$ adb push $CAFFE/examples/style_transfer/a1.caffemodel
$ adb push $CAFFE/examples/style_transfer/HKU.jpg

```
### Load the Android studio project inside the $CAFFE_MOBILE/examples/android/android-caffe/ folder, and run it on your connected device.

# For Ubuntu

## Test Environment

CPU: Intel(R) Xeon(R) CPU E5-2630 v4  
GPU NVIDIA 2080
OS: ubuntu 16.04  
OpenCL Version: 1.2  
C++ Version: 5.4.0  

For a art style transfer neural network, reduce the single inference time from 7.9s to 2.0s (E5 to NVIDIA 2080).



## Step 1: Install dependency

```
$ sudo apt install libprotobuf-dev protobuf-compiler libatlas-dev # Ubuntu
```

## Step 2: Build Caffe-Mobile Lib with cmake

```
$ git clone https://github.com/jizhuoran/caffe-android-opencl.git
$ mkdir build
$ cd ../build
$ cmake ..
$ make -j 40
```

# Thanks

 - Based on https://github.com/solrex/caffe-mobile.git
 - Inspired by https://github.com/BVLC/caffe/tree/opencl
 - Android JNI code based on https://github.com/sh1r0/caffe
