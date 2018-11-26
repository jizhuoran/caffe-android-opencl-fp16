Caffe on Mobile Devices (Still under developing)
==================

Optimized (for memory usage, speed and enegry efficiency) Caffe with OpenCL supporting for less powerful devices such as mobile phone (NO_BACKWARD, NO_BOOST, NO_HDF5, NO_LEVELDB). 

**I am developing this project. You can watch this project if you want to get the latest news**

# Features

- [x] **OpenCL supporting (mobile GPU)** (Partially finished)
- [x] As few dependencies as possible
- [x] Optimized memory usage
- [x] Forward Only 
- [ ] Backward


## Peak Memory Usage Reduction

### Testing on going, I am waitting for a device with large enough memory to get the peak memory usage with the memory usage optimization.

## Layers with OpenCL:

 - [x] Convolution Layer (libdnn)
 - [x] Deconvolution Layer (libdnn)
 - [x] ReLU Layer
 - [ ] Matrix Multiplication
 

# For Android

**The project is test on:**. 

1. Snapdragon 835 development board
2. HUAWEI P9
3. Hikey 970 (Waitting the device)


## Build libcaffe.so

```
$ modify the NDK_HOME path in ./tools/build_android.sh to your NDK_HOME
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
$ git clone --recursive https://github.com/solrex/caffe-mobile.git
$ mkdir build
$ cd ../build
$ cmake ..
$ make -j 16
```

## Step 3: Build Caffe-bin with cmake

```
$ brew install gflags
$ cmake ..
$ make -j 4
```

# Thanks

 - Based on https://github.com/solrex/caffe-mobile.git
 - Inspired by https://github.com/BVLC/caffe/tree/opencl
 - Android JNI code based on https://github.com/sh1r0/caffe