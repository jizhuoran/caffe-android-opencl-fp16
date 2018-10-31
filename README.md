Caffe on Mobile Devices (Still under developing)
==================

Optimized (for memory usage, speed and enegry efficiency) Caffe with OpenCL supporting for less powerful devices such as mobile phone (NO_BACKWARD, NO_BOOST, NO_HDF5, NO_LEVELDB). 

# Features

1. <span style="color:red"> OpenCL supporting (mobile GPU) </span>
2. As few dependencies as possible
3. Optimized memory usage (under developing)
4. Forward Only (backward will be added later)

## Layers with OpenCL:
1. Convolution Layer (libdnn)
2. Deconvolution Layer (libdnn)
3. ReLU Layer

# For Android
Release Soon! Actually, I do not have a android phone now. LoL


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
$ cmake .. -DTOOLS
$ make -j 4
```

# Thanks

 - Based on https://github.com/solrex/caffe-mobile.git
 - Inspired by https://github.com/BVLC/caffe/tree/opencl
 - Android JNI code based on https://github.com/sh1r0/caffe