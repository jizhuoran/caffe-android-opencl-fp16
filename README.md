Caffe on Mobile Devices
==================

Optimized (for memory usage, speed and enegry efficiency) Caffe with OpenCL supporting for less powerful devices such as mobile phone (NO_BACKWARD, NO_BOOST, NO_HDF5, NO_LEVELDB). 

# Still under developed

# For Android (Release Soon)


# For Ubuntu

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

 - Based on https://github.com/BVLC/caffe
 - Inspired by https://github.com/chyh1990/caffe-compact
 - Android JNI code based on https://github.com/sh1r0/caffe
 - Use https://github.com/Yangqing/ios-cmake
 - Use https://github.com/taka-no-me/android-cmake
 - Windows build script inspired by https://github.com/luoyetx/mini-caffe/tree/master/android
