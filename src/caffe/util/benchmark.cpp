#ifdef USE_BOOST
#include <boost/date_time/posix_time/posix_time.hpp>
#endif

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    //TODOTODOO
    // CUDA_CHECK(cudaEventDestroy(start_gpu_));
    // CUDA_CHECK(cudaEventDestroy(stop_gpu_));
#else
    NO_GPU;
#endif
  }
}

void Timer::Start() {
  if (!running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_OPENCL
      clReleaseEvent(start_gpu_cl_);


      cl_int ret;

      cl_kernel kernel = clCreateKernel(Caffe::Get().math_program, "null_kernel_float", &ret);
      OPENCL_CHECK(ret);

      int arg = 0;
      OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&arg));  

      size_t global_size = 1;
  
      OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &global_size, 0, NULL, &start_gpu_cl_));  
      clWaitForEvents(1, &start_gpu_cl_);
      
      clFinish(Caffe::Get().commandQueue);

#else
      NO_GPU;
#endif
    } else {
#ifdef USE_BOOST
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
#else
      gettimeofday(&start_cpu_, NULL);
#endif
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_OPENCL
        

      clWaitForEvents(1, &stop_gpu_cl_);
      clReleaseEvent(stop_gpu_cl_);


      cl_int ret;

      cl_kernel kernel = clCreateKernel(Caffe::Get().math_program, "null_kernel_float", &ret);
      OPENCL_CHECK(ret);

      int arg = 0;
      OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&arg));  

      size_t global_size = 1;
  
      OPENCL_CHECK(clEnqueueNDRangeKernel(Caffe::Get().commandQueue, kernel, 1, NULL, &global_size, &global_size, 0, NULL, &stop_gpu_cl_));  
      
      clWaitForEvents(1, &stop_gpu_cl_);

      clFinish(Caffe::Get().commandQueue);

#else
      NO_GPU;
#endif
    } else {
#ifdef USE_BOOST
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
#else
      gettimeofday(&stop_cpu_, NULL);
#endif
    }
    running_ = false;
  }
}


float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_OPENCL
      
      cl_ulong startTime = 0, stopTime = 0;
      OPENCL_CHECK(clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
          sizeof startTime, &startTime, NULL));
      OPENCL_CHECK(clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
          sizeof stopTime, &stopTime, NULL));
      double ms = static_cast<double>(stopTime - startTime) / 1000.0;
      elapsed_microseconds_ = static_cast<float>(ms);
    
#else
      NO_GPU;
#endif
  } else {
#ifdef USE_BOOST
    elapsed_microseconds_ = (stop_cpu_ - start_cpu_).total_microseconds();
#else
    elapsed_microseconds_ = (stop_cpu_.tv_sec - start_cpu_.tv_sec)*1000000
    		+ (stop_cpu_.tv_usec - start_cpu_.tv_usec);
#endif
  }
  return elapsed_microseconds_;
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_OPENCL
      
      cl_ulong startTime = 0, stopTime = 0;
      clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
          sizeof startTime, &startTime, NULL);
      clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
          sizeof stopTime, &stopTime, NULL);
      double ms = static_cast<double>(stopTime - startTime) / 1000000.0;
      elapsed_milliseconds_ = static_cast<float>(ms);

#else
      NO_GPU;
#endif
  } else {
#ifdef USE_BOOST
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
#else
    elapsed_microseconds_ = (stop_cpu_.tv_sec - start_cpu_.tv_sec)*1000
    		+ (stop_cpu_.tv_usec - start_cpu_.tv_usec)/1000.0;
#endif
  }
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_OPENCL
    start_gpu_cl_ = 0;
    stop_gpu_cl_ = 0;
#else
      NO_GPU;
#endif
    }
    initted_ = true;
  }
}

CPUTimer::CPUTimer() {
  this->initted_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

void CPUTimer::Start() {
  if (!running()) {
#ifdef USE_BOOST
    this->start_cpu_ = boost::posix_time::microsec_clock::local_time();
#else
    gettimeofday(&start_cpu_, NULL);
#endif
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
#ifdef USE_BOOST
    this->stop_cpu_ = boost::posix_time::microsec_clock::local_time();
#else
    gettimeofday(&stop_cpu_, NULL);
#endif
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
#ifdef USE_BOOST
  this->elapsed_milliseconds_ = (this->stop_cpu_ -
                                this->start_cpu_).total_milliseconds();
#else
    elapsed_milliseconds_ = (stop_cpu_.tv_sec - start_cpu_.tv_sec)*1000
    		+ (stop_cpu_.tv_usec - start_cpu_.tv_usec)/1000.0;
#endif
  return this->elapsed_milliseconds_;
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
#ifdef USE_BOOST
  this->elapsed_microseconds_ = (this->stop_cpu_ -
                                this->start_cpu_).total_microseconds();
#else
    elapsed_microseconds_ = (stop_cpu_.tv_sec - start_cpu_.tv_sec)*1000000
    		+ (stop_cpu_.tv_usec - start_cpu_.tv_usec);
#endif
  return this->elapsed_microseconds_;
}

}  // namespace caffe
