#ifdef USE_BOOST
#include <boost/thread.hpp>
#endif
#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/opencl_kernel.hpp"


#define MAX_SOURCE_SIZE (0x1000000)

#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

#ifndef NO_CAFFE_MOBILE
#include <unistd.h>
#endif

namespace caffe {

#ifdef USE_BOOST
// Make sure each thread can have different values.
static boost::thread_specific_ptr<Caffe> thread_instance_;
#else
static Caffe *thread_instance_ = NULL;
#endif

Caffe& Caffe::Get() {
#ifdef USE_BOOST
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
#else
  if (thread_instance_ == NULL) {
      thread_instance_ = new Caffe();
  }
  return *thread_instance_;
#endif
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


void GlobalInit(int* pargc, char*** pargv) {
#ifdef NO_CAFFE_MOBILE
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
#endif
#ifdef USE_GLOG
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
#endif
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), solver_rank_(0), multiprocess_(false) {

}

Caffe::~Caffe() { }

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#elif USE_OPENCL  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : random_generator_(),
    mode_(Caffe::CPU),
    solver_count_(1), solver_rank_(0), multiprocess_(false) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  
  OPENCL_CHECK(clGetPlatformIDs(1, &platformId, &retNumPlatforms));
  OPENCL_CHECK(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices));


  cl_int ret;

  context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);
  OPENCL_CHECK(ret);

  commandQueue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &ret);
  OPENCL_CHECK(ret);

  

  std::stringstream ss;

  ss << generate_opencl_defs(false);
  ss << generate_opencl_math(false);

  build_opencl_program(ss.str(), math_program);


}

Caffe::~Caffe() {
  // if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  // if (curand_generator_) {
  //   CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  // }
}


void Caffe::build_opencl_program(std::string kernel_code, cl_program &program) {

  cl_int ret = -1;

  size_t kernel_size = kernel_code.size() + 1;

  const char* kernelSource = kernel_code.c_str();

  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernel_size, &ret); 
  OPENCL_CHECK(ret);

  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);


  if (ret != CL_SUCCESS) {
    char *buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
      LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__;
      exit(-1);
    }

    buff_erro = (char *)malloc(build_log_len);
    if (!buff_erro) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__;
        exit(-3);
    }
    
    LOG(ERROR) << "Build log: " << buff_erro;

    free(buff_erro);

    LOG(ERROR) << "clBuildProgram failed";

    exit(EXIT_FAILURE);
  }
}

/*
void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  if (Get().curand_generator_) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        LOG(ERROR) <<
            "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}





class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}
*/

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NOT_IMPLEMENT;
}

// void Caffe::DeviceQuery() {
//   NOT_IMPLEMENT;
// }


void Caffe::DeviceQuery(){

  cl_device_id device = Caffe::Get().deviceID;


  char device_string[1024];

  // CL_DEVICE_NAME
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  LOG(INFO) << "  CL_DEVICE_NAME: " << device_string;

  // CL_DEVICE_VENDOR
  clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
  LOG(INFO) << "  CL_DEVICE_VENDOR: " << device_string;

  // CL_DRIVER_VERSION
  clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
  LOG(INFO) << "  CL_DRIVER_VERSION: " << device_string;

  // CL_DEVICE_INFO
  cl_device_type type;
  clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if( type & CL_DEVICE_TYPE_CPU )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_CPU";
  if( type & CL_DEVICE_TYPE_GPU )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_GPU";
  if( type & CL_DEVICE_TYPE_ACCELERATOR )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_ACCELERATOR";
  if( type & CL_DEVICE_TYPE_DEFAULT )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_DEFAULT";

  // CL_DEVICE_MAX_COMPUTE_UNITS
  cl_uint compute_units;
  clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_COMPUTE_UNITS: " << compute_units;

  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  size_t workitem_dims;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << workitem_dims;

  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  size_t workitem_size[3];
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WORK_ITEM_SIZES:" << workitem_size[0] << workitem_size[1] << workitem_size[2];

  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  size_t workgroup_size;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WORK_GROUP_SIZE: " << workgroup_size;

  // CL_DEVICE_MAX_CLOCK_FREQUENCY
  cl_uint clock_frequency;
  clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_CLOCK_FREQUENCY:" << clock_frequency << " MHz";

  // CL_DEVICE_ADDRESS_BITS
  cl_uint addr_bits;
  clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
  LOG(INFO) << "  CL_DEVICE_ADDRESS_BITS:" << addr_bits;

  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  cl_ulong max_mem_alloc_size;
  clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_MEM_ALLOC_SIZE:" << (unsigned int)(max_mem_alloc_size / (1024 * 1024)) << "MByte";

  // CL_DEVICE_GLOBAL_MEM_SIZE
  cl_ulong mem_size;
  clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_GLOBAL_MEM_SIZE:" << (unsigned int)(mem_size / (1024 * 1024)) << "MByte";


  // CL_DEVICE_LOCAL_MEM_TYPE
  cl_device_local_mem_type local_mem_type;
  clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
  if (local_mem_type == 1) {
    LOG(INFO) << "  CL_DEVICE_LOCAL_MEM_TYPE: local";
  } else {
    LOG(INFO) << "  CL_DEVICE_LOCAL_MEM_TYPE: global";
  }

  // CL_DEVICE_LOCAL_MEM_SIZE
  clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_LOCAL_MEM_SIZE:" << (unsigned int)(mem_size / 1024) << "KByte\n";

  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:" << (unsigned int)(mem_size / 1024) << "KByte\n";

  // CL_DEVICE_QUEUE_PROPERTIES
  cl_command_queue_properties queue_properties;
  clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
  if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
    LOG(INFO) << "  CL_DEVICE_QUEUE_PROPERTIES:" << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE";
  if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
    LOG(INFO) << "  CL_DEVICE_QUEUE_PROPERTIES:" << "CL_QUEUE_PROFILING_ENABLE";

  // CL_DEVICE_IMAGE_SUPPORT
  cl_bool image_support;
  clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
  LOG(INFO) << "  CL_DEVICE_IMAGE_SUPPORT:" << image_support;

  // CL_DEVICE_MAX_READ_IMAGE_ARGS
  cl_uint max_read_image_args;
  clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_READ_IMAGE_ARGS:" << max_read_image_args;

  // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  cl_uint max_write_image_args;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:" << max_write_image_args;

  // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
  size_t szMaxDims[5];
  LOG(INFO) << "\n  CL_DEVICE_IMAGE <dim>";
  clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
  LOG(INFO) << "2D_MAX_WIDTH" << szMaxDims[0];
  clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
  LOG(INFO) << "2D_MAX_HEIGHT" << szMaxDims[1];
  clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
  LOG(INFO) << "3D_MAX_WIDTH" << szMaxDims[2];
  clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
  LOG(INFO) << "3D_MAX_HEIGHT" << szMaxDims[3];
  clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
  LOG(INFO) << "3D_MAX_DEPTH" << szMaxDims[4];

  // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
  LOG(INFO) << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>";
  cl_uint vec_width [6];
  clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
  clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
  clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
  clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
  clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
  clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
  LOG(INFO) << "CHAR" << vec_width[0] << "SHORT" << vec_width[1] << "INT" << vec_width[2] 
    << "FLOAT" << vec_width[3] << "DOUBLE" << vec_width[4];

   
}




bool Caffe::CheckDevice(const int device_id) {
  NOT_IMPLEMENT;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NOT_IMPLEMENT;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#endif  // CPU_ONLY

}  // namespace caffe
