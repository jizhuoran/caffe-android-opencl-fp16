#ifndef CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_

#include "caffe/common.hpp"
// #include "caffe/backend/device.hpp"
#include "caffe/opencl/caffe_opencl.hpp"
#include "caffe/opencl/vptr.hpp"


namespace caffe {

#ifdef USE_OPENCL

class OclDevice : public Device {
 public:
  explicit OclDevice(uint32_t id, uint32_t list_id);
  ~OclDevice();

  const char* clGetErrorString(cl_int error);

  virtual void Init();
  // virtual bool CheckCapability(DeviceCapability cap);
  virtual bool CheckVendor(string vendor);
  virtual bool CheckType(string type);
  virtual void SwitchQueue(uint32_t id);
  // virtual void get_threads(const vector<size_t>* work_size, vector<size_t>* group, vector<size_t>* local, DeviceKernel* kernel, bool auto_select);
  virtual void FinishQueues();
  virtual uint32_t num_queues();
  virtual bool is_host_unified();
  bool is_beignet();
  virtual string name();
  // virtual shared_ptr<DeviceProgram> CreateProgram();

  virtual void unlock_buffer(int32_t* lock_id);

  virtual void MallocMemHost(uint32_t size, void** ptr);
  virtual void FreeMemHost(void* ptr);
  virtual vptr<void> MallocMemDevice(uint32_t size, void** ptr, bool zero_copy);
  virtual void FreeMemDevice(vptr<void> ptr);
  virtual bool CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                             uint32_t size);

  void ocl_null_kernel(float arg, cl_event* event);

  virtual void memcpy(const uint32_t n, vptr<const void> x, vptr<void> y);
  virtual void memcpy(const uint32_t n, const void* x, vptr<void> y);
  virtual void memcpy(const uint32_t n, vptr<const void> x, void* y);


};


#endif  // USE_OPENCL

}  // namespace caffe

#endif  // CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_
