//
//  main.m
//  metal_mac
//
//  Created by Tec GSQ on 27/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#include <iostream>
#include "caffe/caffe.hpp"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>


#define MAX_SOURCE_SIZE (0x100000)

#define RGB_COMPONENT_COLOR 255

 
typedef struct {
     unsigned char red,green,blue;
} PPMPixel; 

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;


static PPMImage *readPPM(const char *filename) {
  char buff[16];
  FILE *fp;
  int c, rgb_comp_color;
  PPMImage *img;

 //open PPM file for reading
  
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  //read image format
  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }  

  //check the image format
  if (buff[0] != 'P' || buff[1] != '6') {
       fprintf(stderr, "Invalid image format (must be 'P6')\n");
       exit(1);
  }

  //alloc memory form image
  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
       fprintf(stderr, "Unable to allocate memory\n");
       exit(1);
  }

  //check for comments
  c = getc(fp);
  while (c == '#') {
  while (getc(fp) != '\n') ;
       c = getc(fp);
  }

  ungetc(c, fp);
  //read image size information
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
       fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
       exit(1);
  }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }
    fclose(fp);
    return img;
}

caffe::Net<float> *_net;


int vector_add() {
  int SIZE = 1024;

  // Allocate memories for input arrays and output array.
  float *A = (float*)malloc(sizeof(float)*SIZE);
  float *B = (float*)malloc(sizeof(float)*SIZE);

  // Output
  float *C = (float*)malloc(sizeof(float)*SIZE);
  
  
  // Initialize values for array members.
  int i = 0;
  for (i=0; i<SIZE; ++i) {
    A[i] = i+1;
    B[i] = (i+1)*2;
  }

  // Load kernel from file vecAddKernel.cl

  FILE *kernelFile;
  char *kernelSource;
  size_t kernelSize;

  kernelFile = fopen("/home/zrji/android_caffe/caffe-android-opencl/examples/style_transfer/vecAddKernel.cl", "r");

  if (!kernelFile) {

    fprintf(stderr, "No file named vecAddKernel.cl was found\n");

    exit(-1);

  }
  kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
  kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
  fclose(kernelFile);

  // Getting platform and device information
  cl_platform_id platformId = NULL;
  cl_device_id deviceID = NULL;
  cl_uint retNumDevices;
  cl_uint retNumPlatforms;
  cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
  ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);

  // Creating context.
  cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);


  // Creating command queue
  cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

  // Memory buffers for each array
  cl_mem aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, &ret);
  cl_mem bMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, &ret);
  cl_mem cMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, &ret);


  // Copy lists to memory buffers
  ret = clEnqueueWriteBuffer(commandQueue, aMemObj, CL_TRUE, 0, SIZE * sizeof(float), A, 0, NULL, NULL);;
  ret = clEnqueueWriteBuffer(commandQueue, bMemObj, CL_TRUE, 0, SIZE * sizeof(float), B, 0, NULL, NULL);

  // Create program from kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);  

  // Build program
  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

  // Create kernel
  cl_kernel kernel = clCreateKernel(program, "addVectors", &ret);


  // Set arguments for kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemObj);  
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bMemObj);  
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cMemObj);  


  // Execute the kernel
  size_t globalItemSize = SIZE;
  size_t localItemSize = 64; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16 
  ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);  

  // Read from device back to host.
  ret = clEnqueueReadBuffer(commandQueue, cMemObj, CL_TRUE, 0, SIZE * sizeof(float), C, 0, NULL, NULL);

  // Write result
  /*
  for (i=0; i<SIZE; ++i) {
    printf("%f + %f = %f\n", A[i], B[i], C[i]);
  }
  */

  // Test if correct answer
  for (i=0; i<SIZE; ++i) {
    if (C[i] != (A[i] + B[i])) {
      printf("Something didn't work correctly! Failed test. \n");
      break;
    }
  }
  if (i == SIZE) {
    printf("Everything seems to work fine! \n");
  }

  // Clean up, release memory.
  ret = clFlush(commandQueue);
  ret = clFinish(commandQueue);
  ret = clReleaseCommandQueue(commandQueue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(aMemObj);
  ret = clReleaseMemObject(bMemObj);
  ret = clReleaseMemObject(cMemObj);
  ret = clReleaseContext(context);
  free(A);
  free(B);
  free(C);

  return 0;

}

int main(int argc, char** argv) {

    vector_add();

    caffe::CPUTimer timer;
    
    // Caffe::mode() = Caffe::GPU;
    // caffe::Caffe::Get().set_mode(caffe::Caffe::GPU);
    
    timer.Start();
    // NSString *modle_path = @"style.protobin"; //FilePathForResourceName(@"style", @"protobin");
    _net = new caffe::Net<float>("/home/zrji/android_caffe/caffe-android-opencl/examples/style_transfer/style.protobin", caffe::TEST);
    // NSString *weight_path = @"a1.caffemodel";//FilePathForResourceName(@"weight", @"caffemodel");
    _net->CopyTrainedLayersFrom("/home/zrji/android_caffe/caffe-android-opencl/examples/style_transfer/a1.caffemodel");
    timer.Stop();
    
    PPMImage *image;
    image = readPPM("/home/zrji/android_caffe/caffe-android-opencl/examples/style_transfer/HKU.ppm");



    
    caffe::Blob<float> *input_layer = _net->input_blobs()[0];
    for (int y = 0; y < input_layer->width(); y++) {
        for (int x = 0; x < input_layer->width(); x++) {
          input_layer->mutable_cpu_data()[y * input_layer->width() + x] = image->data[y * input_layer->width() + x].red;
          input_layer->mutable_cpu_data()[y * input_layer->width() + x + input_layer->width() * input_layer->width()] = image->data[y * input_layer->width() + x].green;
          input_layer->mutable_cpu_data()[y * input_layer->width() + x + 2 * input_layer->width() * input_layer->width()] = image->data[y * input_layer->width() + x].blue;
        }
    }

    // cl_mem tmp = (cl_mem)input_layer->mutable_gpu_data();

    // LOG(INFO) << "Input layer info: channels:" << input_layer->channels()
    // << " width: " << input_layer->width() << " Height:" << input_layer->height();
    
    // NSString *test_file_path = @"HKU.jpg"; //FilePathForResourceName(@"test_image", @"jpg");
    // std::vector<float> mean({0, 0, 0});
    // if(! ReadImageToBlob(test_file_path, mean, input_layer)) {
    //     LOG(INFO) << "ReadImageToBlob failed";
    //     return 0;
    // }
    
    timer.Start();
    _net->Forward();
    timer.Stop();
    
    std::cout << "The time used is " << timer.MicroSeconds() << std::endl;
    
    
    
    // //  code for style transfer
    caffe::Blob<float> *output_layer = _net->output_blobs()[0]; 
    FILE *f = fopen("/home/zrji/android_caffe/caffe-android-opencl/examples/style_transfer/input.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", input_layer->width(), input_layer->width());
    for (int y = 0; y < input_layer->width(); y++) {
        for (int x = 0; x < input_layer->width(); x++) {
            fputc(output_layer->cpu_data()[y * input_layer->width() + x], f);   // 0 .. 255
            fputc(output_layer->cpu_data()[y * input_layer->width() + x + input_layer->width() * input_layer->width()], f); // 0 .. 255
            fputc(output_layer->cpu_data()[y * input_layer->width() + x + 2 * input_layer->width() * input_layer->width()], f);  // 0 .. 255
            // std::cout << output_layer->cpu_data()[y * input_layer->width() + x]<< std::endl;   // 0 .. 255
            // std::cout <<output_layer->cpu_data()[y * input_layer->width() + x + input_layer->width() * input_layer->width()] << std::endl; // 0 .. 255
            // std::cout << output_layer->cpu_data()[y * input_layer->width() + x + 2 * input_layer->width() * input_layer->width()]<< std::endl;  // 0 .. 255
        }
    }
    fclose(f);
    
    // delete _net;
    // int i = 9;
    
    // for (i=0; i<10;++i){
    //     cout<<'gg'<<endl;
    // }
    
}
