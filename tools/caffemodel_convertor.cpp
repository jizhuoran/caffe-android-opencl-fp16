//
//  main.m
//  metal_mac
//
//  Created by Tec GSQ on 27/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

// #include <iostream>
#include "caffe/caffe.hpp"
// #include <stdio.h>
// #include <stdlib.h>



int main(int argc, char** argv) {


    if (argc != 4) {
      LOG(INFO) << "./caffemodel_convertor.bin prototxt_file fp32.caffemodel(input) fp16.caffemodel(output)";
      exit(0);
    }
    caffe::Net<float> *_net;

    _net = new caffe::Net<float>(argv[1], caffe::TEST);
    _net->CopyTrainedLayersFrom(argv[2]);
    
    caffe::NetParameter net_param;
    _net->ToHalfProto(&net_param);
    caffe::WriteProtoToBinaryFile(net_param, argv[3]);

}
