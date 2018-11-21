#!/usr/bin/env sh

./build/tools/caffe test --model examples/mnist/lenet_train_test.prototxt --weights examples/mnist/weight.caffemodel
