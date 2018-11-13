#include "caffe/layer.hpp"

namespace caffe {

#ifdef FORWARD_LESS_MEM
	template <typename Dtype> 
	void Layer<Dtype>::Qiaoge_alloc(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { }
	
	template <typename Dtype> 
	void Layer<Dtype>::Qiaoge_free(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { }
#endif

	INSTANTIATE_CLASS(Layer);
}