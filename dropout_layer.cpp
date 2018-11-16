#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  block_size_ = this->layer_param_.dropout_param().block_size();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  DCHECK(block_size_ > 0);
  DCHECK(block_size_ & 1);
  if(block_size_>1){
    block_thres_ = block_size_ * block_size_;
    vector<int>weight_shape(2,bottom[0]->channels());
    for(int i=0;i<2;i++)
      weight_shape.push_back(block_size_);
    channels_=bottom[0]->channels();
    weights_.resize(1);
    weights_[0].reset(new Blob<float>(weight_shape));
    kernel_dim_ = block_thres_;
    vector<int>col_shape;
    num_=bottom[0]->num();
    featuremap_h_=bottom[0]->height();
    featuremap_w_=bottom[0]->width();
    padded_size_=(featuremap_h_+block_size_-1)*(featuremap_w_+block_size_-1);
    this->blobs_.resize(0);
    dim_ = channels_*featuremap_h_*featuremap_w_;
    col_shape.push_back(kernel_dim_);
    col_shape.push_back(featuremap_h_+block_size_-1);
    col_shape.push_back(featuremap_w_+block_size_-1);
    col_buffer_.Reshape(col_shape);
    for(int i=0;i<2;i++)
          rand_vec_[i].Reshape(bottom[0]->shape());
    int_rand_.Reshape(bottom[0]->shape());
    for(int i=0;i<weights_[0]->count();i++)
      weights_[0]->mutable_cpu_data()[i]=1;
    const int count = bottom[0]->count();
    float r = ((1.  - threshold_) / float(block_size_ * block_size_));
    r*=(float)bottom[0]->height()/(float)(bottom[0]->height()-block_size_+1);
    r*=(float)bottom[0]->width()/(float)(bottom[0]->width()-block_size_+1);
  }
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  for(int i=0;i<2;i++)
    rand_vec_[i].Reshape(bottom[0]->shape());
  int_rand_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //no cpu
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //no cpu
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  
