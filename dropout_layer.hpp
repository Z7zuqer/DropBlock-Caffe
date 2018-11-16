#ifndef CAFFE_DROPOUT_LAYER_HPP_
#define CAFFE_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Dropout"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
#ifndef CPU_ONLY
  inline void conv_im2col_gpu_1(const float* data, float* col_buff) {
    im2col_gpu_1(data, conv_out_channels_,
        featuremap_h_, featuremap_w_,
        block_size_, block_size_,
        block_size_ / 2, block_size_ / 2,
        1, 1, 1, 1, col_buff);
    //rewrite im2col to change padding-0 to padding-1 
  }
  inline void forward_gpu_gemm(const float* input, const float* weights, float* output){
    conv_im2col_gpu_1(input, col_buffer_.mutable_gpu_data());
    const float* col_buff = col_buffer_.gpu_data();
    caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans,conv_out_channels_,
        conv_out_spatial_dim_, kernel_dim_,
        1, weights , col_buff,
        0, output);
  }
#endif
  Blob<float> rand_vec_[2];
  Blob<unsigned int> int_rand_;
  Dtype threshold_;
  Dtype scale_;
  unsigned int uint_thres_;
  unsigned int block_size_;
  unsigned int block_thres_;
  int channels_;
  int padded_size_;
  int kernel_dim_;
  int featuremap_h_;
  int featuremap_w_;
  Blob<float> col_buffer_;
  int dim_;
  int num_;
  float sum_ = 0;
  vector<shared_ptr<Blob<float> > > weights_;
};

}

#endif 
