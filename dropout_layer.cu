#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskGene(const int n, float* mask, unsigned int threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    mask[index] = ((mask[index] >= threshold) ? 1 : 0);
  }
}
__global__ void MaskTrans(const int n, float* mask, unsigned int* mask_t, unsigned int threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    mask[index] = mask_t[index] > threshold ? 1 : 0;
  }
}
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const float scale, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * mask[index] * scale;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask_tt = static_cast<unsigned int*>(int_rand_.mutable_gpu_data()); 
    caffe_gpu_rng_uniform(count, mask_tt);
    float* mask_t = static_cast<float*>(rand_vec_[0].mutable_gpu_data());
    MaskTrans<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, mask_t, mask_tt, uint_thres_);
    if (block_size_ > 1) {
      const float* weight = weights_[0]->gpu_data();
      float* mask = static_cast<float*>(rand_vec_[1].mutable_gpu_data());
      const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      for(int n=0;n<this->num_;n++){
        this->forward_gpu_gemm(mask_t+n*this->dim_,weight,
            mask+n*this->dim_);
      }
      MaskGene<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
              mask, this->block_thres_);
      sum_=0;
      caffe_gpu_asum(count,mask,&sum_);
      if(sum_==0)sum_=count;
      DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, mask,
               ((float)count/sum_), top_data);
    }
    else {
        DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, mask_t, scale_, top_data);
    }
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const float* mask = static_cast<const float*>(rand_vec_[block_size_>1].gpu_data());
      const int count = bottom[0]->count();
      float now_scale_=scale_;
      if(block_size_>1)now_scale_=((sum_==0)?1:((float)count/sum_));
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
