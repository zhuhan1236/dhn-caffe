#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EntropyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter entropy_param(this->layer_param_);
  entropy_param.set_type("Entropy");
  
  threshold_ = this->layer_param_.entropy_param().threshold();
  start_iter_ = this->layer_param_.entropy_param().start_iteration();
  now_iter_ = 0;
  loss_weight_ = this->layer_param_.loss_weight(0);
}

template <typename Dtype>
void EntropyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->count(0, 1);
  inner_num_ = bottom[0]->count(1);
  label_dim_ = bottom[1]->count(1);
  /*CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())*/
      //<< "Number of labels must match number of predictions; "
      //<< "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      //<< "label count (number of labels) must be N*H*W, "
      /*<< "with integer values in {0, 1, ..., C-1}.";*/
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  pairwise_p_.Reshape(1, 1, outer_num_, outer_num_);
  loss_.Reshape(1, 1, outer_num_, outer_num_);
  temp_.Reshape(1, 1, outer_num_, outer_num_);
  diff_.Reshape(1, 1, outer_num_, inner_num_);
}

template <typename Dtype>
void EntropyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void EntropyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(EntropyLayer);
#endif

INSTANTIATE_CLASS(EntropyLayer);
REGISTER_LAYER_CLASS(Entropy);

}  // namespace caffe
