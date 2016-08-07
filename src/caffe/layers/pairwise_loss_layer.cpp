#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PairwiseLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter pairwise_param(this->layer_param_);
  pairwise_param.set_type("Pairwise");
  
  l_threshold_ = this->layer_param_.pairwise_param().l_threshold();
  q_threshold_ = this->layer_param_.pairwise_param().q_threshold();
  l_lambda_ = this->layer_param_.pairwise_param().l_lambda();
  q_gamma_ = this->layer_param_.pairwise_param().q_gamma();
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  pairwise_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.pairwise_param().axis());
  outer_num_ = bottom[0]->count(0, pairwise_axis_);
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
  pairwise_s_.Reshape(1, 1, bottom[0]->count(0, 1), bottom[0]->count(0, 1));
  loss_.Reshape(1, 1, bottom[0]->count(0, 1), bottom[0]->count(0, 1));
  q_loss_.Reshape(1, 1, bottom[0]->count(0, 1), bottom[0]->count(1));
  temp_.Reshape(1, 1, outer_num_, outer_num_);
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(PairwiseLossLayer);
#endif

INSTANTIATE_CLASS(PairwiseLossLayer);
REGISTER_LAYER_CLASS(PairwiseLoss);

}  // namespace caffe
