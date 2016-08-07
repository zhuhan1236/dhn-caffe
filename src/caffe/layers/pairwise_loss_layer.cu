#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SimilarityProcess(const int nthreads, Dtype* similarity, Dtype label_dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if((similarity[index] < 0) || (similarity[index] >= label_dim)){
          //unknown label
          similarity[index] = Dtype(-1.0);
      }
      else if(similarity[index] > 0){
          //similar label
          similarity[index] = Dtype(1.0);
      }
  }
}

template <typename Dtype>
__global__ void PairwiseLossForwardGPU(const int nthreads, const int num, const Dtype* similarity, 
       const Dtype* exp_product, const Dtype* product, const Dtype threshold, Dtype* count, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if(similarity[index] >= 0){
          count[index] = Dtype(1.0);
          if((threshold >= 0) && (product[index] >= threshold)){
              loss_data[index] = product[index] * (1 - (similarity[index] > 0));
          }
          else{
              loss_data[index] = log(1 + exp_product[index]) - (similarity[index] > 0) * product[index];
          }
      }
      else{
          count[index] = Dtype(0.0);
          loss_data[index] = Dtype(0.0);
      }
  }
}

template <typename Dtype>
__global__ void ForwardGPU(const int nthreads, const Dtype* bottom_data, Dtype threshold, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if((threshold > 0) && ((abs(bottom_data[index]) - 1) > threshold)){
          loss_data[index] = (abs(bottom_data[index]) - 1) - log(Dtype(2));
      }
      else{
          loss_data[index] = log(cosh(abs(bottom_data[index]) - 1));
      }
  }
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* similarity = pairwise_s_.mutable_gpu_data();
  Dtype* dot_product = pairwise_s_.mutable_gpu_diff();
  Dtype* exp_product = loss_.mutable_gpu_diff();
  Dtype* loss_data = loss_.mutable_gpu_data();
  Dtype* count = temp_.mutable_gpu_data();
  Dtype* label = bottom[1]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int nthreads = outer_num_ * outer_num_;
  //calculate similarity matrix according to label
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
          Dtype(1.0), label, label, Dtype(0.0), similarity);
  SimilarityProcess<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, label_dim_);

  //calculate dot_product and exp_product
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, inner_num_, 
          Dtype(1.0), bottom_data, bottom_data, Dtype(0.0), dot_product);
  caffe_gpu_exp(outer_num_ * outer_num_, dot_product, exp_product);
  
  //calculate pairwise loss
  PairwiseLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity, exp_product, 
              dot_product, l_threshold_, count, loss_data);
  
  Dtype loss, count_num;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  caffe_gpu_asum(nthreads, count, &count_num);
  loss /= (count_num > 0 ? count_num : Dtype(1));
  LOG(INFO) << "L loss:" << loss;
  loss = loss * (l_lambda_ > 0);
  
  //calculate quantization loss
  nthreads = outer_num_ * inner_num_;
  ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, q_threshold_, q_loss_.mutable_gpu_data());

  Dtype q_loss;
  caffe_gpu_asum(nthreads, q_loss_.gpu_data(), &q_loss);
  q_loss /= nthreads;
  LOG(INFO) << "Q loss:" << q_loss;
  q_loss = q_loss * (q_gamma_ > 0);

  top[0]->mutable_cpu_data()[0] = loss + q_loss;
}

template <typename Dtype>
__global__ void PairwiseLossBackwardGPU(const int nthreads, const int num, 
          const Dtype* similarity, const Dtype* exp_product, Dtype* count, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if(similarity[index] >= 0){
          diff[index] = 2 * (
              1 / (1 + 1 / exp_product[index]) - 
              (similarity[index] > 0)
          );
          count[index] = Dtype(1.0);
      }
      else{
          diff[index] = Dtype(0.0);
          count[index] = Dtype(0.0);
      }
  }
}

template <typename Dtype>
__global__ void BackwardGPU(const int nthreads, const int num, const int dim, 
        const Dtype* bottom_data, const Dtype scale, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(bottom_data[index] >= 0){
        bottom_diff[index] += scale * tanh(abs(bottom_data[index]) - 1);
    }
    else{
        bottom_diff[index] += -scale * tanh(abs(bottom_data[index]) - 1);
    }
  }
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* diff = temp_.mutable_gpu_data();
    Dtype* count = temp_.mutable_gpu_diff();
    const Dtype* similarity = pairwise_s_.gpu_data();
    const Dtype* exp_product = loss_.gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    int nthreads = outer_num_ * outer_num_;
    
    //calculate diff
    PairwiseLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity,
                exp_product, count, diff);
        
    //copy to bottom_diff
    Dtype count_num;
    caffe_gpu_asum(nthreads, count, &count_num);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, inner_num_, outer_num_, 
          l_lambda_ / (count_num > 0 ? count_num : Dtype(1)), diff, bottom_data, 
          Dtype(0.0), bottom_diff);

    //add quantization term
    nthreads = inner_num_ * outer_num_;
    BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, inner_num_, bottom_data, 
                q_gamma_ / outer_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLossLayer);

}  // namespace caffe
