#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EntropyLoss(const int nthreads, const Dtype* product, 
        const Dtype* exp_product, const Dtype* similarity, const Dtype threshold, 
        Dtype* count, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if(similarity[index] < 0){
          count[index] = Dtype(1);
          if((threshold > 0) && (abs(product[index]) >= threshold)){
              loss[index] = Dtype(0.0);
          }
          else{
              loss[index] = log(1 + 1 / exp_product[index]) / (1 + 1 / exp_product[index]);
          }    
      }
      else{
          count[index] = Dtype(0);
          loss[index] = Dtype(0);
      }
  }
}

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
void EntropyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if(now_iter_ < start_iter_){
        top[0]->mutable_cpu_data()[0] = Dtype(0);
        return;
    }
    Dtype* similarity = pairwise_p_.mutable_gpu_data();
    Dtype* dot_product = pairwise_p_.mutable_gpu_diff();
    Dtype* exp_product = loss_.mutable_gpu_data();
    Dtype* loss = loss_.mutable_gpu_diff();
    Dtype* label = bottom[1]->mutable_gpu_data();
    Dtype* count = temp_.mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    int nthreads = outer_num_ * outer_num_;
    
    //calculate similarity matrix
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
          Dtype(1.0), label, label, Dtype(0.0), similarity);
    SimilarityProcess<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, label_dim_);
    
    //calculate code similarity and exp(code similarity)
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, inner_num_, 
          Dtype(1.0), bottom_data, bottom_data, Dtype(0.0), dot_product);
    caffe_gpu_exp(outer_num_ * outer_num_, dot_product, exp_product);
    
    //calculate loss data
    EntropyLoss<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, dot_product, exp_product, similarity, 
                threshold_, count, loss);
    
    Dtype loss_value, count_num;
    caffe_gpu_asum(nthreads, loss, &loss_value);
    caffe_gpu_asum(nthreads, count, &count_num);
    
    top[0]->mutable_cpu_data()[0] = loss_value / (count_num > 0 ? count_num : Dtype(1)) * loss_weight_;
}

template <typename Dtype>
__global__ void SigmoidProb(const int nthreads, const Dtype* similarity, 
        const Dtype* exp_product, Dtype* prob) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(similarity[index] < 0){
        prob[index] = Dtype(1) / (Dtype(1) + 1 / exp_product[index]);
    }
    else{
        prob[index] = Dtype(-1);
    }
  }
}

template <typename Dtype>
__global__ void EntropyDiff(const int nthreads, const Dtype* similarity, 
        const Dtype* product, const Dtype* exp_product, const Dtype threshold, 
        Dtype* count, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(similarity[index] < 0){
        count[index] = Dtype(1);
        if((threshold > 0) && (abs(product[index]) >= threshold)){
            diff[index] = Dtype(0.0);
        }
        else{
            diff[index] = -(Dtype(1) - log(Dtype(1) + 1 / exp_product[index])) / (1 / exp_product[index] + exp_product[index] + Dtype(2.0));
        }    
    }
    else{
        count[index] = Dtype(0);
        diff[index] = Dtype(0);
    }
  }
}

template <typename Dtype>
void EntropyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to label inputs.";
    }
    if(now_iter_++ < start_iter_){
        return;
    }
    
    const Dtype* similarity = pairwise_p_.gpu_data();
    const Dtype* dot_product = pairwise_p_.gpu_diff();
    const Dtype* exp_product = loss_.gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* diff = loss_.mutable_gpu_diff();
    Dtype* count = temp_.mutable_gpu_data();
    Dtype* prob = temp_.mutable_gpu_diff();
    int nthreads = outer_num_ * outer_num_;
    
    //calculate prob
    SigmoidProb<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, exp_product, prob);

    //calculate diff 
    EntropyDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, dot_product, exp_product, 
                threshold_, count, diff);

    /*LOG(INFO) << "--------------------------------------";*/
    /*print_gpu_matrix(similarity, outer_num_, outer_num_, 10, 10);*/
    /*LOG(INFO) << "--------------------------------------";*/
    /*print_gpu_matrix(dot_product, outer_num_, outer_num_, 10, 10);*/
    /*LOG(INFO) << "--------------------------------------";*/
    /*print_gpu_matrix(prob, outer_num_, outer_num_, 10, 10);*/
    /*LOG(INFO) << "--------------------------------------";*/
    /*print_gpu_matrix(diff, outer_num_, outer_num_, 10, 10);*/
    /*LOG(INFO) << "--------------------------------------";*/

    //add to bottom_diff
    Dtype count_num;
    caffe_gpu_asum(nthreads, count, &count_num);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, inner_num_, outer_num_, 
          loss_weight_ / (count_num > 0 ? count_num : Dtype(1)), diff, bottom_data, 
          Dtype(0), bottom_diff);

    caffe_gpu_asum(outer_num_ * inner_num_, bottom_diff, &count_num);
    LOG(INFO) << "entropy diff: " << count_num;
    LOG(INFO) << "-----------------------------";
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyLayer);

}  // namespace caffe
