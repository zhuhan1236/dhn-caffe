/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    /*LOG(INFO) << "hdf5: current_row_11 " << current_row_;*/
    /*LOG(INFO) << "shape0000 " << hdf_blobs_[0]->shape(0);*/
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        current_file_ += 1;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.hdf5_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadHDF5FileData(
            hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
      current_row_ = 0;
      if (this->layer_param_.hdf5_data_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
    //LOG(INFO) << "ttttop size" << this->layer_param_.top_size();
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      LOG(INFO) << "data dim  " << data_dim;
      LOG(INFO) << "hdf5: current_row_  " << current_row_;
      LOG(INFO) << "hdf5: data_permutation  " << data_permutation_[current_row_];
      LOG(INFO) << "i  " << i;
      LOG(INFO) << "j  " << j;
      LOG(INFO) << "hdf_blobs_0  " << hdf_blobs_[j]->cpu_data()[0];
      LOG(INFO) << "hdf_blobs_1  " << hdf_blobs_[j]->cpu_data()[1000];
      LOG(INFO) << "hdf_blobs_1  " << hdf_blobs_[j]->cpu_data()[6453*data_dim];
      LOG(INFO) << "hdf_blobs_1  " << hdf_blobs_[j]->cpu_data()[6454*data_dim];
      for(int k = 0; k < data_dim; k ++){
          LOG(INFO) << "hdf6454 " << k << " " << hdf_blobs_[j]->cpu_data()[6454*data_dim+k];
      }
      LOG(INFO) << "hdf_blobs_1  " << hdf_blobs_[j]->cpu_data()[6455*data_dim];
      //print_gpu_matrix(&hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_] * data_dim], 1, data_dim, 1, data_dim);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_gpu_data()[i * data_dim]);
      //caffe_gpu_memcpy(data_dim*sizeof(Dtype), &(hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]]), &(top[j]->mutable_gpu_data()[i * data_dim]));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5DataLayer);

}  // namespace caffe
