# dhn-caffe

This is the implementation of AAAI paper "Deep Hashing Network for Efficient Similarity Retrieval". We fork the repository from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Change the label from a single integer to an array, whose length is the dimension of label.
- Add pairwise loss layer described in the paper

Training Model
---------------

In `models/nus_wide`, we give an example model based on Alexnet and NUS-WIDE dataset.

The `bvlc_reference_caffenet` is used as the pre-trained model.

Data Preparation
---------------
In `data/nus_wide/train.txt`, we give an example to show how to prepare the train data file. In this file, each image in NUS-WIDE has its label whose dimension is 21. The training set is extracted from the whole dataset as described in our paper.

Parameter Tuning
---------------
In ImageDataLayer, parameter `label_dim` should be set to tell the dimension of label. For example, `label_dim: 21` for NUS-WIDE. 

`base_lr`, `stepsize` and `gamma` in `solver.prototxt` can be tuned to achieve better performance. `num_output` of `hash_layer` in `train_val.prototxt` can be tuned to train with different hash bits. `q_gamma` in `pairwise_loss` layer can be tuned to set different loss weight for quantization loss.

Testing
---------------
After training from `bvlc_reference_caffenet`, a finetuned `caffemodel` can be used to generate hash codes for images in database and test samples. The python program in `models/predict` can help gengerating hash codes by given a file like `train.txt`.
