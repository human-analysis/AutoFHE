#pragma	once
#include "seal/seal.h"
#include "SEALcomp.h"
#include "MinicompFunc.h"
#include "func.h"
#include "PolyUpdate.h"
#include "program.h"
#include "Bootstrapper.h"
#include "cnn_seal.h"
#include <omp.h>
#include <NTL/RR.h>
#include <fstream>
#include <vector>
#include <chrono>


// AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE. USENIX Security 2024

void resnet_autofhe_cifar10(string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t num_bootstrapping, size_t start_image_id, size_t end_image_id);

void resnet_autofhe_cifar100(string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t num_bootstrapping, size_t start_image_id, size_t end_image_id);

void vgg_autofhe(string &dataset, string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t num_bootstrapping, size_t start_image_id, size_t end_image_id);

void import_weights_resnet_autofhe_cifar10(string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<double> &in_range, vector<double> &out_range,  vector<int> &depth, vector<int> &boot_loc, size_t layer_num, size_t end_num);

void import_weights_resnet_autofhe_cifar100(string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<vector<double>> &shortcut_weight, vector<vector<double>> &shortcut_bn_bias, vector<vector<double>> &shortcut_bn_mean, vector<vector<double>> &shortcut_bn_var, vector<vector<double>> &shortcut_bn_weight, vector<double> &in_range, vector<double> &out_range,  vector<int> &depth, vector<int> &boot_loc, size_t layer_num, size_t end_num);

void import_weights_vgg_autofhe(size_t n_class, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<double> &in_range, vector<double> &out_range,  vector<int> &depth, vector<int> &boot_loc, size_t layer_num, size_t end_num);

// FHE-MP-CNN: Low-complexity deep convolutional neural networks on fully homomorphic encryption using multiplexed parallel convolutions. ICML 2022

void resnet_mpcnn_cifar10(string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);

void resnet_mpcnn_cifar100(string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);

void vgg_mpcnn(string &dataset, string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);

void import_weights_resnet_mpcnn_cifar10(string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, double &B, size_t layer_num, size_t end_num);

void import_weights_resnet_mpcnn_cifar100(string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<vector<double>> &shortcut_weight, vector<vector<double>> &shortcut_bn_bias, vector<vector<double>> &shortcut_bn_mean, vector<vector<double>> &shortcut_bn_var, vector<vector<double>> &shortcut_bn_weight, double &B, size_t layer_num, size_t end_num);

void import_weights_vgg_mpcnn(size_t n_class, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, double &B, size_t layer_num, size_t end_num);

// AESPA: Accuracy preserving low-degree polynomial activation for fast private inference. arXiv 2022

void resnet_aespa_cifar10(string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);

void resnet_aespa_cifar100(string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);

void vgg_aespa(string &dataset, string &model, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);

void import_weights_resnet_aespa_cifar10(string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<double> &herpn_range, vector<vector<double>> &herpn_w0, vector<vector<double>> &herpn_w1, vector<vector<double>> &herpn_w2, size_t layer_num, size_t end_num);

void import_weights_resnet_aespa_cifar100(string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<double> &herpn_range, vector<vector<double>> &herpn_w0, vector<vector<double>> &herpn_w1, vector<vector<double>> &herpn_w2, vector<vector<double>> &shortcut_weight, vector<vector<double>> &shortcut_bn_bias, vector<vector<double>> &shortcut_bn_mean, vector<vector<double>> &shortcut_bn_var, vector<vector<double>> &shortcut_bn_weight, size_t layer_num, size_t end_num);

void import_weights_vgg_aespa(size_t n_class, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<double> &herpn_range, vector<vector<double>> &herpn_w0, vector<vector<double>> &herpn_w1, vector<vector<double>> &herpn_w2, size_t layer_num, size_t end_num);


// load dataset

void load_dataset(string &dataset, string &dataset_dir, vector<vector<double>> &images, vector<int> &labels);

