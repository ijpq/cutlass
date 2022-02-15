#include <torch/extension.h>

torch::Tensor forward_fp32(torch::Tensor input, torch::Tensor weight);
torch::Tensor backward_data_fp32(torch::Tensor input, torch::Tensor weight);
torch::Tensor backward_filter_fp32(torch::Tensor diff, torch::Tensor input, torch::Tensor weight);
