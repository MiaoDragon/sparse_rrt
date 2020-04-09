#ifndef MPNET_MODEL_
#define MPNET_MODEL_
#include <torch/torch.h>

using namespace torch;
// Define a new Module.

struct MLP : nn::Module {
  MLP(input_size, output_size)
  {
  }

  // Implement the Net's algorithm.
  virtual torch::Tensor forward(torch::Tensor x) = 0;
  // Use one of many "standard library" modules.
  nn::Sequential fc{nullptr};
};

struct Encoder : nn::Module {
    Encoder(input_size, output_size)
    {
    }
    // Implement the Net's algorithm.
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    // Use one of many "standard library" modules.
    nn::Sequential encoder{nullptr}, head{nullptr};
};

struct MPNet: nn::Module
{
    MPNet(encoder_input_size, encoder_output_size, mlp_input_size, output_size)
    {
    }
    virtual torch::Tensor forward(torch::Tensor x, torch::Tensor obs) = 0;
    nn::Module mlp{nullptr}, encoder{nullptr};
}

#endif
