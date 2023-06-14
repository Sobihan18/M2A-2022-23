#include "src/model.h"

using keras2cpp::Model;
using keras2cpp::Tensor;

int main() {
    // Initialize model.
    auto model = Model::load("interface.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in{9};
    in.data_ = {1,1,1,0.5,0.5,0.5,0,0,0};

    // Run prediction.
    Tensor out = model(in);
    out.print();
    return 0;
}
