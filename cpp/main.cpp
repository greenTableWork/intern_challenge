#include <CLI/CLI.hpp>
#include <torch/torch.h>

#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"Placement C++ LibTorch smoke test"};
    bool print_tensor = false;
    app.add_flag("--print-tensor", print_tensor, "Print the generated tensor");
    CLI11_PARSE(app, argc, argv);

    torch::manual_seed(66);
    const torch::Tensor values = torch::rand({2, 3}, torch::kFloat32);
    const torch::Tensor doubled = values * 2.0;

    std::cout << "LibTorch smoke test\n";
    std::cout << "Tensor sizes: " << values.sizes() << "\n";
    std::cout << "Mean doubled value: " << doubled.mean().item<double>() << "\n";

    if (print_tensor) {
        std::cout << doubled << "\n";
    }

    return 0;
}
