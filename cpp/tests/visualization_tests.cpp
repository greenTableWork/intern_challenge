#include "placement/visualization.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to read visualization output");
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

void visualizationWritesPngWithExpectedContent() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);

    const auto initial_cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
        },
        float_options);
    const auto final_cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 5.0F, 0.0F, 2.0F, 2.0F},
        },
        float_options);

    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() / "placement_cpp_visualization_tests" /
        "nested" / "tiny_placement.png";
    const std::filesystem::path single_output_path =
        output_path.parent_path() / "single_placement.png";
    std::filesystem::remove(output_path);
    std::filesystem::remove(single_output_path);

    placement::plotPlacement(initial_cell_features, final_cell_features, output_path);
    placement::plotPlacementState(initial_cell_features, single_output_path, "Initial Placement");

    expect(std::filesystem::exists(output_path), "visualization output exists");
    expect(std::filesystem::file_size(output_path) > 0, "visualization output is nonempty");
    expect(
        std::filesystem::exists(single_output_path),
        "single-state visualization output exists");
    expect(
        std::filesystem::file_size(single_output_path) > 0,
        "single-state visualization output is nonempty");

    const std::string content = readFile(output_path);
    expect(content.size() > 8, "visualization output has png header bytes");
    expect(
        static_cast<unsigned char>(content[0]) == 0x89 && content.substr(1, 3) == "PNG",
        "visualization output is a png");

    const std::string single_content = readFile(single_output_path);
    expect(single_content.size() > 8, "single-state visualization output has png header bytes");
    expect(
        static_cast<unsigned char>(single_content[0]) == 0x89 &&
            single_content.substr(1, 3) == "PNG",
        "single-state visualization output is a png");
}
