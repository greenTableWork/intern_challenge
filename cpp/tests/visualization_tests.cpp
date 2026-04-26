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
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to read visualization output");
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

void visualizationWritesSvgWithExpectedContent() {
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
        "nested" / "tiny_placement.svg";
    std::filesystem::remove(output_path);

    placement::plotPlacement(initial_cell_features, final_cell_features, output_path);

    expect(std::filesystem::exists(output_path), "visualization output exists");
    expect(std::filesystem::file_size(output_path) > 0, "visualization output is nonempty");

    const std::string content = readFile(output_path);
    expect(
        content.find("Initial Placement") != std::string::npos,
        "visualization contains initial label");
    expect(
        content.find("Final Placement") != std::string::npos,
        "visualization contains final label");
    expect(content.find("<rect") != std::string::npos, "visualization contains rectangles");
    expect(
        content.find("Overlaps: 1") != std::string::npos,
        "visualization contains initial overlap count");
    expect(
        content.find("Overlaps: 0") != std::string::npos,
        "visualization contains final overlap count");
    expect(
        content.find("Total Overlap Area: 2.00") != std::string::npos,
        "visualization contains formatted overlap area");
}
