#include <torch/cuda.h>
#include <torch/jit.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace torch::indexing;

constexpr double kMinMacroArea = 100.0;
constexpr double kMaxMacroArea = 10000.0;
constexpr double kStandardCellHeight = 1.0;
constexpr int64_t kMinStandardCellPins = 3;
constexpr int64_t kMaxStandardCellPins = 6;
constexpr double kPinSize = 0.1;
constexpr double kTwoPi = 6.28318530717958647692;

enum class CellFeatureIdx : int64_t {
    Area = 0,
    NumPins = 1,
    X = 2,
    Y = 3,
    Width = 4,
    Height = 5,
};

enum class PinFeatureIdx : int64_t {
    CellIdx = 0,
    PinX = 1,
    PinY = 2,
    X = 3,
    Y = 4,
    Width = 5,
    Height = 6,
};

struct PlacementProblem {
    torch::Tensor cell_features;
    torch::Tensor pin_features;
    torch::Tensor edge_list;
};

struct PlacementTensorSetup {
    torch::Tensor macro_areas;
    torch::Tensor std_area_indices;
    torch::Tensor std_cell_areas;
    torch::Tensor areas;
    torch::Tensor cell_widths;
    torch::Tensor cell_heights;
    torch::Tensor cell_features;
};

int64_t featureIndex(CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

int64_t featureIndex(PinFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

const std::shared_ptr<torch::jit::CompilationUnit>& jitSmokeProgram() {
    static const auto program = torch::jit::compile(R"JIT(
def jit_smoke():
    return torch.zeros([1], device='cuda')
)JIT");
    return program;
}

const std::shared_ptr<torch::jit::CompilationUnit>& placementTensorSetupProgram() {
    static const auto program = torch::jit::compile(R"JIT(
def placement_tensor_setup(num_macros: int,
                           num_std_cells: int,
                           min_macro_area: float,
                           max_macro_area: float,
                           standard_cell_height: float):
    macro_areas = torch.rand([num_macros], device='cuda') * (max_macro_area - min_macro_area) + min_macro_area
    standard_areas = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    std_area_indices = torch.randint(0, standard_areas.size(0), [num_std_cells], device='cuda')
    std_cell_areas = torch.index_select(standard_areas, 0, std_area_indices)
    areas = torch.cat([macro_areas, std_cell_areas])

    macro_widths = torch.sqrt(macro_areas)
    macro_heights = torch.sqrt(macro_areas)
    std_cell_widths = std_cell_areas / standard_cell_height
    std_cell_heights = torch.full([num_std_cells], standard_cell_height, device='cuda')
    cell_widths = torch.cat([macro_widths, std_cell_widths])
    cell_heights = torch.cat([macro_heights, std_cell_heights])

    cell_features = torch.zeros([num_macros + num_std_cells, 6], device='cuda')
    cell_features[:, 0] = areas
    cell_features[:, 4] = cell_widths
    cell_features[:, 5] = cell_heights

    return macro_areas, std_area_indices, std_cell_areas, areas, cell_widths, cell_heights, cell_features
)JIT");
    return program;
}

void checkTensor(
    const torch::Tensor& tensor,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> sizes) {
    TORCH_CHECK(tensor.is_cuda(), "expected CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, "unexpected tensor dtype");
    TORCH_CHECK(tensor.sizes() == sizes, "unexpected tensor shape");
}

void runJitSmokeCheck() {
    const torch::Tensor smoke =
        jitSmokeProgram()->run_method("jit_smoke").toTensor();
    checkTensor(smoke, torch::kFloat32, {1});
}

PlacementTensorSetup buildPlacementTensorSetup(
    int64_t macro_count,
    int64_t std_cell_count) {
    const c10::IValue result = placementTensorSetupProgram()->run_method(
        "placement_tensor_setup",
        macro_count,
        std_cell_count,
        kMinMacroArea,
        kMaxMacroArea,
        kStandardCellHeight);

    const auto tuple = result.toTuple();
    const std::vector<c10::IValue>& elements = tuple->elements();
    TORCH_CHECK(elements.size() == 7, "unexpected JIT tensor setup tuple size");

    PlacementTensorSetup setup{
        elements[0].toTensor(),
        elements[1].toTensor(),
        elements[2].toTensor(),
        elements[3].toTensor(),
        elements[4].toTensor(),
        elements[5].toTensor(),
        elements[6].toTensor(),
    };

    const int64_t total_cells = macro_count + std_cell_count;
    checkTensor(setup.macro_areas, torch::kFloat32, {macro_count});
    checkTensor(setup.std_area_indices, torch::kInt64, {std_cell_count});
    checkTensor(setup.std_cell_areas, torch::kFloat32, {std_cell_count});
    checkTensor(setup.areas, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_widths, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_heights, torch::kFloat32, {total_cells});
    checkTensor(setup.cell_features, torch::kFloat32, {total_cells, 6});

    return setup;
}

PlacementProblem generatePlacementInput(
    int num_macros,
    int num_std_cells,
    const torch::Device& device) {
    const int64_t macro_count = num_macros;
    const int64_t std_cell_count = num_std_cells;
    const int64_t total_cells = macro_count + std_cell_count;
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(device);

    PlacementTensorSetup setup =
        buildPlacementTensorSetup(macro_count, std_cell_count);
    auto macro_areas = setup.macro_areas;
    auto cell_widths = setup.cell_widths;
    auto cell_heights = setup.cell_heights;

    auto num_pins_per_cell = torch::zeros({total_cells}, long_options);
    for (int64_t i = 0; i < macro_count; ++i) {
        const int64_t sqrt_area =
            static_cast<int64_t>(std::sqrt(macro_areas[i].item<double>()));
        num_pins_per_cell.index_put_(
            {i},
            torch::randint(sqrt_area, 2 * sqrt_area + 1, {1}, long_options)[0]);
    }
    if (std_cell_count > 0) {
        num_pins_per_cell.index_put_(
            {Slice(macro_count, total_cells)},
            torch::randint(
                kMinStandardCellPins,
                kMaxStandardCellPins + 1,
                {std_cell_count},
                long_options));
    }

    auto cell_features = setup.cell_features;
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::NumPins)},
        num_pins_per_cell.to(torch::kFloat32));

    const int64_t total_pins = num_pins_per_cell.sum().item<int64_t>();
    auto pin_features = torch::zeros({total_pins, 7}, float_options);

    int64_t pin_idx = 0;
    for (int64_t cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        const int64_t n_pins = num_pins_per_cell[cell_idx].item<int64_t>();
        const double cell_width = cell_widths[cell_idx].item<double>();
        const double cell_height = cell_heights[cell_idx].item<double>();
        const double margin = kPinSize / 2.0;

        torch::Tensor pin_x;
        torch::Tensor pin_y;
        if (cell_width > 2.0 * margin && cell_height > 2.0 * margin) {
            pin_x = torch::rand({n_pins}, float_options) * (cell_width - 2.0 * margin) +
                    margin;
            pin_y = torch::rand({n_pins}, float_options) * (cell_height - 2.0 * margin) +
                    margin;
        } else {
            pin_x = torch::full({n_pins}, cell_width / 2.0, float_options);
            pin_y = torch::full({n_pins}, cell_height / 2.0, float_options);
        }

        const auto rows = Slice(pin_idx, pin_idx + n_pins);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::CellIdx)}, cell_idx);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::PinX)}, pin_x);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::PinY)}, pin_y);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::X)}, pin_x);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::Y)}, pin_y);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::Width)}, kPinSize);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::Height)}, kPinSize);

        pin_idx += n_pins;
    }

    auto pin_to_cell = torch::zeros({total_pins}, long_options);
    pin_idx = 0;
    for (int64_t cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        const int64_t n_pins = num_pins_per_cell[cell_idx].item<int64_t>();
        pin_to_cell.index_put_({Slice(pin_idx, pin_idx + n_pins)}, cell_idx);
        pin_idx += n_pins;
    }

    std::vector<std::pair<int64_t, int64_t>> edges;
    std::vector<std::unordered_set<int64_t>> adjacency(total_pins);
    for (int64_t pin = 0; pin < total_pins; ++pin) {
        const int64_t num_connections =
            torch::randint(1, 4, {1}, long_options)[0].item<int64_t>();
        for (int64_t connection = 0; connection < num_connections; ++connection) {
            const int64_t other_pin =
                torch::randint(0, total_pins, {1}, long_options)[0].item<int64_t>();
            if (other_pin == pin || adjacency[pin].contains(other_pin)) {
                continue;
            }
            edges.emplace_back(std::min(pin, other_pin), std::max(pin, other_pin));
            adjacency[pin].insert(other_pin);
            adjacency[other_pin].insert(pin);
        }
    }

    torch::Tensor edge_list;
    if (edges.empty()) {
        edge_list = torch::zeros({0, 2}, long_options);
    } else {
        std::vector<int64_t> flat_edges;
        flat_edges.reserve(edges.size() * 2);
        for (const auto& edge : edges) {
            flat_edges.push_back(edge.first);
            flat_edges.push_back(edge.second);
        }
        edge_list = torch::from_blob(
                        flat_edges.data(),
                        {static_cast<int64_t>(edges.size()), 2},
                        torch::TensorOptions().dtype(torch::kInt64))
                        .clone()
                        .to(device);
    }

    return {cell_features, pin_features, edge_list};
}

void initializeCellPositions(torch::Tensor& cell_features, const double spread_scale = 0.6) {
    const int64_t total_cells = cell_features.size(0);
    const double total_area =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Area)})
            .sum()
            .item<double>();
    const double spread_radius =
        std::max(std::sqrt(total_area) * spread_scale, 1.0);
    const auto options = cell_features.options();

    const auto angles = torch::rand({total_cells}, options) * kTwoPi;
    const auto radii = torch::rand({total_cells}, options) * spread_radius;
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::X)},
        radii * torch::cos(angles));
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::Y)},
        radii * torch::sin(angles));
}

}  // namespace

int main() {
    constexpr int num_macros = 3;
    constexpr int num_std_cells = 10;
    constexpr uint64_t seed = 42;

    torch::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);
    runJitSmokeCheck();

    const torch::Device device(torch::kCUDA);
    PlacementProblem problem = generatePlacementInput(num_macros, num_std_cells, device);
    initializeCellPositions(problem.cell_features);

    torch::cuda::synchronize();

    return 0;
}
