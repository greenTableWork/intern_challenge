#include "placement/generation.h"

#include <stdexcept>

namespace placement {

PlacementProblem generatePlacementInput(
    int num_macros,
    int num_std_cells,
    const torch::Device& device,
    bool verbose) {
    (void)num_macros;
    (void)num_std_cells;
    (void)device;
    (void)verbose;
    throw std::logic_error("generatePlacementInput is implemented in Step 3");
}

void initializeCellPositions(torch::Tensor& cell_features, double spread_scale) {
    (void)cell_features;
    (void)spread_scale;
    throw std::logic_error("initializeCellPositions is implemented in Step 3");
}

}  // namespace placement
