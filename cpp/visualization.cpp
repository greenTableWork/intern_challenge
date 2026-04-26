#include "placement/visualization.h"

#include "placement/metrics.h"
#include "placement/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace torch::indexing;

constexpr double kSvgWidth = 1200.0;
constexpr double kSvgHeight = 600.0;
constexpr double kPanelWidth = 560.0;
constexpr double kPlotTop = 95.0;
constexpr double kPlotWidth = 500.0;
constexpr double kPlotHeight = 455.0;
constexpr double kWorldMargin = 10.0;

int64_t featureIndex(placement::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

bool isUsableCoordinate(double value) {
    constexpr double kMaxCoordinate = 1.0e12;
    return std::isfinite(value) && std::abs(value) <= kMaxCoordinate;
}

std::string formatDouble(double value, int precision = 2) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

struct CellRect {
    double center_x = 0.0;
    double center_y = 0.0;
    double width = 0.0;
    double height = 0.0;
    bool has_finite_center = false;
    bool drawable = false;
};

struct Bounds {
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    void include(double x, double y) {
        if (!isUsableCoordinate(x) || !isUsableCoordinate(y)) {
            return;
        }

        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    bool valid() const {
        return min_x <= max_x && min_y <= max_y && std::isfinite(min_x) &&
               std::isfinite(max_x) && std::isfinite(min_y) && std::isfinite(max_y);
    }
};

struct PanelData {
    torch::Tensor cells;
    std::vector<CellRect> rects;
    Bounds bounds;
    placement::OverlapMetrics metrics;
};

struct FinalBounds {
    double min_x = -10.0;
    double max_x = 10.0;
    double min_y = -10.0;
    double max_y = 10.0;
};

struct Transform {
    FinalBounds bounds;
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double scale = 1.0;
    double x_padding = 0.0;
    double y_padding = 0.0;

    double svgX(double world_x) const {
        return x + x_padding + (world_x - bounds.min_x) * scale;
    }

    double svgY(double world_y) const {
        return y + y_padding + (bounds.max_y - world_y) * scale;
    }
};

torch::Tensor prepareCellFeatures(const torch::Tensor& cell_features) {
    if (!cell_features.defined()) {
        throw std::invalid_argument("cell feature tensor must be defined");
    }
    if (cell_features.dim() != 2) {
        throw std::invalid_argument("cell feature tensor must be two-dimensional");
    }
    if (cell_features.size(0) > 0 &&
        cell_features.size(1) <= featureIndex(placement::CellFeatureIdx::Height)) {
        throw std::invalid_argument(
            "cell feature tensor must contain area, pins, x, y, width, and height");
    }

    return cell_features.detach()
        .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
        .contiguous();
}

CellRect readRect(const torch::Tensor& cells, int64_t row) {
    CellRect rect;
    rect.center_x =
        cells.index({row, featureIndex(placement::CellFeatureIdx::X)}).item<double>();
    rect.center_y =
        cells.index({row, featureIndex(placement::CellFeatureIdx::Y)}).item<double>();
    rect.width =
        cells.index({row, featureIndex(placement::CellFeatureIdx::Width)})
            .item<double>();
    rect.height =
        cells.index({row, featureIndex(placement::CellFeatureIdx::Height)})
            .item<double>();

    rect.has_finite_center =
        isUsableCoordinate(rect.center_x) && isUsableCoordinate(rect.center_y);
    rect.drawable = rect.has_finite_center && isUsableCoordinate(rect.width) &&
                    isUsableCoordinate(rect.height) && rect.width > 0.0 &&
                    rect.height > 0.0;
    return rect;
}

PanelData buildPanelData(const torch::Tensor& cell_features) {
    PanelData panel;
    panel.cells = prepareCellFeatures(cell_features);
    panel.metrics = placement::calculateOverlapMetrics(panel.cells);

    const int64_t num_cells = panel.cells.size(0);
    panel.rects.reserve(static_cast<std::size_t>(num_cells));
    for (int64_t index = 0; index < num_cells; ++index) {
        CellRect rect = readRect(panel.cells, index);
        if (rect.drawable) {
            panel.bounds.include(rect.center_x - rect.width / 2.0, rect.center_y);
            panel.bounds.include(rect.center_x + rect.width / 2.0, rect.center_y);
            panel.bounds.include(rect.center_x, rect.center_y - rect.height / 2.0);
            panel.bounds.include(rect.center_x, rect.center_y + rect.height / 2.0);
        } else if (rect.has_finite_center) {
            panel.bounds.include(rect.center_x, rect.center_y);
        }
        panel.rects.push_back(rect);
    }

    return panel;
}

FinalBounds finalizeBounds(const Bounds& bounds) {
    if (!bounds.valid()) {
        return {};
    }

    FinalBounds final_bounds{
        bounds.min_x - kWorldMargin,
        bounds.max_x + kWorldMargin,
        bounds.min_y - kWorldMargin,
        bounds.max_y + kWorldMargin,
    };

    if (final_bounds.min_x >= final_bounds.max_x) {
        final_bounds.min_x -= 1.0;
        final_bounds.max_x += 1.0;
    }
    if (final_bounds.min_y >= final_bounds.max_y) {
        final_bounds.min_y -= 1.0;
        final_bounds.max_y += 1.0;
    }

    return final_bounds;
}

Transform makeTransform(
    const FinalBounds& bounds,
    double plot_x,
    double plot_y,
    double plot_width,
    double plot_height) {
    const double world_width = std::max(bounds.max_x - bounds.min_x, 1.0);
    const double world_height = std::max(bounds.max_y - bounds.min_y, 1.0);
    const double scale = std::min(plot_width / world_width, plot_height / world_height);
    const double used_width = world_width * scale;
    const double used_height = world_height * scale;

    return {
        bounds,
        plot_x,
        plot_y,
        plot_width,
        plot_height,
        scale,
        (plot_width - used_width) / 2.0,
        (plot_height - used_height) / 2.0,
    };
}

void writeText(
    std::ostream& output,
    double x,
    double y,
    const std::string& text,
    int font_size,
    const std::string& anchor = "middle",
    const std::string& weight = "normal") {
    output << "<text x=\"" << formatDouble(x) << "\" y=\"" << formatDouble(y)
           << "\" text-anchor=\"" << anchor << "\" font-family=\"Arial, sans-serif\""
           << " font-size=\"" << font_size << "\" font-weight=\"" << weight
           << "\" fill=\"#111827\">" << text << "</text>\n";
}

void writeGrid(std::ostream& output, const Transform& transform) {
    constexpr int kGridLines = 5;
    output << "<g stroke=\"#94a3b8\" stroke-width=\"0.75\" opacity=\"0.35\">\n";
    for (int index = 0; index <= kGridLines; ++index) {
        const double ratio = static_cast<double>(index) / kGridLines;
        const double world_x =
            transform.bounds.min_x +
            (transform.bounds.max_x - transform.bounds.min_x) * ratio;
        const double x = transform.svgX(world_x);
        output << "<line x1=\"" << formatDouble(x) << "\" y1=\""
               << formatDouble(transform.y) << "\" x2=\"" << formatDouble(x)
               << "\" y2=\"" << formatDouble(transform.y + transform.height)
               << "\" />\n";

        const double world_y =
            transform.bounds.min_y +
            (transform.bounds.max_y - transform.bounds.min_y) * ratio;
        const double y = transform.svgY(world_y);
        output << "<line x1=\"" << formatDouble(transform.x) << "\" y1=\""
               << formatDouble(y) << "\" x2=\""
               << formatDouble(transform.x + transform.width) << "\" y2=\""
               << formatDouble(y) << "\" />\n";
    }
    output << "</g>\n";
}

void writeCellRects(
    std::ostream& output,
    const std::vector<CellRect>& rects,
    const Transform& transform) {
    output << "<g fill=\"lightblue\" fill-opacity=\"0.7\" stroke=\"darkblue\""
           << " stroke-width=\"0.5\">\n";
    for (const CellRect& rect : rects) {
        if (!rect.drawable) {
            continue;
        }

        const double left = rect.center_x - rect.width / 2.0;
        const double top = rect.center_y + rect.height / 2.0;
        output << "<rect x=\"" << formatDouble(transform.svgX(left)) << "\" y=\""
               << formatDouble(transform.svgY(top)) << "\" width=\""
               << formatDouble(rect.width * transform.scale) << "\" height=\""
               << formatDouble(rect.height * transform.scale) << "\" />\n";
    }
    output << "</g>\n";
}

void writePanel(
    std::ostream& output,
    const PanelData& panel,
    const std::string& title,
    double panel_x) {
    const double center_x = panel_x + kPanelWidth / 2.0;
    writeText(output, center_x, 34.0, title, 18, "middle", "bold");
    writeText(
        output,
        center_x,
        58.0,
        "Overlaps: " + std::to_string(panel.metrics.overlap_count) +
            ", Total Overlap Area: " +
            formatDouble(panel.metrics.total_overlap_area),
        14);

    const Transform transform = makeTransform(
        finalizeBounds(panel.bounds),
        panel_x + 30.0,
        kPlotTop,
        kPlotWidth,
        kPlotHeight);

    output << "<rect x=\"" << formatDouble(transform.x) << "\" y=\""
           << formatDouble(transform.y) << "\" width=\""
           << formatDouble(transform.width) << "\" height=\""
           << formatDouble(transform.height)
           << "\" fill=\"#ffffff\" stroke=\"#cbd5e1\" stroke-width=\"1\" />\n";
    writeGrid(output, transform);
    writeCellRects(output, panel.rects, transform);
}

}  // namespace

namespace placement {

void plotPlacement(
    const torch::Tensor& initial_cell_features,
    const torch::Tensor& final_cell_features,
    const std::filesystem::path& output_path) {
    const PanelData initial_panel = buildPanelData(initial_cell_features);
    const PanelData final_panel = buildPanelData(final_cell_features);

    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "unable to open placement visualization output path: " +
            output_path.string());
    }

    output << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""
           << formatDouble(kSvgWidth, 0) << "\" height=\"" << formatDouble(kSvgHeight, 0)
           << "\" viewBox=\"0 0 " << formatDouble(kSvgWidth, 0) << " "
           << formatDouble(kSvgHeight, 0) << "\" role=\"img\">\n";
    output << "<title>Placement Visualization</title>\n";
    output << "<rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\" />\n";
    writePanel(output, initial_panel, "Initial Placement", 20.0);
    writePanel(output, final_panel, "Final Placement", 620.0);
    output << "</svg>\n";
}

}  // namespace placement
