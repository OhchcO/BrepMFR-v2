#pragma once

#include "result_formatter.hpp"

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
constexpr std::size_t kInstanceBrowserPageSize = 50;

struct InstanceBrowserSummary
{
    int face_count = 0;
    std::size_t instance_count = 0;
    bool cache_hit = false;
    double pipeline_seconds = 0.0;
    std::chrono::milliseconds process_elapsed{0};
    std::size_t mapped_faces = 0;
    std::size_t colored_instances = 0;
    std::chrono::milliseconds mapping_elapsed{0};
};

struct InstanceBrowserRow
{
    int instance_id = 0;
    std::string feature_name_utf8;
    double confidence = 0.0;
    std::vector<int> occ_face_indices;
    std::vector<FaceKey> nx_face_tags;
    int color_index = 0;
};

struct InstanceBrowserModel
{
    InstanceBrowserSummary summary;
    std::vector<InstanceBrowserRow> rows;
    DisplayText complete_text;
};

InstanceBrowserModel BuildInstanceBrowserModel(
    const SuccessResult& recognition,
    std::chrono::milliseconds process_elapsed,
    const MappingDisplay& mapping,
    DisplayText complete_text);

class InstanceBrowserState
{
public:
    explicit InstanceBrowserState(std::size_t row_count);

    std::size_t CurrentPage() const noexcept;
    std::size_t PageCount() const noexcept;
    std::size_t PageBegin() const noexcept;
    std::size_t PageEnd() const noexcept;
    bool PaginationVisible() const noexcept;
    bool CanGoPrevious() const noexcept;
    bool CanGoNext() const noexcept;
    bool GoPrevious();
    bool GoNext();
    std::size_t GlobalIndexForVisibleRow(std::size_t visible_row) const;
    void SelectGlobalIndex(std::size_t global_index);
    void ClearSelection() noexcept;
    std::optional<std::size_t> SelectedGlobalIndex() const noexcept;

private:
    std::size_t row_count_ = 0;
    std::size_t current_page_ = 0;
    std::optional<std::size_t> selected_global_index_;
};
}
}
