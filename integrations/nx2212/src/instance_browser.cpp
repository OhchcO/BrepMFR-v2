#include "instance_browser.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace brepmfr
{
namespace nx2212
{
InstanceBrowserModel BuildInstanceBrowserModel(
    const SuccessResult& recognition,
    std::chrono::milliseconds process_elapsed,
    const MappingDisplay& mapping,
    DisplayText complete_text)
{
    if (mapping.status != MappingDisplayStatus::Colored)
    {
        throw std::invalid_argument(
            "instance browser requires a colored mapping");
    }
    if (mapping.mapped_faces != static_cast<std::size_t>(recognition.face_count))
    {
        throw std::invalid_argument(
            "colored mapping does not cover every recognition face");
    }
    std::unordered_map<int, const InstanceResult*> recognition_by_id;
    recognition_by_id.reserve(recognition.instances.size());
    for (const auto& instance : recognition.instances)
    {
        if (!recognition_by_id.emplace(instance.instance_id, &instance).second)
        {
            throw std::invalid_argument(
                "duplicate recognition instance ID " +
                std::to_string(instance.instance_id));
        }
    }

    std::unordered_map<int, const InstanceColorRecord*> color_by_id;
    color_by_id.reserve(mapping.instances.size());
    std::unordered_set<FaceKey> assigned_face_tags;
    for (const auto& color : mapping.instances)
    {
        if (recognition_by_id.find(color.instance_id) == recognition_by_id.end())
        {
            throw std::invalid_argument(
                "color record references unknown recognition instance " +
                std::to_string(color.instance_id));
        }
        if (!color_by_id.emplace(color.instance_id, &color).second)
        {
            throw std::invalid_argument(
                "duplicate color record for recognition instance " +
                std::to_string(color.instance_id));
        }
        if (color.nx_face_tags.empty())
        {
            throw std::invalid_argument(
                "recognition instance " + std::to_string(color.instance_id) +
                " has no NX faces");
        }
        for (const FaceKey tag : color.nx_face_tags)
        {
            if (!assigned_face_tags.insert(tag).second)
            {
                throw std::invalid_argument(
                    "duplicate NX face tag " + std::to_string(tag));
            }
        }
    }

    if (mapping.colored_instances != recognition.instances.size())
    {
        throw std::invalid_argument(
            "colored mapping instance count does not match recognition");
    }

    InstanceBrowserModel model;
    model.summary.face_count = recognition.face_count;
    model.summary.instance_count = recognition.instances.size();
    model.summary.cache_hit = recognition.cache_hit;
    model.summary.pipeline_seconds = recognition.total_seconds;
    model.summary.process_elapsed = process_elapsed;
    model.summary.mapped_faces = mapping.mapped_faces;
    model.summary.colored_instances = mapping.colored_instances;
    model.summary.mapping_elapsed = mapping.elapsed;
    model.complete_text = std::move(complete_text);
    model.rows.reserve(recognition.instances.size());

    for (const auto& instance : recognition.instances)
    {
        const auto found = color_by_id.find(instance.instance_id);
        if (found == color_by_id.end())
        {
            throw std::invalid_argument(
                "missing color record for recognition instance " +
                std::to_string(instance.instance_id));
        }
        const auto& color = *found->second;
        InstanceBrowserRow row;
        row.instance_id = instance.instance_id;
        row.feature_name_utf8 = instance.feature_name;
        row.confidence = instance.confidence;
        row.occ_face_indices = instance.face_indices;
        row.nx_face_tags = color.nx_face_tags;
        row.color_index = color.color_index;
        model.rows.push_back(std::move(row));
    }

    std::sort(
        model.rows.begin(), model.rows.end(),
        [](const InstanceBrowserRow& left, const InstanceBrowserRow& right) {
            return left.instance_id < right.instance_id;
        });
    return model;
}

InstanceBrowserState::InstanceBrowserState(std::size_t row_count)
    : row_count_(row_count)
{
}

std::size_t InstanceBrowserState::CurrentPage() const noexcept
{
    return current_page_;
}

std::size_t InstanceBrowserState::PageCount() const noexcept
{
    if (row_count_ == 0)
    {
        return 1;
    }
    return row_count_ / kInstanceBrowserPageSize +
        (row_count_ % kInstanceBrowserPageSize == 0 ? 0 : 1);
}

std::size_t InstanceBrowserState::PageBegin() const noexcept
{
    return current_page_ * kInstanceBrowserPageSize;
}

std::size_t InstanceBrowserState::PageEnd() const noexcept
{
    return (std::min)(row_count_, PageBegin() + kInstanceBrowserPageSize);
}

bool InstanceBrowserState::PaginationVisible() const noexcept
{
    return PageCount() > 1;
}

bool InstanceBrowserState::CanGoPrevious() const noexcept
{
    return current_page_ > 0;
}

bool InstanceBrowserState::CanGoNext() const noexcept
{
    return current_page_ + 1 < PageCount();
}

bool InstanceBrowserState::GoPrevious()
{
    if (!CanGoPrevious())
    {
        return false;
    }
    --current_page_;
    ClearSelection();
    return true;
}

bool InstanceBrowserState::GoNext()
{
    if (!CanGoNext())
    {
        return false;
    }
    ++current_page_;
    ClearSelection();
    return true;
}

std::size_t InstanceBrowserState::GlobalIndexForVisibleRow(
    std::size_t visible_row) const
{
    if (visible_row >= PageEnd() - PageBegin())
    {
        throw std::out_of_range("visible browser row is outside current page");
    }
    return PageBegin() + visible_row;
}

void InstanceBrowserState::SelectGlobalIndex(std::size_t global_index)
{
    if (global_index < PageBegin() || global_index >= PageEnd())
    {
        throw std::out_of_range(
            "browser row selection is outside current page");
    }
    selected_global_index_ = global_index;
}

void InstanceBrowserState::ClearSelection() noexcept
{
    selected_global_index_.reset();
}

std::optional<std::size_t> InstanceBrowserState::SelectedGlobalIndex() const noexcept
{
    return selected_global_index_;
}
}
}
