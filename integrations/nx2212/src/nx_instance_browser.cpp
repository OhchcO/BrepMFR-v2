#include "nx_instance_browser.hpp"

#include "nx_part_adapter.hpp"

#include <NXOpen/BlockStyler_BlockDialog.hxx>
#include <NXOpen/BlockStyler_CompositeBlock.hxx>
#include <NXOpen/BlockStyler_Node.hxx>
#include <NXOpen/BlockStyler_PropertyList.hxx>
#include <NXOpen/BlockStyler_Tree.hxx>
#include <NXOpen/BlockStyler_UIBlock.hxx>
#include <NXOpen/Body.hxx>
#include <NXOpen/BodyCollection.hxx>
#include <NXOpen/Face.hxx>
#include <NXOpen/ListingWindow.hxx>
#include <NXOpen/NXMessageBox.hxx>
#include <NXOpen/NXString.hxx>
#include <NXOpen/Part.hxx>
#include <NXOpen/PartCollection.hxx>
#include <NXOpen/Session.hxx>
#include <NXOpen/UI.hxx>

#include <filesystem>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr char kDialogTitleUtf8[] = u8"BrepMFR 特征实例浏览器";
constexpr char kNoSelectionUtf8[] = u8"请选择一个特征实例";

NXOpen::NXString Utf8(const std::string& value)
{
    return NXOpen::NXString(value, NXOpen::NXString::UTF8);
}

void SetBlockValue(
    NXOpen::BlockStyler::UIBlock* block,
    const std::string& value)
{
    if (block == nullptr)
    {
        throw std::runtime_error("instance browser UI block is unavailable");
    }
    std::unique_ptr<NXOpen::BlockStyler::PropertyList> properties(
        block->GetProperties());
    if (!properties)
    {
        throw std::runtime_error(
            "instance browser UI properties are unavailable");
    }
    properties->SetString("Value", Utf8(value));
}

void ShowBrowserError(const std::string& message) noexcept
{
    try
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui != nullptr && ui->NXMessageBox() != nullptr)
        {
            ui->NXMessageBox()->Show(
                NXOpen::NXString(kDialogTitleUtf8, NXOpen::NXString::UTF8),
                NXOpen::NXMessageBox::DialogTypeError,
                Utf8(message));
        }
    }
    catch (...)
    {
    }
}

template <typename Value>
std::string JoinValues(const std::vector<Value>& values)
{
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
        {
            stream << ", ";
        }
        stream << values[index];
    }
    stream << ']';
    return stream.str();
}

std::string FormatSeconds(std::chrono::milliseconds elapsed)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1)
           << static_cast<double>(elapsed.count()) / 1000.0;
    return stream.str();
}

std::string FormatSummary(const InstanceBrowserSummary& summary)
{
    std::ostringstream stream;
    stream << u8"面数量：" << summary.face_count
           << u8"    特征实例：" << summary.instance_count
           << u8"    BIN 缓存：" << (summary.cache_hit ? u8"命中" : u8"未命中")
           << '\n'
           << u8"流水线用时：" << std::fixed << std::setprecision(1)
           << summary.pipeline_seconds
           << u8" 秒    Python 进程总用时："
           << FormatSeconds(summary.process_elapsed) << u8" 秒"
           << '\n'
           << u8"已映射面：" << summary.mapped_faces << '/'
           << summary.face_count
           << u8"    已着色实例：" << summary.colored_instances
           << u8"    映射用时：" << FormatSeconds(summary.mapping_elapsed)
           << u8" 秒";
    return stream.str();
}

std::string FormatDetails(const InstanceBrowserRow& row)
{
    std::ostringstream stream;
    stream << u8"实例：" << row.instance_id << '\n'
           << u8"特征类型：" << row.feature_name_utf8 << '\n'
           << u8"置信度：" << std::fixed << std::setprecision(4)
           << row.confidence << '\n'
           << u8"OCC/AAG 面：" << JoinValues(row.occ_face_indices) << '\n'
           << u8"NX Face Tags：" << JoinValues(row.nx_face_tags);
    return stream.str();
}

std::unordered_map<FaceKey, NXOpen::Face*> CurrentBodyFacesByTag(
    const PartSnapshot& expected)
{
    ValidateNxCurrentPart(expected);
    NXOpen::Session* session = NXOpen::Session::GetSession();
    if (session == nullptr || session->Parts() == nullptr)
    {
        throw std::runtime_error(
            "NX session or Part collection is unavailable");
    }
    NXOpen::Part* part = session->Parts()->Work();
    if (part == nullptr || part->Bodies() == nullptr ||
        static_cast<FaceKey>(part->Tag()) != expected.part_tag)
    {
        throw std::runtime_error(
            "current Work Part no longer matches recognition");
    }

    std::vector<NXOpen::Body*> bodies;
    for (NXOpen::Body* body : *part->Bodies())
    {
        if (body != nullptr)
        {
            bodies.push_back(body);
        }
    }
    if (bodies.size() != 1 ||
        static_cast<FaceKey>(bodies.front()->Tag()) != expected.body_tag)
    {
        throw std::runtime_error(
            "current body no longer matches recognition");
    }

    const auto faces = bodies.front()->GetFaces();
    if (faces.size() != expected.face_count)
    {
        throw std::runtime_error(
            "current body face count no longer matches recognition");
    }
    std::unordered_map<FaceKey, NXOpen::Face*> faces_by_tag;
    faces_by_tag.reserve(faces.size());
    for (NXOpen::Face* face : faces)
    {
        if (face == nullptr ||
            !faces_by_tag.emplace(
                static_cast<FaceKey>(face->Tag()), face).second)
        {
            throw std::runtime_error(
                "current body contains a null or duplicate NX face");
        }
    }
    return faces_by_tag;
}

class NxTemporaryFaceHighlighter final : public TemporaryHighlightSink
{
public:
    explicit NxTemporaryFaceHighlighter(PartSnapshot expected)
        : expected_(std::move(expected))
    {
    }

    ~NxTemporaryFaceHighlighter() override
    {
        Clear();
    }

    void Replace(const std::vector<FaceKey>& face_tags) override
    {
        Clear();
        if (face_tags.empty())
        {
            throw std::runtime_error(
                "temporary highlight requires at least one NX face");
        }
        const auto faces_by_tag = CurrentBodyFacesByTag(expected_);
        std::set<FaceKey> unique_tags;
        std::vector<NXOpen::Face*> targets;
        targets.reserve(face_tags.size());
        for (const FaceKey tag : face_tags)
        {
            if (!unique_tags.insert(tag).second)
            {
                throw std::runtime_error(
                    "temporary highlight contains duplicate NX face tags");
            }
            const auto found = faces_by_tag.find(tag);
            if (found == faces_by_tag.end())
            {
                throw std::runtime_error(
                    "temporary highlight references a face outside the current body");
            }
            targets.push_back(found->second);
        }

        try
        {
            for (NXOpen::Face* face : targets)
            {
                face->Highlight();
                highlighted_.push_back(face);
            }
        }
        catch (...)
        {
            Clear();
            throw;
        }
    }

    void Clear() noexcept override
    {
        for (NXOpen::Face* face : highlighted_)
        {
            try
            {
                if (face != nullptr)
                {
                    face->Unhighlight();
                }
            }
            catch (...)
            {
            }
        }
        highlighted_.clear();
    }

private:
    PartSnapshot expected_;
    std::vector<NXOpen::Face*> highlighted_;
};

class TreeRedrawGuard
{
public:
    explicit TreeRedrawGuard(NXOpen::BlockStyler::Tree* tree)
        : tree_(tree)
    {
        tree_->Redraw(false);
    }

    ~TreeRedrawGuard()
    {
        try
        {
            tree_->Redraw(true);
        }
        catch (...)
        {
        }
    }

    TreeRedrawGuard(const TreeRedrawGuard&) = delete;
    TreeRedrawGuard& operator=(const TreeRedrawGuard&) = delete;

private:
    NXOpen::BlockStyler::Tree* tree_;
};

class NxInstanceBrowserDialog
{
public:
    NxInstanceBrowserDialog(
        const std::filesystem::path& dlx_path,
        PartSnapshot part,
        const InstanceBrowserModel& model)
        : model_(model),
          highlighter_(std::move(part)),
          session_(model_, highlighter_)
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui == nullptr)
        {
            throw std::runtime_error("NX UI is unavailable");
        }
        dialog_.reset(
            ui->CreateDialog(Utf8(dlx_path.generic_u8string())));
        if (dialog_ == nullptr)
        {
            throw std::runtime_error(
                "NX failed to create the feature instance browser");
        }
        dialog_->AddInitializeHandler(
            NXOpen::make_callback(
                this, &NxInstanceBrowserDialog::Initialize));
        dialog_->AddDialogShownHandler(
            NXOpen::make_callback(
                this, &NxInstanceBrowserDialog::DialogShown));
        dialog_->AddUpdateHandler(
            NXOpen::make_callback(
                this, &NxInstanceBrowserDialog::Update));
        dialog_->AddOkHandler(
            NXOpen::make_callback(this, &NxInstanceBrowserDialog::Ok));
        dialog_->AddCloseHandler(
            NXOpen::make_callback(
                this, &NxInstanceBrowserDialog::Close));
    }

    ~NxInstanceBrowserDialog()
    {
        session_.Close();
    }

    void Launch()
    {
        try
        {
            dialog_->Launch();
        }
        catch (...)
        {
            session_.Close();
            throw;
        }
        session_.Close();
    }

private:
    NXOpen::BlockStyler::UIBlock* FindRequiredBlock(const char* block_id)
    {
        if (dialog_ == nullptr || dialog_->TopBlock() == nullptr)
        {
            throw std::runtime_error(
                "instance browser top block is unavailable");
        }
        NXOpen::BlockStyler::UIBlock* block =
            dialog_->TopBlock()->FindBlock(block_id);
        if (block == nullptr)
        {
            throw std::runtime_error(
                std::string("instance browser block is missing: ") +
                block_id);
        }
        return block;
    }

    void Initialize()
    {
        try
        {
            summary_ = FindRequiredBlock("summary");
            tree_ = dynamic_cast<NXOpen::BlockStyler::Tree*>(
                FindRequiredBlock("instance_tree"));
            paging_group_ = FindRequiredBlock("paging_group");
            previous_page_ = FindRequiredBlock("previous_page");
            page_label_ = FindRequiredBlock("page_label");
            next_page_ = FindRequiredBlock("next_page");
            details_ = FindRequiredBlock("details");
            show_full_result_ = FindRequiredBlock("show_full_result");
            if (tree_ == nullptr)
            {
                throw std::runtime_error(
                    "instance_tree is not an NX Tree block");
            }
            tree_->SetOnSelectHandler(
                NXOpen::make_callback(
                    this, &NxInstanceBrowserDialog::OnSelect));
            paging_group_->SetShow(
                session_.State().PaginationVisible());
        }
        catch (const std::exception& error)
        {
            DisableAfterError(error.what());
        }
        catch (...)
        {
            DisableAfterError(
                "unknown error while initializing the instance browser");
        }
    }

    void DialogShown()
    {
        if (disabled_after_error_)
        {
            return;
        }
        try
        {
            tree_->InsertColumn(0, Utf8(u8"实例"), 70);
            tree_->InsertColumn(1, Utf8(u8"特征类型"), 180);
            tree_->InsertColumn(2, Utf8(u8"面数"), 70);
            tree_->InsertColumn(3, Utf8(u8"置信度"), 100);
            for (int column = 0; column < 4; ++column)
            {
                tree_->SetColumnSortable(column, false);
            }
            tree_->SetColumnResizePolicy(
                0,
                NXOpen::BlockStyler::Tree::
                    ColumnResizePolicyConstantWidth);
            tree_->SetColumnResizePolicy(
                1,
                NXOpen::BlockStyler::Tree::
                    ColumnResizePolicyResizeWithTree);
            tree_->SetColumnResizePolicy(
                2,
                NXOpen::BlockStyler::Tree::
                    ColumnResizePolicyConstantWidth);
            tree_->SetColumnResizePolicy(
                3,
                NXOpen::BlockStyler::Tree::
                    ColumnResizePolicyConstantWidth);
            SetBlockValue(summary_, FormatSummary(model_.summary));
            ResetDetails();
            RefreshPage();
        }
        catch (const std::exception& error)
        {
            DisableAfterError(error.what());
        }
        catch (...)
        {
            DisableAfterError(
                "unknown error while showing the instance browser");
        }
    }

    int Update(NXOpen::BlockStyler::UIBlock* block)
    {
        if (disabled_after_error_ && block != show_full_result_)
        {
            return 0;
        }
        try
        {
            if (block == previous_page_ && session_.GoPrevious())
            {
                ResetDetails();
                RefreshPage();
            }
            else if (block == next_page_ && session_.GoNext())
            {
                ResetDetails();
                RefreshPage();
            }
            else if (block == show_full_result_)
            {
                ShowFullResult();
            }
        }
        catch (const std::exception& error)
        {
            ShowBrowserError(
                std::string(u8"实例浏览器操作失败：") + error.what());
        }
        catch (...)
        {
            ShowBrowserError(u8"实例浏览器操作发生未知错误");
        }
        return 0;
    }

    int Ok()
    {
        session_.Close();
        return 0;
    }

    int Close()
    {
        session_.Close();
        return 0;
    }

    void OnSelect(
        NXOpen::BlockStyler::Tree*,
        NXOpen::BlockStyler::Node* node,
        int,
        bool selected)
    {
        if (!selected || disabled_after_error_ || node == nullptr)
        {
            return;
        }
        const auto found = visible_row_by_node_.find(node);
        if (found == visible_row_by_node_.end())
        {
            return;
        }
        int instance_id = -1;
        try
        {
            const std::size_t global_index =
                session_.State().GlobalIndexForVisibleRow(found->second);
            instance_id = model_.rows.at(global_index).instance_id;
            const InstanceBrowserRow& row =
                session_.SelectVisibleRow(found->second);
            SetBlockValue(details_, FormatDetails(row));
        }
        catch (const std::exception& error)
        {
            HandleHighlightFailure(instance_id, error.what());
        }
        catch (...)
        {
            HandleHighlightFailure(
                instance_id, "unknown NX highlight error");
        }
    }

    void RefreshPage()
    {
        session_.ClearSelection();
        TreeRedrawGuard redraw(tree_);
        visible_row_by_node_.clear();
        while (NXOpen::BlockStyler::Node* node = tree_->RootNode())
        {
            tree_->DeleteNode(node);
        }

        const InstanceBrowserState& state = session_.State();
        for (std::size_t global_index = state.PageBegin();
             global_index < state.PageEnd();
             ++global_index)
        {
            const InstanceBrowserRow& row = model_.rows.at(global_index);
            NXOpen::BlockStyler::Node* node =
                tree_->CreateNode(std::to_string(row.instance_id).c_str());
            if (node == nullptr)
            {
                throw std::runtime_error(
                    "NX failed to create an instance tree node");
            }
            node->SetColumnDisplayText(1, Utf8(row.feature_name_utf8));
            node->SetColumnDisplayText(
                2, std::to_string(row.nx_face_tags.size()).c_str());
            std::ostringstream confidence;
            confidence << std::fixed << std::setprecision(4)
                       << row.confidence;
            node->SetColumnDisplayText(3, confidence.str().c_str());
            tree_->InsertNode(
                node,
                nullptr,
                nullptr,
                NXOpen::BlockStyler::Tree::NodeInsertOptionLast);
            visible_row_by_node_.emplace(
                node, global_index - state.PageBegin());
        }
        UpdatePagination();
    }

    void UpdatePagination()
    {
        const InstanceBrowserState& state = session_.State();
        paging_group_->SetShow(state.PaginationVisible());
        previous_page_->SetEnable(state.CanGoPrevious());
        next_page_->SetEnable(state.CanGoNext());
        std::ostringstream label;
        label << u8"第 " << state.CurrentPage() + 1 << " / "
              << state.PageCount() << u8" 页";
        SetBlockValue(page_label_, label.str());
    }

    void ResetDetails()
    {
        SetBlockValue(details_, kNoSelectionUtf8);
    }

    void ShowFullResult()
    {
        NXOpen::Session* nx_session = NXOpen::Session::GetSession();
        if (nx_session == nullptr || nx_session->ListingWindow() == nullptr)
        {
            throw std::runtime_error("NX Listing Window is unavailable");
        }
        NXOpen::ListingWindow* listing = nx_session->ListingWindow();
        listing->Open();
        while (full_result_lines_written_ <
               model_.complete_text.listing_lines_utf8.size())
        {
            listing->WriteFullline(Utf8(
                model_.complete_text.listing_lines_utf8[
                    full_result_lines_written_]));
            ++full_result_lines_written_;
        }
    }

    void HandleHighlightFailure(
        int instance_id,
        const std::string& diagnostic) noexcept
    {
        session_.ClearSelection();
        try
        {
            ResetDetails();
        }
        catch (...)
        {
        }
        try
        {
            if (tree_ != nullptr)
            {
                tree_->SetEnable(false);
            }
            if (previous_page_ != nullptr)
            {
                previous_page_->SetEnable(false);
            }
            if (next_page_ != nullptr)
            {
                next_page_->SetEnable(false);
            }
        }
        catch (...)
        {
        }
        disabled_after_error_ = true;
        try
        {
            std::ostringstream message;
            message << u8"实例 " << instance_id
                    << u8" 的 NX 临时高亮失败：\n" << diagnostic;
            ShowBrowserError(message.str());
        }
        catch (...)
        {
            ShowBrowserError(u8"NX 临时高亮失败，且错误详情无法显示");
        }
    }

    void DisableAfterError(const std::string& diagnostic) noexcept
    {
        session_.Close();
        disabled_after_error_ = true;
        try
        {
            if (tree_ != nullptr)
            {
                tree_->SetEnable(false);
            }
            if (previous_page_ != nullptr)
            {
                previous_page_->SetEnable(false);
            }
            if (next_page_ != nullptr)
            {
                next_page_->SetEnable(false);
            }
        }
        catch (...)
        {
            ShowBrowserError(
                u8"无法初始化特征实例浏览器，且错误详情无法显示");
            return;
        }
        try
        {
            ShowBrowserError(
                std::string(u8"无法初始化特征实例浏览器：\n") +
                diagnostic);
        }
        catch (...)
        {
            ShowBrowserError(
                u8"无法初始化特征实例浏览器，且错误详情无法显示");
        }
    }

    const InstanceBrowserModel& model_;
    NxTemporaryFaceHighlighter highlighter_;
    InstanceBrowserSession session_;
    std::unique_ptr<NXOpen::BlockStyler::BlockDialog> dialog_;
    NXOpen::BlockStyler::UIBlock* summary_ = nullptr;
    NXOpen::BlockStyler::Tree* tree_ = nullptr;
    NXOpen::BlockStyler::UIBlock* paging_group_ = nullptr;
    NXOpen::BlockStyler::UIBlock* previous_page_ = nullptr;
    NXOpen::BlockStyler::UIBlock* page_label_ = nullptr;
    NXOpen::BlockStyler::UIBlock* next_page_ = nullptr;
    NXOpen::BlockStyler::UIBlock* details_ = nullptr;
    NXOpen::BlockStyler::UIBlock* show_full_result_ = nullptr;
    std::unordered_map<NXOpen::BlockStyler::Node*, std::size_t>
        visible_row_by_node_;
    std::size_t full_result_lines_written_ = 0;
    bool disabled_after_error_ = false;
};
}

void ShowNxInstanceBrowser(
    const std::filesystem::path& dlx_path,
    const PartSnapshot& part,
    const InstanceBrowserModel& model)
{
    std::error_code error;
    if (!std::filesystem::is_regular_file(dlx_path, error))
    {
        throw std::runtime_error(
            "NX instance browser layout is unavailable: " +
            dlx_path.generic_u8string());
    }
    NxInstanceBrowserDialog dialog(dlx_path, part, model);
    dialog.Launch();
}
}
}
