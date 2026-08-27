#include "plugin_runtime.hpp"

#include "instance_browser.hpp"
#include "output_paths.hpp"

#include "test_support.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace
{
nlohmann::json SuccessDocument(const std::filesystem::path& step_path)
{
    return {
        {"schema_version", "1.0"},
        {"status", "success"},
        {"input",
         {{"step_path", step_path.generic_u8string()},
          {"step_sha256", "abc"},
          {"face_count", 2}}},
        {"cache",
         {{"bin_path", "D:/output/input.bin"},
          {"metadata_path", "D:/output/input.bin.meta.json"},
          {"hit", true},
          {"fingerprint", "fp"}}},
        {"model",
         {{"checkpoint_path", "D:/model.ckpt"},
          {"checkpoint_sha256", "def"},
          {"checkpoint_epoch", 187},
          {"experiment_mode", "dynamic_affinity"},
          {"device", "cuda"},
          {"num_classes", 25},
          {"affinity_threshold", 0.5}}},
        {"faces",
         {{{"face_index", 0},
           {"class_id", 24},
           {"feature_name", "stock"},
           {"confidence", 0.99},
           {"instance_id", nullptr}},
          {{"face_index", 1},
           {"class_id", 12},
           {"feature_name", "blind_hole"},
           {"confidence", 0.95},
           {"instance_id", 0}}}},
        {"instances",
         {{{"instance_id", 0},
           {"class_id", 12},
           {"feature_name", "blind_hole"},
           {"face_indices", nlohmann::json::array({1})},
           {"confidence", 0.95}}}},
        {"timings_seconds",
         {{"checkpoint_load", 1.0},
          {"conversion", 2.0},
          {"inference", 3.0},
          {"total", 6.0}}},
        {"warnings", nlohmann::json::array()}};
}

nlohmann::json ErrorDocument(const std::filesystem::path& step_path)
{
    return {
        {"schema_version", "1.0"},
        {"status", "error"},
        {"stage", "conversion"},
        {"error", {{"type", "ValueError"}, {"message", "shape mismatch"}}},
        {"input", {{"step_path", step_path.generic_u8string()}}}};
}

nlohmann::json SignatureJson(
    int face_index,
    double marker,
    int neighbor_count)
{
    const double x = marker * 0.1;
    const double y = marker * 0.02;
    return {
        {"face_index", face_index},
        {"surface_type", "plane"},
        {"area", 1.0 + marker},
        {"perimeter", 4.0 + marker},
        {"centroid", {x, y, 0.0}},
        {"bounding_box", {x - 0.01, y - 0.01, 0.0, x + 0.01, y + 0.01, 0.0}},
        {"loop_count", 1},
        {"edge_count", 4},
        {"neighbor_count", neighbor_count},
        {"surface_parameters",
         {{"origin", {x, y, 0.0}}, {"axis", {0.0, 0.0, 1.0}}}},
        {"sample_points", nlohmann::json::array({{x, y, 0.0}})}};
}

nlohmann::json ReadyMapping(
    int face_count = 2,
    bool duplicate_geometry = false,
    bool connected = true)
{
    nlohmann::json faces = nlohmann::json::array();
    for (int index = 0; index < face_count; ++index)
    {
        const double marker = duplicate_geometry ? 0.0 : static_cast<double>(index);
        faces.push_back(SignatureJson(
            index, marker, connected ? 1 : 0));
    }
    return {
        {"contract_version", "1.0"},
        {"status", "ready"},
        {"normalization",
         {{"kind", "aag_unit_box"},
          {"box_side", 1.0},
          {"scale_rule", "2_over_longest_bbox_extent"}}},
        {"faces", faces},
        {"adjacency_edges",
         connected
             ? nlohmann::json::array({{0, 1}})
             : nlohmann::json::array()}};
}

nlohmann::json SuccessDocument11(
    const std::filesystem::path& step_path,
    const nlohmann::json& mapping)
{
    auto document = SuccessDocument(step_path);
    document["schema_version"] = "1.1";
    document["face_mapping"] = mapping;
    return document;
}

nlohmann::json LargeColorFailureDocument(
    const std::filesystem::path& step_path)
{
    auto document = SuccessDocument(step_path);
    document["schema_version"] = "1.1";
    document["input"]["face_count"] = 217;
    document["faces"] = nlohmann::json::array();
    document["instances"] = nlohmann::json::array();
    for (int index = 0; index < 217; ++index)
    {
        document["faces"].push_back({
            {"face_index", index},
            {"class_id", 12},
            {"feature_name", "feature"},
            {"confidence", 1.0},
            {"instance_id", index}});
        document["instances"].push_back({
            {"instance_id", index},
            {"class_id", 12},
            {"feature_name", "feature"},
            {"face_indices", nlohmann::json::array({index})},
            {"confidence", 1.0}});
    }
    document["face_mapping"] = ReadyMapping(217, false, false);
    return document;
}

brepmfr::nx2212::FaceSignature NativeFace(
    brepmfr::nx2212::FaceKey tag,
    double marker,
    int neighbor_count)
{
    using namespace brepmfr::nx2212;
    const double x = marker * 0.1;
    const double y = marker * 0.02;
    FaceSignature face{};
    face.key = tag;
    face.surface_type = SurfaceType::Plane;
    face.area = 1.0 + marker;
    face.perimeter = 4.0 + marker;
    face.centroid = {x, y, 0.0};
    face.bounds = {{x - 0.01, y - 0.01, 0.0}, {x + 0.01, y + 0.01, 0.0}};
    face.loop_count = 1;
    face.edge_count = 4;
    face.neighbor_count = neighbor_count;
    face.surface.origin = Vec3{x, y, 0.0};
    face.surface.axis = Vec3{0.0, 0.0, 1.0};
    face.sample_points = {{x, y, 0.0}};
    return face;
}

brepmfr::nx2212::FaceGraph NativeGraph(
    int face_count = 2,
    bool duplicate_geometry = false,
    bool connected = true)
{
    using namespace brepmfr::nx2212;
    FaceGraph graph;
    for (int index = 0; index < face_count; ++index)
    {
        const double marker = duplicate_geometry ? 0.0 : static_cast<double>(index);
        graph.faces.push_back(NativeFace(
            static_cast<FaceKey>(500 + index),
            marker,
            connected ? 1 : 0));
    }
    if (connected)
    {
        graph.adjacency_edges = {{500, 501}};
    }
    return graph;
}

struct ModuleFixture
{
    explicit ModuleFixture(brepmfr::nx2212::test::TemporaryDirectory& temporary)
        : module_directory(temporary.Path() / "module"),
          python(module_directory / "python.exe"),
          script(module_directory / "infer_step.py"),
          checkpoint(module_directory / "model.ckpt"),
          output_root(temporary.Path() / "recognition-output")
    {
        std::filesystem::create_directories(module_directory);
        temporary.WriteFile("module/python.exe", "");
        temporary.WriteFile("module/infer_step.py", "");
        temporary.WriteFile("module/model.ckpt", "");
        const nlohmann::json config = {
            {"schema_version", "1.0"},
            {"python_executable", python.generic_u8string()},
            {"inference_script", script.generic_u8string()},
            {"checkpoint", checkpoint.generic_u8string()},
            {"output_root", output_root.generic_u8string()},
            {"device", "auto"},
            {"affinity_threshold", 0.5}};
        temporary.WriteUtf8(
            "module/BrepMFRNx2212.config.json", config.dump(2));
    }

    std::filesystem::path module_directory;
    std::filesystem::path python;
    std::filesystem::path script;
    std::filesystem::path checkpoint;
    std::filesystem::path output_root;
};

class RecordingHost : public brepmfr::nx2212::PluginHost
{
public:
    explicit RecordingHost(std::filesystem::path module_directory)
        : module_directory_(std::move(module_directory))
    {
    }

    std::filesystem::path ModuleDirectory() const override
    {
        return module_directory_;
    }

    brepmfr::nx2212::PartSnapshot CaptureCurrentPart() override
    {
        ++capture_calls;
        if (capture_behavior)
        {
            return capture_behavior();
        }
        return part;
    }

    brepmfr::nx2212::FaceGraph ReadCurrentFaceGraph(
        const brepmfr::nx2212::PartSnapshot& snapshot) override
    {
        ++graph_calls;
        graph_snapshot = snapshot;
        return face_graph;
    }

    void ApplyFaceColorsAtomically(
        const brepmfr::nx2212::PartSnapshot& snapshot,
        const std::vector<brepmfr::nx2212::FaceColorAssignment>& assignments) override
    {
        ++color_calls;
        color_snapshot = snapshot;
        color_assignments = assignments;
        if (color_behavior)
        {
            color_behavior(snapshot, assignments);
        }
    }

    brepmfr::nx2212::ProcessExecution RunProcess(
        const std::filesystem::path& executable,
        const std::vector<std::wstring>& arguments,
        const std::filesystem::path& log_path,
        const brepmfr::nx2212::ProcessLaunchOptions& options) override
    {
        ++process_calls;
        process_executable = executable;
        process_arguments = arguments;
        process_log_path = log_path;
        process_options = options;
        if (process_behavior)
        {
            return process_behavior(executable, arguments, log_path, options);
        }
        return process_execution;
    }

    void ShowSuccess(const brepmfr::nx2212::DisplayText& display) override
    {
        ++success_calls;
        success_display = display;
    }

    void ShowInstanceBrowser(
        const brepmfr::nx2212::PartSnapshot& snapshot,
        const brepmfr::nx2212::InstanceBrowserModel& model) override
    {
        ++browser_calls;
        browser_snapshot = snapshot;
        browser_model = model;
    }

    void ShowError(const std::string& message_utf8) noexcept override
    {
        ++error_calls;
        error_message = message_utf8;
    }

    std::filesystem::path module_directory_;
    brepmfr::nx2212::PartSnapshot part{{}, 10, 20, 2};
    brepmfr::nx2212::FaceGraph face_graph;
    std::function<brepmfr::nx2212::PartSnapshot()> capture_behavior;
    std::function<void(
        const brepmfr::nx2212::PartSnapshot&,
        const std::vector<brepmfr::nx2212::FaceColorAssignment>&)> color_behavior;
    brepmfr::nx2212::ProcessExecution process_execution{
        0, std::chrono::milliseconds(7500)};
    std::function<brepmfr::nx2212::ProcessExecution(
        const std::filesystem::path&,
        const std::vector<std::wstring>&,
        const std::filesystem::path&,
        const brepmfr::nx2212::ProcessLaunchOptions&)> process_behavior;
    int capture_calls = 0;
    int graph_calls = 0;
    int color_calls = 0;
    int process_calls = 0;
    int success_calls = 0;
    int browser_calls = 0;
    int error_calls = 0;
    std::filesystem::path process_executable;
    std::vector<std::wstring> process_arguments;
    std::filesystem::path process_log_path;
    brepmfr::nx2212::ProcessLaunchOptions process_options;
    brepmfr::nx2212::DisplayText success_display;
    brepmfr::nx2212::PartSnapshot browser_snapshot;
    brepmfr::nx2212::InstanceBrowserModel browser_model;
    std::string error_message;
    brepmfr::nx2212::PartSnapshot graph_snapshot;
    brepmfr::nx2212::PartSnapshot color_snapshot;
    std::vector<brepmfr::nx2212::FaceColorAssignment> color_assignments;
};

brepmfr::nx2212::OutputPaths ConfigureDocumentProcess(
    brepmfr::nx2212::test::TemporaryDirectory& temporary,
    const ModuleFixture& module,
    RecordingHost& host,
    nlohmann::json document,
    unsigned long exit_code = 0)
{
    using namespace brepmfr::nx2212;
    const auto output = BuildOutputPaths(module.output_root, host.part.step_path);
    host.process_behavior = [&, output, document = std::move(document), exit_code](
                                const std::filesystem::path&,
                                const std::vector<std::wstring>&,
                                const std::filesystem::path&,
                                const ProcessLaunchOptions&) {
        temporary.WriteUtf8(
            output.result_path.lexically_relative(temporary.Path()),
            document.dump(2));
        return ProcessExecution{exit_code, std::chrono::milliseconds(7500)};
    };
    return output;
}
}

BREPMFR_TEST(InferenceWorkflowStopsWhenCurrentWorkPartIsUnavailable)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    host.capture_behavior = []() -> PartSnapshot {
        throw std::runtime_error("Work Part unavailable");
    };

    const int result = RunInferenceWorkflow(host);

    ExpectEqual(result, kFailure, "missing Work Part returned success");
    ExpectEqual(host.capture_calls, 1, "Work Part capture call count mismatch");
    ExpectEqual(host.process_calls, 0, "missing Work Part launched Python");
    ExpectEqual(host.graph_calls, 0, "missing Work Part read an NX graph");
    ExpectEqual(host.color_calls, 0, "missing Work Part applied colors");
    ExpectEqual(host.success_calls, 0, "missing Work Part displayed success");
    ExpectEqual(host.browser_calls, 0, "missing Work Part opened browser");
    ExpectEqual(host.error_calls, 1, "missing Work Part error call count mismatch");
    ExpectContains(host.error_message, "Work Part unavailable",
                   "missing Work Part diagnostic missing");
}

BREPMFR_TEST(InferenceWorkflowRunsPythonAndDisplaysFreshSuccess)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    const auto selected_step = temporary.WriteUtf8("parts/sample.step", "STEP");
    host.part.step_path = selected_step;
    const auto output = BuildOutputPaths(module.output_root, selected_step);
    std::filesystem::create_directories(output.directory);
    temporary.WriteUtf8(
        output.result_path.lexically_relative(temporary.Path()),
        "stale result must be removed");

    const std::vector<std::wstring> expected_arguments = {
        L"-u",
        module.script.wstring(),
        L"--step",
        selected_step.wstring(),
        L"--checkpoint",
        module.checkpoint.wstring(),
        L"--output-dir",
        output.directory.wstring(),
        L"--device",
        L"auto",
        L"--affinity-threshold",
        L"0.5"};
    host.process_behavior = [&](const std::filesystem::path& executable,
                                const std::vector<std::wstring>& arguments,
                                const std::filesystem::path& log_path,
                                const ProcessLaunchOptions&) {
        ExpectEqual(executable, module.python, "Python executable mismatch");
        ExpectEqual(arguments, expected_arguments, "Python arguments mismatch");
        ExpectEqual(log_path, output.log_path, "log path mismatch");
        Expect(
            std::filesystem::is_directory(output.directory),
            "output directory was not created");
        Expect(
            !std::filesystem::exists(output.result_path),
            "stale result.json was not removed before launch");
        temporary.WriteUtf8(
            output.result_path.lexically_relative(temporary.Path()),
            SuccessDocument(selected_step).dump(2));
        return ProcessExecution{0, std::chrono::milliseconds(7500)};
    };

    const int workflow_result = RunInferenceWorkflow(host);

    ExpectEqual(workflow_result, kSuccess, "successful workflow returned failure");
    ExpectEqual(host.process_calls, 1, "Python process call count mismatch");
    ExpectEqual(host.capture_calls, 1, "Work Part was not captured once");
    ExpectEqual(host.success_calls, 1, "success display call count mismatch");
    ExpectEqual(host.browser_calls, 0, "JSON 1.0 success opened browser");
    ExpectEqual(host.error_calls, 0, "successful workflow displayed an error");
    ExpectEqual(
        host.process_options.working_directory,
        module.script.parent_path().parent_path(),
        "workflow did not derive the Python project working directory");
    ExpectEqual(
        host.process_options.display_input_path,
        selected_step,
        "workflow did not pass the selected STEP to the progress UI");
    Expect(
        host.process_options.isolate_python_environment,
        "workflow did not request Python environment isolation");
    Expect(
        host.success_display.summary_utf8.find("面数量：2") != std::string::npos,
        "success display did not contain parsed result");
    bool found_process_timing = false;
    for (const auto& line : host.success_display.listing_lines_utf8)
    {
        found_process_timing = found_process_timing ||
            line == u8"进程总用时：7.5 秒；识别流水线：6.0 秒";
    }
    Expect(found_process_timing, "success display omitted process duration");
}

BREPMFR_TEST(InferenceWorkflowMapsAndColorsReadyJson11Once)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    host.part.step_path = temporary.WriteUtf8("parts/current.step", "STEP");
    host.face_graph = NativeGraph();
    ConfigureDocumentProcess(
        temporary,
        module,
        host,
        SuccessDocument11(host.part.step_path, ReadyMapping()));

    const int result = RunInferenceWorkflow(host);

    ExpectEqual(result, kSuccess, "ready JSON 1.1 workflow failed");
    ExpectEqual(host.graph_calls, 1, "NX graph was not read once");
    ExpectEqual(host.color_calls, 1, "complete mapping was not colored once");
    ExpectEqual(host.color_assignments.size(), static_cast<std::size_t>(1),
                "feature color assignment count mismatch");
    ExpectEqual(host.color_assignments[0].nx_face_tag, FaceKey{501},
                "recognized feature mapped to wrong NX Face Tag");
    ExpectEqual(host.color_assignments[0].instance_id, 0,
                "feature instance ID mismatch");
    ExpectEqual(host.color_assignments[0].color_index, 1,
                "feature color mismatch");
    ExpectEqual(host.success_calls, 0, "colored result opened legacy popup");
    ExpectEqual(host.browser_calls, 1, "colored success did not open browser");
    ExpectEqual(host.browser_snapshot.body_tag, host.part.body_tag,
                "browser snapshot mismatch");
    ExpectEqual(host.browser_model.rows.size(), std::size_t{1},
                "browser row count mismatch");
    ExpectEqual(host.browser_model.rows[0].nx_face_tags,
                std::vector<FaceKey>{501}, "browser face tags mismatch");
    ExpectEqual(host.error_calls, 0, "colored result displayed Python error");
    ExpectContains(host.browser_model.complete_text.summary_utf8,
                   u8"已着色实例：1",
                   "color summary missing");
    ExpectEqual(host.process_options.display_input_path, host.part.step_path,
                "progress path did not use current Work Part");
}

BREPMFR_TEST(InferenceWorkflowDoesNotReadNxGraphForUnavailableOrInvalidMapping)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    for (const bool invalid : {false, true})
    {
        TemporaryDirectory temporary;
        ModuleFixture module(temporary);
        RecordingHost host(module.module_directory);
        host.part.step_path = temporary.WriteUtf8("current.step", "STEP");
        nlohmann::json mapping;
        if (invalid)
        {
            mapping = ReadyMapping();
            mapping["faces"][0].erase("area");
        }
        else
        {
            mapping = {
                {"contract_version", "1.0"},
                {"status", "unavailable"},
                {"reason", "geometry signatures unavailable"}};
        }
        ConfigureDocumentProcess(
            temporary,
            module,
            host,
            SuccessDocument11(host.part.step_path, mapping));

        const int result = RunInferenceWorkflow(host);
        ExpectEqual(result, kSuccess, "mapping warning changed recognition status");
        ExpectEqual(host.graph_calls, 0, "unusable mapping read the NX graph");
        ExpectEqual(host.color_calls, 0, "unusable mapping applied colors");
        ExpectEqual(host.success_calls, 1, "mapping warning hid recognition result");
        ExpectEqual(host.browser_calls, 0,
                    "mapping warning opened browser");
        ExpectEqual(host.error_calls, 0, "mapping warning became Python error");
        ExpectContains(host.success_display.summary_utf8,
                       u8"识别成功，但面映射失败",
                       "mapping warning summary missing");
    }
}

BREPMFR_TEST(InferenceWorkflowKeepsTextWhenMatcherFailsOrIsAmbiguous)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    for (const bool ambiguous : {false, true})
    {
        TemporaryDirectory temporary;
        ModuleFixture module(temporary);
        RecordingHost host(module.module_directory);
        host.part.step_path = temporary.WriteUtf8("current.step", "STEP");
        const auto mapping = ambiguous
            ? ReadyMapping(2, true, false)
            : ReadyMapping();
        host.face_graph = ambiguous
            ? NativeGraph(2, true, false)
            : NativeGraph();
        if (!ambiguous)
        {
            host.face_graph.faces[0].centroid.x += 1e-3;
        }
        ConfigureDocumentProcess(
            temporary,
            module,
            host,
            SuccessDocument11(host.part.step_path, mapping));

        const int result = RunInferenceWorkflow(host);
        ExpectEqual(result, kSuccess, "matcher failure changed recognition status");
        ExpectEqual(host.graph_calls, 1, "matcher case did not read NX graph");
        ExpectEqual(host.color_calls, 0, "failed matcher applied colors");
        ExpectEqual(host.success_calls, 1, "failed matcher hid text result");
        ExpectEqual(host.browser_calls, 0,
                    "failed matcher opened browser");
        ExpectEqual(host.error_calls, 0, "failed matcher became Python error");
        ExpectContains(host.success_display.summary_utf8,
                       u8"识别成功，但面映射失败",
                       "matcher failure summary missing");
        if (ambiguous)
        {
            ExpectContains(host.success_display.summary_utf8, "ambiguous",
                           "ambiguity diagnostic missing");
        }
    }
}

BREPMFR_TEST(InferenceWorkflowDoesNotColorWhenColorPlanExceedsPalette)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    host.part.step_path = temporary.WriteUtf8("large.step", "STEP");
    host.part.face_count = 217;
    host.face_graph = NativeGraph(217, false, false);
    ConfigureDocumentProcess(
        temporary,
        module,
        host,
        LargeColorFailureDocument(host.part.step_path));

    const int result = RunInferenceWorkflow(host);

    ExpectEqual(result, kSuccess, "color plan failure changed recognition status");
    ExpectEqual(host.graph_calls, 1, "color plan case did not read NX graph");
    ExpectEqual(host.color_calls, 0, "failed color plan applied colors");
    ExpectEqual(host.success_calls, 1, "failed color plan hid text result");
    ExpectEqual(host.browser_calls, 0, "failed color plan opened browser");
    ExpectContains(host.success_display.summary_utf8, "216",
                   "palette capacity diagnostic missing");
}

BREPMFR_TEST(InferenceWorkflowConvertsAtomicColorExceptionToMappingFailure)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    ModuleFixture module(temporary);
    RecordingHost host(module.module_directory);
    host.part.step_path = temporary.WriteUtf8("current.step", "STEP");
    host.face_graph = NativeGraph();
    host.color_behavior = [](const PartSnapshot&,
                             const std::vector<FaceColorAssignment>&) {
        throw std::runtime_error("synthetic color failure");
    };
    ConfigureDocumentProcess(
        temporary,
        module,
        host,
        SuccessDocument11(host.part.step_path, ReadyMapping()));

    const int result = RunInferenceWorkflow(host);

    ExpectEqual(result, kSuccess, "color exception changed recognition status");
    ExpectEqual(host.color_calls, 1, "atomic coloring was retried");
    ExpectEqual(host.success_calls, 1, "color exception hid text result");
    ExpectEqual(host.browser_calls, 0, "color exception opened browser");
    ExpectEqual(host.error_calls, 0, "color exception became Python error");
    ExpectContains(host.success_display.summary_utf8, "synthetic color failure",
                   "color exception diagnostic missing");
}

BREPMFR_TEST(InferenceWorkflowReportsAllPostLaunchFailureBoundaries)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    enum class Outcome
    {
        NonzeroError,
        ZeroError,
        NonzeroSuccess,
        MissingJson,
        MalformedJson,
        StandardException,
        UnknownException
    };
    struct Case
    {
        const char* name;
        Outcome outcome;
        std::vector<std::string> expected_text;
    };
    const std::vector<Case> cases = {
        {"nonzero plus error",
         Outcome::NonzeroError,
         {"conversion", "ValueError", "shape mismatch", "7"}},
        {"zero plus error",
         Outcome::ZeroError,
         {"status=error", "exit code 0"}},
        {"nonzero plus success",
         Outcome::NonzeroSuccess,
         {"status=success", "exit code 7"}},
        {"missing fresh JSON",
         Outcome::MissingJson,
         {"could not open result.json", "Python exit code: 0"}},
        {"malformed JSON",
         Outcome::MalformedJson,
         {"JSON parse failed", "Python exit code: 0"}},
        {"standard exception",
         Outcome::StandardException,
         {"synthetic process failure"}},
        {"unknown exception",
         Outcome::UnknownException,
         {"unknown error"}}};

    for (const auto& test_case : cases)
    {
        TemporaryDirectory temporary;
        ModuleFixture module(temporary);
        RecordingHost host(module.module_directory);
        const auto selected_step = temporary.WriteUtf8("part.step", "STEP");
        host.part.step_path = selected_step;
        const auto output = BuildOutputPaths(module.output_root, selected_step);
        host.process_behavior = [&](const std::filesystem::path&,
                                    const std::vector<std::wstring>&,
                                    const std::filesystem::path&,
                                    const ProcessLaunchOptions&) -> ProcessExecution {
            const auto relative =
                output.result_path.lexically_relative(temporary.Path());
            switch (test_case.outcome)
            {
            case Outcome::NonzeroError:
                temporary.WriteUtf8(
                    relative, ErrorDocument(selected_step).dump(2));
                return {7, std::chrono::milliseconds(100)};
            case Outcome::ZeroError:
                temporary.WriteUtf8(
                    relative, ErrorDocument(selected_step).dump(2));
                return {0, std::chrono::milliseconds(100)};
            case Outcome::NonzeroSuccess:
                temporary.WriteUtf8(
                    relative, SuccessDocument(selected_step).dump(2));
                return {7, std::chrono::milliseconds(100)};
            case Outcome::MissingJson:
                return {0, std::chrono::milliseconds(100)};
            case Outcome::MalformedJson:
                temporary.WriteUtf8(relative, "{broken json");
                return {0, std::chrono::milliseconds(100)};
            case Outcome::StandardException:
                throw std::runtime_error("synthetic process failure");
            case Outcome::UnknownException:
                throw 42;
            }
            throw std::runtime_error("unhandled test outcome");
        };

        const int workflow_result = RunInferenceWorkflow(host);
        ExpectEqual(
            workflow_result,
            kFailure,
            std::string(test_case.name) + " returned success");
        ExpectEqual(
            host.success_calls,
            0,
            std::string(test_case.name) + " displayed success");
        ExpectEqual(
            host.error_calls,
            1,
            std::string(test_case.name) + " error call count mismatch");
        for (const auto& expected : test_case.expected_text)
        {
            Expect(
                host.error_message.find(expected) != std::string::npos,
                std::string(test_case.name) + " omitted '" + expected + "'");
        }
        Expect(
            host.error_message.find(output.result_path.generic_u8string()) !=
                std::string::npos,
            std::string(test_case.name) + " omitted result path");
        Expect(
            host.error_message.find(output.log_path.generic_u8string()) !=
                std::string::npos,
            std::string(test_case.name) + " omitted log path");
    }
}

BREPMFR_TEST(InferenceWorkflowReportsConfigPathBeforeOutputDerivation)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto module_directory = temporary.Path() / "missing-module";
    RecordingHost host(module_directory);
    const auto expected_config =
        module_directory / "BrepMFRNx2212.config.json";

    const int workflow_result = RunInferenceWorkflow(host);

    ExpectEqual(workflow_result, kFailure, "missing config returned success");
    ExpectEqual(host.process_calls, 0, "missing config launched Python");
    ExpectEqual(host.error_calls, 1, "missing config error call count mismatch");
    Expect(
        host.error_message.find(expected_config.generic_u8string()) !=
            std::string::npos,
        "configuration failure did not identify the config path");
}
