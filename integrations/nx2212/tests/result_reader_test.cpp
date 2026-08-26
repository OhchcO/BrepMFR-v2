#include "result_reader.hpp"

#include "test_support.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace
{
std::string FixtureText(const char* filename)
{
    const auto fixture_path = std::filesystem::path(__FILE__).parent_path() /
        "fixtures" / filename;
    return brepmfr::nx2212::test::ReadUtf8(fixture_path);
}

std::filesystem::path MaterializeFixture(
    brepmfr::nx2212::test::TemporaryDirectory& temporary,
    const char* filename,
    const std::filesystem::path& selected_step)
{
    auto text = FixtureText(filename);
    const std::string placeholder = "__STEP_PATH__";
    const auto position = text.find(placeholder);
    brepmfr::nx2212::test::Expect(
        position != std::string::npos,
        "fixture STEP placeholder was missing");
    text.replace(position, placeholder.size(), selected_step.generic_u8string());
    return temporary.WriteUtf8(filename, text);
}

nlohmann::json FixtureDocument(
    const char* filename,
    const std::filesystem::path& selected_step)
{
    auto text = FixtureText(filename);
    const std::string placeholder = "__STEP_PATH__";
    const auto position = text.find(placeholder);
    brepmfr::nx2212::test::Expect(
        position != std::string::npos,
        "fixture STEP placeholder was missing");
    text.replace(position, placeholder.size(), selected_step.generic_u8string());
    return nlohmann::json::parse(text);
}

void ExpectRejected(
    brepmfr::nx2212::test::TemporaryDirectory& temporary,
    const std::filesystem::path& selected_step,
    const nlohmann::json& document,
    const std::string& case_name)
{
    const auto result_path = temporary.WriteUtf8(
        "invalid-result.json", document.dump(2));
    bool rejected = false;
    try
    {
        (void)brepmfr::nx2212::ReadInferenceResult(
            result_path, selected_step);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    brepmfr::nx2212::test::Expect(
        rejected, case_name + " was accepted");
}
}

BREPMFR_TEST(ReadInferenceResultBuildsTypedSuccess)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    const auto result_path = MaterializeFixture(
        temporary, "result_success.json", selected_step);

    const auto parsed = ReadInferenceResult(result_path, selected_step);
    Expect(
        std::holds_alternative<SuccessResult>(parsed),
        "success document was not parsed as success");
    const auto& success = std::get<SuccessResult>(parsed);
    ExpectEqual(success.face_count, 2, "face count mismatch");
    ExpectEqual(success.faces.size(), static_cast<std::size_t>(2), "faces mismatch");
    ExpectEqual(
        success.instances.size(),
        static_cast<std::size_t>(1),
        "instances mismatch");
    Expect(success.cache_hit, "cache hit was not preserved");
    ExpectEqual(success.total_seconds, 6.0, "total time mismatch");
    ExpectEqual(
        success.instances[0].face_indices,
        std::vector<int>{1},
        "instance face indices mismatch");
    Expect(
        success.mapping.status == MappingPayloadStatus::Unsupported,
        "JSON 1.0 unexpectedly enabled mapping");
}

BREPMFR_TEST(ReadInferenceResultBuildsReadyFaceMappingFromJson11)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    const auto result_path = MaterializeFixture(
        temporary, "result_success_1_1.json", selected_step);

    const auto parsed = ReadInferenceResult(result_path, selected_step);
    const auto& success = std::get<SuccessResult>(parsed);
    ExpectEqual(success.schema_version, std::string("1.1"), "schema mismatch");
    Expect(
        success.mapping.status == MappingPayloadStatus::Ready,
        success.mapping.diagnostic);
    ExpectEqual(
        success.mapping.graph.faces.size(),
        static_cast<std::size_t>(2),
        "mapping face count mismatch");
    ExpectEqual(success.mapping.graph.faces[0].key, FaceKey{0}, "first key mismatch");
    ExpectEqual(success.mapping.graph.faces[1].key, FaceKey{1}, "second key mismatch");
    Expect(
        success.mapping.graph.faces[0].surface.origin.has_value(),
        "plane origin missing");
    Expect(
        success.mapping.graph.faces[0].surface.axis.has_value(),
        "plane axis missing");
    ExpectEqual(
        *success.mapping.graph.faces[0].surface.axis,
        Vec3{0.0, 0.0, 1.0},
        "plane axis mismatch");
    ExpectEqual(
        success.mapping.graph.faces[1].sample_points,
        std::vector<Vec3>{{0.5, 0.0, 0.0}},
        "sample points mismatch");
    ExpectEqual(
        success.mapping.graph.adjacency_edges,
        std::vector<AdjacencyEdge>{{0, 1}},
        "adjacency mismatch");
}

BREPMFR_TEST(ReadInferenceResultBuildsUnavailableFaceMappingFromJson11)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    auto document = FixtureDocument("result_success_1_1.json", selected_step);
    document["face_mapping"] = {
        {"contract_version", "1.0"},
        {"status", "unavailable"},
        {"reason", "geometry signatures unavailable"}};
    const auto path = temporary.WriteUtf8("unavailable.json", document.dump(2));

    const auto parsed = ReadInferenceResult(path, selected_step);
    const auto& success = std::get<SuccessResult>(parsed);
    Expect(
        success.mapping.status == MappingPayloadStatus::Unavailable,
        "unavailable mapping status mismatch");
    ExpectContains(
        success.mapping.diagnostic,
        "geometry signatures unavailable",
        "unavailable reason missing");
    Expect(success.mapping.graph.faces.empty(), "unavailable mapping leaked faces");
}

BREPMFR_TEST(ReadInferenceResultQuarantinesCorruptJson11Mapping)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;
    using Json = nlohmann::json;
    using Mutation = std::function<void(Json&)>;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    const std::vector<std::pair<std::string, Mutation>> cases = {
        {"missing field", [](Json& value) {
             value["face_mapping"]["faces"][0].erase("area");
         }},
        {"non-finite numeric payload", [](Json& value) {
             value["face_mapping"]["faces"][0]["area"] = "NaN";
         }},
        {"wrong normalization", [](Json& value) {
             value["face_mapping"]["normalization"]["box_side"] = 2.0;
         }},
        {"unavailable without reason", [](Json& value) {
             value["face_mapping"] = {
                 {"contract_version", "1.0"},
                 {"status", "unavailable"},
                 {"reason", ""}};
         }},
        {"duplicate key", [](Json& value) {
             value["face_mapping"]["faces"][1]["face_index"] = 0;
         }},
        {"empty samples", [](Json& value) {
             value["face_mapping"]["faces"][0]["sample_points"] = Json::array();
         }},
        {"bad adjacency", [](Json& value) {
             value["face_mapping"]["adjacency_edges"] = Json::array({{0, 0}});
         }},
        {"mapping face count mismatch", [](Json& value) {
             value["face_mapping"]["faces"].erase(
                 value["face_mapping"]["faces"].begin() + 1);
             value["face_mapping"]["adjacency_edges"] = Json::array();
         }}};

    for (const auto& test_case : cases)
    {
        auto document = FixtureDocument("result_success_1_1.json", selected_step);
        test_case.second(document);
        const auto result_path = temporary.WriteUtf8(
            test_case.first + ".json", document.dump(2));
        const auto parsed = ReadInferenceResult(result_path, selected_step);
        Expect(
            std::holds_alternative<SuccessResult>(parsed),
            test_case.first + " destroyed the core success result");
        const auto& success = std::get<SuccessResult>(parsed);
        ExpectEqual(success.faces.size(), static_cast<std::size_t>(2),
                    test_case.first + " changed core faces");
        ExpectEqual(success.instances.size(), static_cast<std::size_t>(1),
                    test_case.first + " changed core instances");
        Expect(
            success.mapping.status == MappingPayloadStatus::Invalid,
            test_case.first + " was not quarantined");
        Expect(
            !success.mapping.diagnostic.empty(),
            test_case.first + " has no diagnostic");
        Expect(
            success.mapping.graph.faces.empty() &&
                success.mapping.graph.adjacency_edges.empty(),
            test_case.first + " leaked a partial mapping graph");
    }
}

BREPMFR_TEST(ReadInferenceResultBuildsTypedError)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    const auto result_path = MaterializeFixture(
        temporary, "result_error.json", selected_step);

    const auto parsed = ReadInferenceResult(result_path, selected_step);
    Expect(
        std::holds_alternative<ErrorResult>(parsed),
        "error document was not parsed as error");
    const auto& error = std::get<ErrorResult>(parsed);
    ExpectEqual(error.stage, std::string("conversion"), "error stage mismatch");
    ExpectEqual(error.error_type, std::string("ValueError"), "error type mismatch");
    ExpectEqual(error.message, std::string("shape mismatch"), "error message mismatch");
}

BREPMFR_TEST(ReadInferenceResultRejectsInvalidSuccessSemantics)
{
    using namespace brepmfr::nx2212::test;
    using Json = nlohmann::json;
    using Mutation = std::function<void(Json&)>;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    const std::vector<std::pair<std::string, Mutation>> cases = {
        {"unsupported schema version", [](Json& value) {
             value["schema_version"] = "1.2";
         }},
        {"missing status", [](Json& value) {
             value.erase("status");
         }},
        {"STEP path mismatch", [](Json& value) {
             value["input"]["step_path"] = "C:/different/part.step";
         }},
        {"face count mismatch", [](Json& value) {
             value["input"]["face_count"] = 3;
         }},
        {"duplicate face index", [](Json& value) {
             value["faces"][1]["face_index"] = 0;
         }},
        {"non-contiguous face index", [](Json& value) {
             value["faces"][1]["face_index"] = 2;
         }},
        {"wrong confidence type", [](Json& value) {
             value["faces"][0]["confidence"] = "high";
         }},
        {"duplicate instance id", [](Json& value) {
             value["instances"].push_back(value["instances"][0]);
         }},
        {"unsorted instance face indices", [](Json& value) {
             value["faces"][0]["class_id"] = 12;
             value["faces"][0]["feature_name"] = "blind_hole";
             value["faces"][0]["instance_id"] = 0;
             value["instances"][0]["face_indices"] = Json::array({1, 0});
         }},
        {"duplicate instance face index", [](Json& value) {
             value["instances"][0]["face_indices"] = Json::array({1, 1});
         }},
        {"nonexistent instance face", [](Json& value) {
             value["instances"][0]["face_indices"] = Json::array({2});
         }},
        {"unknown per-face instance id", [](Json& value) {
             value["faces"][1]["instance_id"] = 4;
         }},
        {"instance-to-face id mismatch", [](Json& value) {
             value["instances"][0]["face_indices"] = Json::array({0});
         }},
        {"instance class mismatch", [](Json& value) {
             value["instances"][0]["class_id"] = 11;
         }},
        {"instance feature name mismatch", [](Json& value) {
             value["instances"][0]["feature_name"] = "through_hole";
         }}};

    for (const auto& test_case : cases)
    {
        auto document = FixtureDocument("result_success.json", selected_step);
        test_case.second(document);
        ExpectRejected(
            temporary, selected_step, document, test_case.first);
    }
}

BREPMFR_TEST(ReadInferenceResultRejectsNonFiniteConfidence)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    auto text = FixtureText("result_success.json");
    const std::string placeholder = "__STEP_PATH__";
    text.replace(
        text.find(placeholder),
        placeholder.size(),
        selected_step.generic_u8string());
    const std::string finite = "\"confidence\": 0.99";
    text.replace(text.find(finite), finite.size(), "\"confidence\": 1e9999");
    const auto result_path = temporary.WriteUtf8("nonfinite.json", text);

    bool rejected = false;
    try
    {
        (void)ReadInferenceResult(result_path, selected_step);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    Expect(rejected, "non-finite confidence was accepted");
}

BREPMFR_TEST(ReadInferenceResultRejectsInvalidErrorSemantics)
{
    using namespace brepmfr::nx2212::test;
    using Json = nlohmann::json;
    using Mutation = std::function<void(Json&)>;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    const std::vector<std::pair<std::string, Mutation>> cases = {
        {"empty error stage", [](Json& value) {
             value["stage"] = "";
         }},
        {"empty error type", [](Json& value) {
             value["error"]["type"] = "";
         }},
        {"missing error message", [](Json& value) {
             value["error"].erase("message");
         }},
        {"empty error message", [](Json& value) {
             value["error"]["message"] = "";
         }},
        {"error STEP path mismatch", [](Json& value) {
             value["input"]["step_path"] = "C:/different/part.step";
         }}};

    for (const auto& test_case : cases)
    {
        auto document = FixtureDocument("result_error.json", selected_step);
        test_case.second(document);
        ExpectRejected(
            temporary, selected_step, document, test_case.first);
    }
}

BREPMFR_TEST(ReadInferenceResultAllowsExtensionsAndSortsInstances)
{
    using namespace brepmfr::nx2212;
    using namespace brepmfr::nx2212::test;

    TemporaryDirectory temporary;
    const auto selected_step = temporary.WriteUtf8("selected.step", "STEP");
    auto document = FixtureDocument("result_success.json", selected_step);
    auto upper_path = selected_step.generic_u8string();
    for (char& character : upper_path)
    {
        if (character >= 'a' && character <= 'z')
        {
            character = static_cast<char>(character - 'a' + 'A');
        }
    }
    document["input"]["step_path"] = upper_path;
    document["input"]["face_count"] = 3;
    document["faces"].push_back({
        {"face_index", 2},
        {"class_id", 2},
        {"feature_name", "triangular_passage"},
        {"confidence", 0.8},
        {"instance_id", 2}});
    const nlohmann::json second_instance = {
        {"instance_id", 2},
        {"class_id", 2},
        {"feature_name", "triangular_passage"},
        {"face_indices", nlohmann::json::array({2})},
        {"confidence", 0.8}};
    document["instances"].insert(
        document["instances"].begin(), second_instance);
    document["future_extension"] = {"allowed", true};
    const auto result_path = temporary.WriteUtf8(
        "extended-result.json", document.dump(2));

    const auto parsed = ReadInferenceResult(result_path, selected_step);
    const auto& success = std::get<SuccessResult>(parsed);
    ExpectEqual(success.instances.size(), static_cast<std::size_t>(2), "instance count mismatch");
    ExpectEqual(success.instances[0].instance_id, 0, "instances were not sorted");
    ExpectEqual(success.instances[1].instance_id, 2, "instances were not sorted");
}
