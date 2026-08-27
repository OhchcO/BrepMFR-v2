# NX2212 Feature Instance Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-shot NX2212 success popup with a modal NX-native browser that summarizes recognition, pages through instances, and temporarily highlights the selected instance without changing its base color.

**Architecture:** Keep Python, JSON 1.1, face matching, and atomic base coloring unchanged. Add a pure C++ browser model/state/session layer that is covered by the existing native runtime test executable, then add a thin NXOpen Block Styler adapter and a scoped NX Face highlighter. The existing degraded-success path remains available when mapping or coloring is unavailable; only a fully colored result opens the browser.

**Tech Stack:** C++17, NXOpen C++/UFUN for NX2212, Block Styler `.dlx`, VS2019 v142/x64, the repository's native `BREPMFR_TEST` harness, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-nx-instance-browser-design.md`

## Global Constraints

- Keep NX2212, VS2019/v142, x64, Windows 10 SDK, and C++17 as the supported native build contract.
- Keep `tools/infer_step.py`, JSON 1.1, the Python process boundary, face matching, and base coloring behavior unchanged.
- Use a modal NX Block Styler dialog loaded from `BrepMFRInstanceBrowser.dlx` next to the DLL.
- Use exactly 50 instances per page; hide pagination controls when the instance count is at most 50.
- Keep all instance base colors; temporary selection uses only `Highlight()` / `Unhighlight()`.
- Clear temporary highlight on selection changes, page changes, dialog close, and exception paths.
- Do not automatically open the Listing Window on a fully colored success; write the complete text only after the user presses `查看完整结果`.
- Do not auto-save the NX Part or add model attributes.
- Follow TDD for every pure C++ or Python-testable behavior and make one commit per task.

## File Structure

- Create `integrations/nx2212/include/instance_browser.hpp`: NX-independent browser model, pagination state, highlight sink interface, and browser session API.
- Create `integrations/nx2212/src/instance_browser.cpp`: model validation/building plus paging and session behavior.
- Create `integrations/nx2212/tests/instance_browser_test.cpp`: model, paging, selection, cleanup, and failure tests.
- Create `integrations/nx2212/include/nx_instance_browser.hpp`: public NX adapter entry point used by `NxPluginHost`.
- Create `integrations/nx2212/src/nx_instance_browser.cpp`: Block Styler dialog, Tree population, callbacks, Listing Window output, and NX Face highlight implementation.
- Create `integrations/nx2212/BrepMFRInstanceBrowser.dlx`: checked-in Block Styler layout asset.
- Modify `integrations/nx2212/include/plugin_runtime.hpp`: add the browser host boundary while retaining degraded-success text display.
- Modify `integrations/nx2212/include/nx_plugin_host.hpp`: implement the new host boundary.
- Modify `integrations/nx2212/src/plugin_runtime.cpp`: route fully colored recognition to the browser.
- Modify `integrations/nx2212/src/nx_plugin_host.cpp`: launch the NX browser and keep old `ShowSuccess` for degraded success.
- Modify `integrations/nx2212/tests/plugin_runtime_test.cpp`: assert success routing and browser payload.
- Modify both NX `.vcxproj` files: compile pure browser code/tests, compile the NX adapter only into the DLL, and copy the `.dlx` beside the DLL.
- Modify `tests/test_nx2212_project.py`: verify project wiring, `.dlx` contract, and post-build asset.
- Modify `integrations/nx2212/README.md`: document the browser, pagination, temporary highlight, and fallback behavior.

---

### Task 1: Pure browser model and pagination state

**Files:**
- Create: `integrations/nx2212/include/instance_browser.hpp`
- Create: `integrations/nx2212/src/instance_browser.cpp`
- Create: `integrations/nx2212/tests/instance_browser_test.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`

**Interfaces:**
- Consumes: `SuccessResult`, `MappingDisplay`, `DisplayText`, `InstanceColorRecord`, `FaceKey`.
- Produces: `kInstanceBrowserPageSize`, `InstanceBrowserSummary`, `InstanceBrowserRow`, `InstanceBrowserModel`, `BuildInstanceBrowserModel(...)`, and `InstanceBrowserState`.

- [ ] **Step 1: Add failing model-construction tests**

Add test helpers that construct two `InstanceResult` values in reverse ID order and two matching `InstanceColorRecord` values. Add these exact behavioral tests:

```cpp
BREPMFR_TEST(InstanceBrowserModelJoinsAndSortsRecognitionRows)
{
    SuccessResult recognition;
    recognition.face_count = 5;
    recognition.cache_hit = true;
    recognition.total_seconds = 3.4;
    recognition.instances = {
        {7, 3, "rectangular_pocket", {3, 4}, 0.875},
        {2, 12, "blind_hole", {1}, 0.95}};

    MappingDisplay mapping;
    mapping.status = MappingDisplayStatus::Colored;
    mapping.mapped_faces = 5;
    mapping.colored_instances = 2;
    mapping.elapsed = std::chrono::milliseconds(300);
    mapping.instances = {
        {7, 2, {507, 508}},
        {2, 1, {501}}};

    DisplayText complete_text{"summary", {"line one", "line two"}};
    const auto model = BuildInstanceBrowserModel(
        recognition,
        std::chrono::milliseconds(9100),
        mapping,
        complete_text);

    ExpectEqual(model.rows.size(), std::size_t{2}, "row count mismatch");
    ExpectEqual(model.rows[0].instance_id, 2, "rows were not ID sorted");
    ExpectEqual(model.rows[0].nx_face_tags, std::vector<FaceKey>{501},
                "row tags mismatch");
    ExpectEqual(model.rows[1].instance_id, 7, "second row mismatch");
    ExpectEqual(model.summary.face_count, 5, "summary face count mismatch");
    ExpectEqual(model.summary.process_elapsed,
                std::chrono::milliseconds(9100), "process timing mismatch");
    ExpectEqual(model.complete_text.listing_lines_utf8.size(), std::size_t{2},
                "complete text was not retained");
}
```

Also add one table-driven test that rejects each of these inputs with a diagnostic containing the stated text:

| Invalid input | Required diagnostic fragment |
| --- | --- |
| `mapping.status != Colored` | `colored mapping` |
| duplicate recognition `instance_id` | `duplicate recognition instance` |
| missing color record | `missing color record` |
| extra color record | `unknown recognition instance` |
| empty `nx_face_tags` | `has no NX faces` |
| duplicate NX tag inside one row | `duplicate NX face tag` |

- [ ] **Step 2: Add failing pagination boundary tests**

Add a loop over the exact cases below and assert page count, pagination visibility, page bounds, and navigation enablement:

```cpp
struct PageCase
{
    std::size_t rows;
    std::size_t pages;
    bool visible;
};
const std::vector<PageCase> cases = {
    {0, 1, false}, {1, 1, false}, {50, 1, false},
    {51, 2, true}, {100, 2, true}, {101, 3, true}};

for (const auto& item : cases)
{
    InstanceBrowserState state(item.rows);
    ExpectEqual(state.PageCount(), item.pages, "page count mismatch");
    ExpectEqual(state.PaginationVisible(), item.visible,
                "pagination visibility mismatch");
    ExpectEqual(state.PageBegin(), std::size_t{0}, "first page begin mismatch");
    ExpectEqual(state.PageEnd(),
                (std::min)(item.rows, kInstanceBrowserPageSize),
                "first page end mismatch");
    Expect(!state.CanGoPrevious(), "previous was enabled on first page");
    ExpectEqual(state.CanGoNext(), item.pages > 1,
                "next enablement mismatch");
}
```

Add separate assertions that a 101-row state navigates through pages with bounds `[0,50)`, `[50,100)`, and `[100,101)`, refuses to move beyond either boundary, and clears its selected global row whenever a page actually changes.

- [ ] **Step 3: Build the native tests to verify they fail**

Run:

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
```

Expected: compilation fails because `instance_browser.hpp` and its API do not yet exist.

- [ ] **Step 4: Implement the pure model and state API**

Define the public API exactly as follows:

```cpp
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
};
```

`BuildInstanceBrowserModel` must use ID-keyed maps to join records, reject all inconsistent inputs before returning, sort rows by ID, and copy the structured summary directly from recognition and mapping fields. `InstanceBrowserState` must compute page count without division by zero and throw `std::out_of_range` for invalid visible/global indices.

- [ ] **Step 5: Run the focused native tests**

Run the MSBuild command from Step 3, then:

```powershell
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
```

Expected: all native tests pass and the summary reports zero failures.

- [ ] **Step 6: Commit the pure browser model**

```powershell
git add integrations/nx2212/include/instance_browser.hpp integrations/nx2212/src/instance_browser.cpp integrations/nx2212/tests/instance_browser_test.cpp integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj
git commit -m "feat: add NX instance browser model"
```

---

### Task 2: Selection session and temporary-highlight lifecycle

**Files:**
- Modify: `integrations/nx2212/include/instance_browser.hpp`
- Modify: `integrations/nx2212/src/instance_browser.cpp`
- Modify: `integrations/nx2212/tests/instance_browser_test.cpp`

**Interfaces:**
- Consumes: `InstanceBrowserModel`, `InstanceBrowserState`, and row `nx_face_tags` from Task 1.
- Produces: `TemporaryHighlightSink` and `InstanceBrowserSession` for the Block Styler adapter.

- [ ] **Step 1: Write failing session tests with a recording sink**

Add this test double:

```cpp
class RecordingHighlightSink final : public TemporaryHighlightSink
{
public:
    void Replace(const std::vector<FaceKey>& tags) override
    {
        operations.push_back("replace");
        replacements.push_back(tags);
        if (replace_error)
        {
            throw std::runtime_error(*replace_error);
        }
    }

    void Clear() noexcept override
    {
        operations.push_back("clear");
        ++clear_calls;
    }

    std::vector<std::string> operations;
    std::vector<std::vector<FaceKey>> replacements;
    std::optional<std::string> replace_error;
    int clear_calls = 0;
};
```

Test these exact behaviors:

- selecting visible row `0` calls `Replace` with that row's tags and stores its global index;
- selecting the already selected row performs no second `Replace`;
- selecting another row calls `Replace` once with only the new tags;
- `GoNext()` and `GoPrevious()` call `Clear()` only when the page changes and clear selection;
- `Close()` can be called twice but calls `Clear()` only once;
- the destructor calls `Close()` if it was not called explicitly;
- a throwing `Replace()` leaves `SelectedGlobalIndex()` empty and preserves the exception text.

- [ ] **Step 2: Run the focused test executable to verify failure**

Run:

```powershell
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
```

Expected: the new tests fail to compile because the session interfaces do not exist.

- [ ] **Step 3: Implement the session API**

Add exactly these interfaces:

```cpp
class TemporaryHighlightSink
{
public:
    virtual ~TemporaryHighlightSink() = default;
    virtual void Replace(const std::vector<FaceKey>& face_tags) = 0;
    virtual void Clear() noexcept = 0;
};

class InstanceBrowserSession
{
public:
    InstanceBrowserSession(
        const InstanceBrowserModel& model,
        TemporaryHighlightSink& highlight_sink);
    ~InstanceBrowserSession();
    InstanceBrowserSession(const InstanceBrowserSession&) = delete;
    InstanceBrowserSession& operator=(const InstanceBrowserSession&) = delete;

    const InstanceBrowserState& State() const noexcept;
    const InstanceBrowserRow* SelectedRow() const noexcept;
    const InstanceBrowserRow& SelectVisibleRow(std::size_t visible_row);
    bool GoPrevious();
    bool GoNext();
    void ClearSelection() noexcept;
    void Close() noexcept;
};
```

Selection must resolve the row first, return immediately when the same row is selected, call `Replace()` before committing the new selected index, and clear the state if `Replace()` throws. Page navigation must clear the sink and state only after confirming that navigation is possible. `Close()` must use an internal `closed_` flag to be idempotent.

- [ ] **Step 4: Run all native runtime tests**

Rebuild and execute:

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
```

Expected: build succeeds and all native tests pass.

- [ ] **Step 5: Commit session behavior**

```powershell
git add integrations/nx2212/include/instance_browser.hpp integrations/nx2212/src/instance_browser.cpp integrations/nx2212/tests/instance_browser_test.cpp
git commit -m "feat: manage NX instance browser selection"
```

---

### Task 3: Route fully colored recognition into the browser host boundary

**Files:**
- Modify: `integrations/nx2212/include/plugin_runtime.hpp`
- Modify: `integrations/nx2212/src/plugin_runtime.cpp`
- Modify: `integrations/nx2212/include/nx_plugin_host.hpp`
- Modify: `integrations/nx2212/src/nx_plugin_host.cpp`
- Modify: `integrations/nx2212/tests/plugin_runtime_test.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`

**Interfaces:**
- Consumes: `BuildInstanceBrowserModel(...)` and `InstanceBrowserModel` from Task 1.
- Produces: `PluginHost::ShowInstanceBrowser(const PartSnapshot&, const InstanceBrowserModel&)` and deterministic routing between the browser and existing `ShowSuccess`.

- [ ] **Step 1: Change the recording host and add failing routing assertions**

Extend `RecordingHost` with:

```cpp
void ShowInstanceBrowser(
    const PartSnapshot& snapshot,
    const InstanceBrowserModel& model) override
{
    ++browser_calls;
    browser_snapshot = snapshot;
    browser_model = model;
}

int browser_calls = 0;
PartSnapshot browser_snapshot;
InstanceBrowserModel browser_model;
```

Update `InferenceWorkflowMapsAndColorsReadyJson11Once` to require:

```cpp
ExpectEqual(host.browser_calls, 1, "colored success did not open browser");
ExpectEqual(host.success_calls, 0, "colored success opened legacy popup");
ExpectEqual(host.browser_snapshot.body_tag, host.part.body_tag,
            "browser snapshot mismatch");
ExpectEqual(host.browser_model.rows.size(), std::size_t{1},
            "browser row count mismatch");
ExpectEqual(host.browser_model.rows[0].nx_face_tags,
            std::vector<FaceKey>{501}, "browser face tags mismatch");
```

Keep the unavailable/invalid/matcher/color-plan failure tests expecting `success_calls == 1`, and add `browser_calls == 0` to each. The JSON 1.0 success test must also expect `browser_calls == 0`.

- [ ] **Step 2: Run native tests and verify compilation fails**

Run the Task 2 rebuild command.

Expected: compilation fails because `PluginHost::ShowInstanceBrowser` is missing.

- [ ] **Step 3: Add the host method and route success**

Add this pure virtual method while retaining `ShowSuccess`:

```cpp
virtual void ShowInstanceBrowser(
    const PartSnapshot& part,
    const InstanceBrowserModel& model) = 0;
```

In `RunInferenceWorkflow`, compute `DisplayText display` once. Route as follows:

```cpp
const auto display = FormatSuccessDisplay(
    success, output->result_path, process_execution->elapsed, mapping);
if (mapping.status == MappingDisplayStatus::Colored)
{
    host.ShowInstanceBrowser(
        part,
        BuildInstanceBrowserModel(
            success,
            process_execution->elapsed,
            mapping,
            display));
}
else
{
    host.ShowSuccess(display);
}
```

Task 1 already added `instance_browser.cpp/.hpp` and `instance_browser_test.cpp` to the runtime-test project. Add `instance_browser.cpp/.hpp` to the DLL project. To keep this intermediate commit buildable before the Block Styler adapter exists, add the new override to `NxPluginHost` and temporarily implement it as:

```cpp
void NxPluginHost::ShowInstanceBrowser(
    const PartSnapshot&,
    const InstanceBrowserModel& model)
{
    ShowSuccess(model.complete_text);
}
```

Task 5 replaces this transitional legacy display with the real modal browser.

- [ ] **Step 4: Rebuild and run all native tests**

Rebuild the full solution and execute the runtime tests:

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
```

Expected: all native tests pass; colored success records one browser call, and degraded success records one legacy display call.

- [ ] **Step 5: Commit runtime routing**

```powershell
git add integrations/nx2212/include/plugin_runtime.hpp integrations/nx2212/src/plugin_runtime.cpp integrations/nx2212/include/nx_plugin_host.hpp integrations/nx2212/src/nx_plugin_host.cpp integrations/nx2212/tests/plugin_runtime_test.cpp integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj integrations/nx2212/BrepMFRNx2212.vcxproj
git commit -m "feat: route NX results to instance browser"
```

---

### Task 4: NX Face temporary highlighter

**Files:**
- Create: `integrations/nx2212/include/nx_instance_browser.hpp`
- Create: `integrations/nx2212/src/nx_instance_browser.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`

**Interfaces:**
- Consumes: `TemporaryHighlightSink`, `PartSnapshot`, `FaceKey`, and `ValidateNxCurrentPart(...)`.
- Produces: NX-only `NxTemporaryFaceHighlighter`, used privately by the dialog; public adapter declaration `ShowNxInstanceBrowser(...)` is completed in Task 5.

- [ ] **Step 1: Declare the NX adapter and scoped highlighter**

Declare the external entry point now so project compilation checks the boundary:

```cpp
void ShowNxInstanceBrowser(
    const std::filesystem::path& dlx_path,
    const PartSnapshot& part,
    const InstanceBrowserModel& model);
```

Inside `nx_instance_browser.cpp`, define a final private implementation of `TemporaryHighlightSink` that owns the expected `PartSnapshot` and the currently highlighted `NXOpen::Face*` list. The modal dialog prevents modeling operations while these pointers are retained, and keeping the resolved objects allows close cleanup even if the current display focus changes.

- [ ] **Step 2: Implement fail-closed face resolution and replacement**

`Replace()` must perform these concrete operations in order:

```cpp
Clear();
ValidateNxCurrentPart(expected_);
const auto faces_by_tag = CurrentBodyFacesByTag(expected_);
std::vector<NXOpen::Face*> targets;
for (const FaceKey tag : face_tags)
{
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
```

`CurrentBodyFacesByTag` must re-enumerate the unique current solid Body, verify `part_tag`, `body_tag`, and `face_count`, reject null/duplicate faces, and return an ID map. `Clear() noexcept` must call `Unhighlight()` on every recorded `NXOpen::Face*`, swallow individual NX exceptions, and always empty `highlighted_`.

- [ ] **Step 3: Add the NX source/header to the DLL project and build**

Run:

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
```

Expected: DLL builds without introducing NXOpen sources into `BrepMFRNx2212RuntimeTests`.

- [ ] **Step 4: Commit the scoped highlighter**

```powershell
git add integrations/nx2212/include/nx_instance_browser.hpp integrations/nx2212/src/nx_instance_browser.cpp integrations/nx2212/BrepMFRNx2212.vcxproj
git commit -m "feat: add scoped NX face highlighting"
```

---

### Task 5: Block Styler dialog, Tree interaction, and `.dlx` asset

**Files:**
- Create: `integrations/nx2212/BrepMFRInstanceBrowser.dlx`
- Modify: `integrations/nx2212/src/nx_instance_browser.cpp`
- Modify: `integrations/nx2212/src/nx_plugin_host.cpp`
- Modify: `integrations/nx2212/include/nx_plugin_host.hpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**
- Consumes: `InstanceBrowserSession`, `NxTemporaryFaceHighlighter`, `DisplayText`, and `ModuleDirectory()`.
- Produces: the complete `ShowNxInstanceBrowser(...)` modal adapter and `NxPluginHost::ShowInstanceBrowser(...)`.

- [ ] **Step 1: Add failing repository tests for the UI asset and build wiring**

Extend `tests/test_nx2212_project.py` with constants for the source/output `.dlx` paths. Add this contract test:

```python
def test_nx2212_instance_browser_asset_has_required_blocks():
    dlx = NX_ROOT / "BrepMFRInstanceBrowser.dlx"
    assert dlx.is_file()
    root = ET.parse(str(dlx)).getroot()
    block_ids = {
        node.attrib.get("value")
        for node in root.findall(".//Property")
        if node.attrib.get("sname") == "BlockID"
    }
    assert {
        "summary",
        "instance_tree",
        "paging_group",
        "previous_page",
        "page_label",
        "next_page",
        "details",
        "show_full_result",
    } <= block_ids
```

Extend the build test to require `build/Release/BrepMFRInstanceBrowser.dlx` and byte-for-byte equality with the source asset. Extend the portable-contract test to require `src\nx_instance_browser.cpp`, `include\nx_instance_browser.hpp`, and the `.dlx` copy command.

- [ ] **Step 2: Run the focused pytest and verify failure**

Run:

```powershell
python -m pytest tests/test_nx2212_project.py -q
```

Expected: the new asset test fails because the `.dlx` does not exist.

- [ ] **Step 3: Create the Block Styler layout with fixed IDs**

Create the `.dlx` as NX Block Styler XML with this exact control contract:

| Block ID | Block type | Required properties |
| --- | --- | --- |
| `summary` | Multiline String | read-only, height 100, stretch width |
| `instance_tree` | Tree List | single selection, headers/multiple columns on, height 280, stretch both |
| `paging_group` | Group | one horizontal row, initially hidden |
| `previous_page` | Button | label `上一页` |
| `page_label` | Label | initial label `第 1 / 1 页` |
| `next_page` | Button | label `下一页` |
| `details` | Multiline String | read-only, initial value `请选择一个特征实例`, height 100 |
| `show_full_result` | Button | label `查看完整结果` |

Set the dialog title to `BrepMFR 特征实例浏览器`, allow resize, hide Apply, keep the standard OK navigation button as the close action, and enable initialize/update/dialog-shown/ok/close callbacks.

- [ ] **Step 4: Implement dialog creation and callbacks**

Create an internal `NxInstanceBrowserDialog` whose constructor obtains `NXOpen::UI::GetUI()`, creates the dialog from the absolute UTF-8 `.dlx` path, and registers:

```cpp
dialog_->AddInitializeHandler(
    make_callback(this, &NxInstanceBrowserDialog::Initialize));
dialog_->AddDialogShownHandler(
    make_callback(this, &NxInstanceBrowserDialog::DialogShown));
dialog_->AddUpdateHandler(
    make_callback(this, &NxInstanceBrowserDialog::Update));
dialog_->AddOkHandler(
    make_callback(this, &NxInstanceBrowserDialog::Ok));
dialog_->AddCloseHandler(
    make_callback(this, &NxInstanceBrowserDialog::Close));
```

`Initialize()` must find and type-check all eight block IDs, register `instance_tree->SetOnSelectHandler(...)`, and throw a diagnostic naming any missing block. `DialogShown()` must set `summary` to these structured lines before rendering the Tree:

```text
面数量：<face_count>    特征实例：<instance_count>    BIN 缓存：<命中|未命中>
流水线用时：<pipeline_seconds, one decimal> 秒    Python 进程总用时：<process_elapsed, one decimal> 秒
已映射面：<mapped_faces>/<face_count>    已着色实例：<colored_instances>    映射用时：<mapping_elapsed, one decimal> 秒
```

Then insert four non-sortable text columns with IDs `0..3`, titles `实例`, `特征类型`, `面数`, `置信度`, widths `70, 180, 70, 100`, and call `RenderPage()`. Set multiline values through a scoped `PropertyList` and `SetString("Value", NXString(utf8, NXString::UTF8))`; never parse `DisplayText::summary_utf8`.

`RenderPage()` must freeze Tree redraw, delete all existing root nodes, insert rows for `[PageBegin(), PageEnd())`, store `NXOpen::BlockStyler::Node* -> visible row index` in a map, restore redraw in an exception-safe guard, update page label/buttons, hide `paging_group` when `PaginationVisible()` is false, and reset details to `请选择一个特征实例`.

`OnSelect(..., selected=true)` must look up the node map, call `session_.SelectVisibleRow(...)`, and set details to:

```text
实例：<id>
特征类型：<feature_name>
置信度：<four decimals>
OCC/AAG 面：[comma-separated indices]
NX Face Tags：[comma-separated tags]
```

If selection/highlight throws, call `session_.ClearSelection()`, reset details, disable the Tree and page buttons for the remainder of the dialog, and show an NX error message containing the instance ID and original exception. This leaves no temporary highlight and prevents callbacks from continuing against a failed session.

`Update()` must route `previous_page`, `next_page`, and `show_full_result`. Page buttons call the corresponding session method then `RenderPage()`. The full-result button writes `complete_text.listing_lines_utf8` once, opens the Listing Window, and only reopens it on later clicks without appending duplicate lines; a Listing Window exception is shown as an NX error but does not disable the browser. `Ok()`, `Close()`, the dialog destructor, and exceptions leaving `Launch()` must all call `session_.Close()`.

- [ ] **Step 5: Wire the host and post-build copy**

Implement:

```cpp
void NxPluginHost::ShowInstanceBrowser(
    const PartSnapshot& part,
    const InstanceBrowserModel& model)
{
    ShowNxInstanceBrowser(
        ModuleDirectory() / "BrepMFRInstanceBrowser.dlx",
        part,
        model);
}
```

Add the `.dlx` as a `<None Include=...>` item and extend the post-build command with:

```bat
copy /Y "$(ProjectDir)BrepMFRInstanceBrowser.dlx" "$(TargetDir)BrepMFRInstanceBrowser.dlx"
```

- [ ] **Step 6: Run focused repository tests and native builds**

Run:

```powershell
python -m pytest tests/test_nx2212_project.py -q
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
```

Expected: pytest passes, solution build succeeds, native runtime tests pass, and the Release directory contains both DLL and `.dlx`.

- [ ] **Step 7: Commit the NX browser adapter**

```powershell
git add integrations/nx2212/BrepMFRInstanceBrowser.dlx integrations/nx2212/include/nx_plugin_host.hpp integrations/nx2212/src/nx_plugin_host.cpp integrations/nx2212/src/nx_instance_browser.cpp integrations/nx2212/BrepMFRNx2212.vcxproj tests/test_nx2212_project.py
git commit -m "feat: add NX Block Styler instance browser"
```

---

### Task 6: Operator documentation and complete automated verification

**Files:**
- Modify: `integrations/nx2212/README.md`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**
- Consumes: the completed browser behavior and deployment artifact from Tasks 1-5.
- Produces: an operator-facing usage contract and final automated evidence.

- [ ] **Step 1: Add a failing operator-guide contract**

Extend `test_nx2212_operator_guide_describes_current_part_face_mapping` to require all of these phrases:

```python
for required in (
    "特征实例浏览器",
    "50 个实例一页",
    "临时高亮",
    "查看完整结果",
    "关闭浏览器",
):
    assert required in guide
```

Also assert that the old wording `成功后，信息框显示` is absent.

- [ ] **Step 2: Run the focused guide test and verify failure**

Run:

```powershell
python -m pytest tests/test_nx2212_project.py::test_nx2212_operator_guide_describes_current_part_face_mapping -q
```

Expected: failure because the guide still documents the success message box.

- [ ] **Step 3: Update the operator guide**

Replace the successful-result instructions with the exact workflow: browser opens directly after complete mapping/coloring; summary stays at the top; rows are single-select; selected faces receive temporary NX highlight while all base colors remain; 50 rows per page; `查看完整结果` opens Listing Window once; closing clears temporary highlight but leaves base colors; mapping/coloring failures retain the legacy text result path. Add `BrepMFRInstanceBrowser.dlx` to the documented Release artifacts and troubleshooting section.

- [ ] **Step 4: Run complete verification**

Run:

```powershell
python -m pytest -q
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git diff --check
git status --short
```

Expected: full pytest suite passes, Release/x64 solution rebuild succeeds, native runtime test summary has zero failures, `git diff --check` emits nothing, and status lists only the intended README/test changes before commit.

- [ ] **Step 5: Commit documentation**

```powershell
git add integrations/nx2212/README.md tests/test_nx2212_project.py
git commit -m "docs: explain NX instance browser workflow"
```

---

### Task 7: NX2212 field acceptance and merge readiness

**Files:**
- No source changes expected; if field testing finds a defect, return to the smallest responsible task and repeat its red/green cycle before continuing.

**Interfaces:**
- Consumes: `build/Release/BrepMFRNx2212.dll`, `BrepMFRInstanceBrowser.dlx`, current local config, and the verified 41-face/9-instance STEP.
- Produces: recorded field evidence that the UI and highlight lifecycle work inside NX2212.

- [ ] **Step 1: Verify deployable files**

Run:

```powershell
Get-Item integrations\nx2212\build\Release\BrepMFRNx2212.dll
Get-Item integrations\nx2212\build\Release\BrepMFRInstanceBrowser.dlx
Get-Item integrations\nx2212\build\Release\BrepMFRNx2212.config.json
```

Expected: all three files exist in the same directory; the real config remains ignored by Git.

- [ ] **Step 2: Perform the 41-face/9-instance NX field test**

In NX2212, open the existing golden STEP without modification and execute the rebuilt DLL. Verify all ten acceptance checks from spec section 11.3: direct browser opening, correct top summary, hidden pagination for 9 rows, correct per-row temporary highlight, old highlight removal, no base-color changes, non-duplicating full text, and complete highlight cleanup on close.

- [ ] **Step 3: Exercise pagination with 51 rows through the tested model boundary**

Use the native 51-row fixture from `instance_browser_test.cpp` to confirm two-page state automatically. If an NX field fixture with more than 50 recognized instances is available, open it and additionally confirm the visible page controls and highlight cleanup on page change; absence of such a CAD fixture does not invalidate the automated boundary test.

- [ ] **Step 4: Record final repository state**

Run:

```powershell
git status --short --branch
git log --oneline -7
```

Expected: the feature worktree is clean and the task commits are visible in order. Do not merge or delete the worktree until the user approves the field result.
