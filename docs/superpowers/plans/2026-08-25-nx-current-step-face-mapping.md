# NX2212 Current STEP Face Mapping and Instance Coloring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户在 NX2212 打开一个未修改的单实体 STEP 后直接点击插件，完成 Python 识别、OCC/AAG 面到 NX Face 的可靠全局映射，并按特征实例原子着色。

**Architecture:** Python 继续作为独立进程，负责生成 BIN、OCC 几何签名 sidecar 和自包含 JSON 1.1；NX C++ 负责当前 Work Part 校验、NX 几何签名、纯 C++ 全局一对一匹配、颜色计划和显示。映射采用离散候选门控、连续代价最小分配、邻接图验证和歧义拒绝，任何不完整结果都只显示文字而不着色。

**Tech Stack:** Python 3.8、pythonOCC/occwl、DGL、PyTorch、pytest、C++17、MSVC v142/x64、Siemens NX2212 NXOpen/UFUN、nlohmann/json 3.12.0。

**Spec:** `docs/superpowers/specs/2026-08-25-nx-current-step-face-mapping-design.md`

## Global Constraints

- 只支持 NX2212、`Release|x64`、`PlatformToolset=v142`、`LanguageStandard=stdcpp17`。
- 正常入口只读取当前 Work Part；不弹 STEP 文件选择器，不自动导出 STEP。
- 首版只接受一个未修改的、恰好包含一个非收敛实体 Body 的 `.step`/`.stp` Work Part。
- Python、PyTorch、CUDA 和 checkpoint 始终留在独立 Python 进程，不进入 NX 地址空间。
- JSON 1.0 必须继续显示原文字结果但禁止着色；只有 JSON 1.1 映射扩展有效时才允许着色。
- OCC/AAG `face_index` 永远不能按数组位置转换为 NX Face Tag。
- Python 与 NX 都使用包围盒中心和平移后乘 `2 / longest_bbox_extent` 的 `aag_unit_box` 规范化。
- 面映射必须完整、一对一、误差合格且邻接图一致；无解或歧义时颜色计划必须为空。
- 同一实例所有面同色，不同实例使用互异 NX 颜色；stock 面不修改。
- 着色必须原子应用，失败后恢复本次修改前的颜色；插件不调用保存。
- 不新增在线运行依赖，不修改 vendored nlohmann/json。
- 每项生产行为采用严格 TDD：先观察预期失败，再写最小实现，再运行相关回归。

## File Structure Map

- `brepmfr/inference/face_signature.py`：OCC/AAG 面签名、邻接图、sidecar 校验与序列化契约。
- `brepmfr/inference/conversion.py`：一次提取中生成并成对发布 BIN 与签名 sidecar。
- `brepmfr/inference/cache.py`：缓存版本 3，同时验证 BIN 和签名文件的身份及哈希。
- `brepmfr/inference/result.py`：生成 JSON 1.1，并保留 1.0 核心字段语义。
- `brepmfr/inference/pipeline.py`：缓存命中或转换后把映射扩展写入最终结果。
- `integrations/nx2212/include|src/face_signature.*`：纯 C++ 几何签名、规范化和图校验。
- `integrations/nx2212/include|src/assignment_solver.*`：方阵最小代价一对一分配。
- `integrations/nx2212/include|src/face_matcher.*`：候选门控、代价、拓扑和歧义验证。
- `integrations/nx2212/include|src/color_plan.*`：实例到 NX 颜色及 Face Tag 的纯计划。
- `integrations/nx2212/include|src/nx_part_adapter.*`：当前 Part、Body、Face 和 NX 几何签名适配。
- `integrations/nx2212/include|src/nx_coloring.*`：NX 显示修改、颜色保存和失败回滚。
- `integrations/nx2212/plugin_runtime.*`：从当前 Part 到推理、映射、着色、文字显示的编排。
- `integrations/nx2212/result_reader.*`：分层解析 JSON 1.0/1.1；映射扩展错误不破坏核心结果。
- `integrations/nx2212/result_formatter.*`：旧文字结果、映射诊断、NX Tags、颜色和计时。

---

### Task 1: Python OCC/AAG Face Signature Contract

**Files:**
- Create: `brepmfr/inference/face_signature.py`
- Create: `tests/test_face_signature.py`

**Interfaces:**

```python
FACE_MAPPING_CONTRACT_VERSION = "1.0"
NORMALIZATION_KIND = "aag_unit_box"

def extract_face_mapping(extractor: object, extraction: Mapping[str, object]) -> Dict[str, object]: ...
def validate_face_mapping_document(
    document: Mapping[str, object], expected_face_count: Optional[int] = None
) -> Dict[str, object]: ...
def read_face_mapping(path: Path, expected_face_count: Optional[int] = None) -> Dict[str, object]: ...
def face_mapping_json_bytes(document: Mapping[str, object]) -> bytes: ...
```

- `extract_face_mapping` 只使用 `extractor._graph` 中已经缩放的 occwl Face 和 `extraction["graph_face_grid"]`。
- ready 文档固定包含 `contract_version`、`status`、`normalization`、`faces` 和 `adjacency_edges`。
- 每个面固定包含 `face_index`、`surface_type`、`area`、`perimeter`、`centroid`、`bounding_box`、`loop_count`、`edge_count`、`neighbor_count`、`surface_parameters` 和非空 `sample_points`。

- [ ] **Step 1: 写黄金 STEP 的失败测试**

```python
def test_golden_step_face_mapping_is_complete_and_aligned():
    extractor = AAGExtractor(COMMITTED_STEP, load_schema(DEFAULT_SCHEMA_FILE))
    extraction = extractor.process()

    mapping = extract_face_mapping(extractor, extraction)

    assert mapping["contract_version"] == "1.0"
    assert mapping["status"] == "ready"
    assert [face["face_index"] for face in mapping["faces"]] == list(range(41))
    assert all(face["sample_points"] for face in mapping["faces"])
    assert mapping["adjacency_edges"] == sorted(mapping["adjacency_edges"])
    assert all(left < right for left, right in mapping["adjacency_edges"])
```

测试还要递归检查全部浮点值有限、所有邻接端点在 `[0, 40]` 内、无重复边，并确认每个
`neighbor_count` 等于邻接图中的度。

同一步写入平移和整体缩放不变性测试：用 pythonOCC 读取黄金 STEP，通过
`gp_Trsf.SetTranslation` 和 `gp_Trsf.SetScale`、`BRepBuilderAPI_Transform`、
`STEPControl_Writer` 分别写出平移版和 25.4 倍整体缩放版。对三个 STEP 提取签名，比较面型、
拓扑、规范化面积/周长/质心/包围盒和解析参数，浮点容差使用 `1e-6`：

```python
np.testing.assert_allclose(
    transformed_face["centroid"], original_face["centroid"], atol=1e-6, rtol=0.0
)
assert transformed["adjacency_edges"] == original["adjacency_edges"]
```

- [ ] **Step 2: 运行测试确认 RED**

```powershell
python -m pytest tests/test_face_signature.py -v
```

Expected: FAIL，因为 `brepmfr.inference.face_signature` 尚不存在。

- [ ] **Step 3: 实现最小 OCC 签名提取**

使用 `_graph.nodes[face_index]["face"]` 取得与 AAG 相同的 Face。面积和质心使用
`brepgprop_SurfaceProperties`，周长对该面的唯一 TopoDS Edge 使用
`brepgprop_LinearProperties` 求和；环数来自 TopologyExplorer 的 wires；面型和解析参数来自
`BRepAdaptor_Surface`：

```python
surface = BRepAdaptor_Surface(topods_face)
if surface.GetType() == GeomAbs_Plane:
    plane = surface.Plane()
    axis = _canonical_axis(plane.Axis().Direction())
    surface_type = "plane"
    parameters = {
        "origin": _closest_point_on_plane_to_origin(_point(plane.Location()), axis),
        "axis": axis,
    }
elif surface.GetType() == GeomAbs_Cylinder:
    cylinder = surface.Cylinder()
    axis = _canonical_axis(cylinder.Axis().Direction())
    surface_type = "cylinder"
    parameters = {
        "origin": _closest_point_on_axis_to_origin(_point(cylinder.Axis().Location()), axis),
        "axis": axis,
        "radius": float(cylinder.Radius()),
    }
```

平面和圆柱的 `origin` 分别规范为平面/轴线上距离全局原点最近的点，避免两个内核返回不同但
等价的参数原点。圆锥保存 `origin`（apex）、axis、half_angle；球面保存 `origin`（center）、
radius；圆环面保存 `origin`（center）、axis、`radius`（major）和 `secondary_radius`（minor）；
其他面归一为 `bspline`、`revolution` 或 `other`，不伪造解析参数。
位置、长度、面积和解析参数必须直接取自 `extractor._graph` 的已缩放 shape，因此已经处于
AAG 坐标系；禁止再次应用 center/scale。样本从 `graph_face_grid[face_index]` 的 XYZ 通道中
选取 mask 通道为真的点，转成无序坐标数组。邻接边使用：

```python
adjacency = sorted(
    {tuple(sorted((int(left), int(right)))) for left, right in graph.edges if left != right}
)
```

- [ ] **Step 4: 实现严格校验、确认 GREEN 并提交**

`validate_face_mapping_document` 返回 JSON round-trip 后的新字典；拒绝非 ready/unavailable 状态、
错误契约、非连续编号、非有限数值、空样本、错误数组长度、重复邻接、自环和度数不一致。
`face_mapping_json_bytes` 使用 `sort_keys=True`、UTF-8、缩进 2 和结尾换行。

```powershell
python -m pytest tests/test_face_signature.py -v
git add brepmfr/inference/face_signature.py tests/test_face_signature.py
git commit -m "feat: extract OCC face mapping signatures"
```

Expected: 黄金、平移、缩放和校验测试全部 PASS，提交只包含本任务两个文件。

---

### Task 2: Paired BIN and Signature Cache Publication

**Files:**
- Modify: `brepmfr/inference/conversion.py`
- Modify: `brepmfr/inference/cache.py`
- Modify: `brepmfr/inference/pipeline.py`
- Modify: `tests/test_inference_conversion.py`
- Modify: `tests/test_inference_cache.py`
- Modify: `tests/test_inference_pipeline.py`

**Interfaces:**

```python
INFERENCE_BIN_CONTRACT_VERSION = 3

@dataclass(frozen=True)
class ConversionArtifacts:
    face_count: int
    face_mapping: Dict[str, object]

def convert_step_to_inference_artifacts(
    step_path: Path,
    bin_path: Path,
    signature_path: Path,
    requirements: GeometryRequirements,
    seed: int,
    extractor_file: Path = DEFAULT_EXTRACTOR_FILE,
    schema_file: Path = DEFAULT_SCHEMA_FILE,
) -> ConversionArtifacts: ...

@dataclass(frozen=True)
class CacheDecision:
    hit: bool
    fingerprint: str
    bin_path: Path
    signature_path: Path
    metadata_path: Path
    reason: str

def inspect_cache(
    bin_path: Path,
    signature_path: Path,
    metadata_path: Path,
    identity: Mapping[str, object],
    validate_bin_file: Callable[[Path], None],
    validate_signature_file: Callable[[Path], None],
    force_rebuild: bool = False,
) -> CacheDecision: ...

def write_cache_metadata_atomic(
    path: Path,
    identity: Mapping[str, object],
    fingerprint: str,
    bin_path: Path,
    signature_path: Path,
) -> None: ...
```

- [ ] **Step 1: 写缓存双文件失败测试**

扩展 `_cache_inputs` 创建 `input.face_signatures.json`。有效元数据必须精确包含：

```python
{
    "schema_version": 3,
    "fingerprint": fingerprint,
    "identity": identity,
    "bin_sha256": sha256_file(bin_path),
    "face_signatures_sha256": sha256_file(signature_path),
}
```

表驱动覆盖 `missing-signatures`、`invalid-signatures`、签名哈希不匹配、签名验证器抛错，且这些
情况都不能调用 BIN 后的加载流程。

- [ ] **Step 2: 运行缓存测试确认 RED**

```powershell
python -m pytest tests/test_inference_cache.py -v
```

Expected: FAIL，因为 `CacheDecision` 和元数据仍只有 BIN。

- [ ] **Step 3: 将缓存契约升级为版本 3**

把签名路径纳入 `CacheDecision`。`inspect_cache` 先验证两个文件和两个哈希，再依次调用两个
验证器；签名失败固定返回 `invalid-signatures`。元数据最后发布，代码结构为：

```python
document = {
    "schema_version": INFERENCE_BIN_CONTRACT_VERSION,
    "fingerprint": fingerprint,
    "identity": dict(identity),
    "bin_sha256": sha256_file(bin_path),
    "face_signatures_sha256": sha256_file(signature_path),
}
```

运行缓存测试，Expected: PASS。

- [ ] **Step 4: 写成对发布失败测试**

把现有转换测试改为调用 `convert_step_to_inference_artifacts`，断言正式 BIN 和 sidecar 在两个
临时文件都写完并通过校验前均不存在。注入第二次 `Path.replace` 失败，断言旧元数据不能让
下一次 `inspect_cache` 命中；`.tmp` 文件必须清理。

```python
artifacts = convert_step_to_inference_artifacts(
    step_path, bin_path, signature_path, requirements, seed=42
)
assert artifacts.face_count == 2
assert read_face_mapping(signature_path, expected_face_count=2) == artifacts.face_mapping
```

- [ ] **Step 5: 运行转换测试确认 RED**

```powershell
python -m pytest tests/test_inference_conversion.py -v
```

Expected: FAIL，因为成对发布接口尚不存在。

- [ ] **Step 6: 实现成对发布和实现指纹**

一次 `extractor.process()` 同时构建 DGL graph 和 `extract_face_mapping`。写入
`input.bin.tmp` 与 `input.face_signatures.json.tmp`，分别回读校验后再替换两个正式文件；缓存
元数据由 pipeline 在两者都成功后单独写入。把 `face_signature.py` 加入
`default_conversion_implementation_paths()`，使签名算法变化触发缓存失效。

```python
mapping = extract_face_mapping(extractor, extraction)
temporary_signature.write_bytes(face_mapping_json_bytes(mapping))
validate_face_mapping_document(
    json.loads(temporary_signature.read_text(encoding="utf-8")),
    expected_face_count=face_count,
)
temporary_bin.replace(bin_path)
temporary_signature.replace(signature_path)
return ConversionArtifacts(face_count=face_count, face_mapping=mapping)
```

同一步把 pipeline 改为固定构造 `input.face_signatures.json`，向 `inspect_cache` 传入签名验证器，
miss 时调用新转换函数并把 signature_path 传给元数据写入。此任务仍生成 JSON 1.0，不把映射
嵌入结果；Task 3 只负责升级最终结果契约。同步更新 pipeline 测试中的 `CacheDecision` 和 mock
签名，保证本任务提交后完整现有 pipeline 测试仍通过。

- [ ] **Step 7: 运行回归并提交**

```powershell
python -m pytest tests/test_inference_conversion.py tests/test_inference_cache.py tests/test_inference_pipeline.py tests/test_face_signature.py -q
git add brepmfr/inference/conversion.py brepmfr/inference/cache.py brepmfr/inference/pipeline.py tests/test_inference_conversion.py tests/test_inference_cache.py tests/test_inference_pipeline.py
git commit -m "feat: cache BIN and face signatures together"
```

Expected: 全部通过，缓存契约版本精确为 3。

---

### Task 3: Python Pipeline JSON 1.1 Integration

**Files:**
- Modify: `brepmfr/inference/result.py`
- Modify: `brepmfr/inference/pipeline.py`
- Modify: `tests/test_inference_result.py`
- Modify: `tests/test_inference_pipeline.py`

**Interfaces:**

```python
RESULT_SCHEMA_VERSION = "1.1"

def build_success_document(
    ...,
    face_mapping: Mapping[str, object],
) -> Dict[str, object]: ...
```

最终输出目录固定增加：

```text
input.bin
input.face_signatures.json
input.bin.meta.json
result.json
run.log
```

- [ ] **Step 1: 写 JSON 1.1 失败测试**

更新 `_success_document` 传入一个两面的 ready 映射，断言：

```python
assert RESULT_SCHEMA_VERSION == "1.1"
assert document["schema_version"] == "1.1"
assert document["face_mapping"]["contract_version"] == "1.0"
assert list(document)[-1] == "face_mapping"
```

另测 `status="unavailable"`、非有限签名数值和 `face_count` 不一致；后两者必须拒绝。

- [ ] **Step 2: 运行结果测试确认 RED**

```powershell
python -m pytest tests/test_inference_result.py -v
```

Expected: FAIL，因为结果版本仍为 1.0。

- [ ] **Step 3: 生成 JSON 1.1 并保持核心字段顺序**

`build_success_document` 先调用 `validate_face_mapping_document(face_mapping, face_count)`，再把深
拷贝后的扩展放在 `warnings` 之后。错误文档也使用 1.1，但保持 `status/stage/error/input` 语义。

- [ ] **Step 4: 写 pipeline 缓存 miss/hit 失败测试**

复用 Task 2 已验证的 signature_path 编排。miss 测试让转换返回 ready mapping，并断言最终结果
嵌入该映射；hit 测试模拟 `read_face_mapping`，断言不转换且结果嵌入 sidecar：

```python
assert outcome.document["schema_version"] == "1.1"
assert outcome.document["face_mapping"] == ready_mapping
assert outcome.document["cache"]["hit"] is True
```

进度顺序保持 checkpoint、cache、conversion（仅 miss）、loading、inference、postprocess、
serialization，不增加伪百分比。

- [ ] **Step 5: 运行 pipeline 测试确认 RED**

```powershell
python -m pytest tests/test_inference_pipeline.py -v
```

Expected: FAIL，因为 pipeline 尚未把 ready mapping 嵌入成功结果。

- [ ] **Step 6: 接入缓存、sidecar 和最终结果**

pipeline 固定构造：

```python
bin_path = output_dir / "input.bin"
signature_path = output_dir / "input.face_signatures.json"
metadata_path = output_dir / "input.bin.meta.json"
```

miss 使用 `convert_step_to_inference_artifacts` 的 `face_mapping`；hit 使用
`read_face_mapping(decision.signature_path, expected_face_count=face_count)`。在模型加载批次得到
`face_count` 后再次校验 sidecar 数量，随后传入 `build_success_document`。

- [ ] **Step 7: 运行 Python 回归并提交**

```powershell
python -m pytest tests/test_inference_result.py tests/test_inference_pipeline.py tests/test_inference_cache.py tests/test_inference_conversion.py tests/test_inference_golden.py -q
git add brepmfr/inference/result.py brepmfr/inference/pipeline.py tests/test_inference_result.py tests/test_inference_pipeline.py
git commit -m "feat: emit face mapping in inference JSON 1.1"
```

Expected: 相关测试通过；`tests/golden/20221121_154647_0.json` 是论文数值基准记录，不改其独立
`schema_version`。

---

### Task 4: Pure C++ Face Signature and Normalization Types

**Files:**
- Create: `integrations/nx2212/include/face_signature.hpp`
- Create: `integrations/nx2212/src/face_signature.cpp`
- Create: `integrations/nx2212/tests/face_signature_test.cpp`
- Modify: `integrations/nx2212/tests/test_support.hpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`

**Interfaces:**

```cpp
namespace brepmfr::nx2212 {
using FaceKey = std::uint64_t;
struct Vec3 { double x; double y; double z; };
struct Bounds3 { Vec3 minimum; Vec3 maximum; };
bool operator==(const Vec3& left, const Vec3& right) noexcept;
enum class SurfaceType { Plane, Cylinder, Cone, Sphere, Torus, Revolution, BSpline, Other };

struct SurfaceParameters {
    std::optional<Vec3> origin;
    std::optional<Vec3> axis;
    std::optional<double> radius;
    std::optional<double> secondary_radius;
    std::optional<double> half_angle;
};

struct FaceSignature {
    FaceKey key;
    SurfaceType surface_type;
    double area;
    double perimeter;
    Vec3 centroid;
    Bounds3 bounds;
    int loop_count;
    int edge_count;
    int neighbor_count;
    SurfaceParameters surface;
    std::vector<Vec3> sample_points;
};

using AdjacencyEdge = std::pair<FaceKey, FaceKey>;
struct FaceGraph {
    std::vector<FaceSignature> faces;
    std::vector<AdjacencyEdge> adjacency_edges;
};

struct UnitBoxTransform { Vec3 center; double scale; };
UnitBoxTransform MakeUnitBoxTransform(const Bounds3& body_bounds);
Vec3 NormalizePoint(const UnitBoxTransform& transform, const Vec3& point);
double NormalizeLength(const UnitBoxTransform& transform, double length);
double NormalizeArea(const UnitBoxTransform& transform, double area);
Vec3 CanonicalAxis(const Vec3& axis);
void ValidateFaceGraph(const FaceGraph& graph);
}

namespace brepmfr::nx2212::test {
void ExpectNear(double actual, double expected, double tolerance,
                const std::string& message);
void ExpectContains(std::string_view text, std::string_view needle,
                    const std::string& message);
}
```

- [ ] **Step 1: 写纯 C++ 规范化失败测试**

```cpp
BREPMFR_TEST(UnitBoxTransformUsesCenterAndLongestExtent)
{
    const Bounds3 bounds{{10.0, 20.0, 30.0}, {14.0, 22.0, 31.0}};
    const auto transform = MakeUnitBoxTransform(bounds);
    ExpectEqual(transform.center, Vec3{12.0, 21.0, 30.5}, "center mismatch");
    ExpectEqual(transform.scale, 0.5, "scale must be 2 / 4");
    ExpectEqual(NormalizePoint(transform, bounds.minimum),
                Vec3{-1.0, -0.5, -0.25}, "minimum mismatch");
}
```

再测零最长边、NaN、零轴、轴反向、重复 key、自环、重复邻接、未知端点、空样本和
`neighbor_count` 不一致均抛出 `std::invalid_argument`。

- [ ] **Step 2: 构建确认 RED**

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
```

Expected: 编译失败，因为 `face_signature.hpp` 尚不存在。

- [ ] **Step 3: 实现数学和图不变量**

核心公式必须精确为：

```cpp
const double longest = std::max({dx, dy, dz});
return UnitBoxTransform{
    {(min.x + max.x) / 2.0, (min.y + max.y) / 2.0, (min.z + max.z) / 2.0},
    2.0 / longest};
```

`CanonicalAxis` 先单位化，再令第一个绝对值大于 `1e-12` 的分量为正。图校验根据邻接边重新
计算每个 key 的度，并要求与签名一致。把 `ExpectNear` 实现为有限值检查加绝对误差断言，
把 `ExpectContains` 实现为 `text.find(needle) != npos`，供后续求解器、matcher 和 runtime
测试复用。

- [ ] **Step 4: 运行测试并提交**

```powershell
$nxPlanVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$nxPlanVsRoot = & $nxPlanVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$nxPlanMsbuild = Join-Path $nxPlanVsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $nxPlanMsbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git add integrations/nx2212/include/face_signature.hpp integrations/nx2212/src/face_signature.cpp integrations/nx2212/tests/face_signature_test.cpp integrations/nx2212/tests/test_support.hpp integrations/nx2212/*.vcxproj
git commit -m "feat: define native face signature contract"
```

Expected: 原生测试退出 0。

---

### Task 5: Layered C++ JSON 1.0 and 1.1 Reader

**Files:**
- Modify: `integrations/nx2212/include/result_reader.hpp`
- Modify: `integrations/nx2212/src/result_reader.cpp`
- Create: `integrations/nx2212/tests/fixtures/result_success_1_1.json`
- Modify: `integrations/nx2212/tests/result_reader_test.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj`

**Interfaces:**

```cpp
enum class MappingPayloadStatus { Unsupported, Ready, Unavailable, Invalid };

struct FaceMappingPayload {
    MappingPayloadStatus status = MappingPayloadStatus::Unsupported;
    std::string diagnostic;
    FaceGraph graph;
};

struct SuccessResult {
    std::string schema_version;
    std::filesystem::path step_path;
    int face_count = 0;
    bool cache_hit = false;
    double total_seconds = 0.0;
    std::vector<FaceResult> faces;
    std::vector<InstanceResult> instances;
    FaceMappingPayload mapping;
};
```

- [ ] **Step 1: 创建完整 1.1 fixture 和失败测试**

fixture 复用现有两面识别核心，并增加两个 ready 面签名与 `[[0, 1]]` 邻接。测试断言
`mapping.status == Ready`、key 为 0/1、解析参数和样本读入正确。

另把现有 1.0 fixture 断言为：

```cpp
Expect(success.mapping.status == MappingPayloadStatus::Unsupported,
       "JSON 1.0 unexpectedly enabled mapping");
```

- [ ] **Step 2: 构建确认 RED**

运行 Task 4 的原生构建命令。

Expected: 编译失败，因为 `SuccessResult` 尚无 mapping。

- [ ] **Step 3: 分层实现 1.1 ready/unavailable 解析**

只接受版本 `1.0` 和 `1.1`。先完成现有 `ReadSuccess` 核心校验，再单独解析映射：

```cpp
if (version == "1.0") {
    result.mapping.status = MappingPayloadStatus::Unsupported;
} else {
    try {
        result.mapping = ReadFaceMapping(RequireObjectField(document, "face_mapping", "root"));
        if (result.mapping.status == MappingPayloadStatus::Ready)
            ValidateFaceGraph(result.mapping.graph);
    } catch (const std::exception& error) {
        result.mapping.status = MappingPayloadStatus::Invalid;
        result.mapping.diagnostic = error.what();
        result.mapping.graph = {};
    }
}
```

`unavailable` 必须有非空 reason；ready 要求 normalization 三个字面值
`aag_unit_box/1.0/2_over_longest_bbox_extent`，并检查签名 key 与核心 `face_index` 集合一致。

- [ ] **Step 4: 写损坏扩展仍保留核心结果的失败测试**

逐项损坏映射：缺字段、NaN、重复 key、空样本、错误邻接、face_count 不一致。每个文档仍需
返回 `SuccessResult`、保留两面和一个实例，但 mapping 必须为 Invalid 且 graph 为空。未知
`schema_version="1.2"` 仍整体拒绝。

- [ ] **Step 5: 完成校验、运行并提交**

```powershell
$nxPlanVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$nxPlanVsRoot = & $nxPlanVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$nxPlanMsbuild = Join-Path $nxPlanVsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $nxPlanMsbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git add integrations/nx2212/include/result_reader.hpp integrations/nx2212/src/result_reader.cpp integrations/nx2212/tests/result_reader_test.cpp integrations/nx2212/tests/fixtures/result_success_1_1.json integrations/nx2212/BrepMFRNx2212RuntimeTests.vcxproj
git commit -m "feat: read face mapping from JSON 1.1"
```

Expected: 1.0、1.1 和扩展损坏测试全部通过。

---

### Task 6: Minimum-Cost Assignment Solver

**Files:**
- Create: `integrations/nx2212/include/assignment_solver.hpp`
- Create: `integrations/nx2212/src/assignment_solver.cpp`
- Create: `integrations/nx2212/tests/assignment_solver_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
struct AssignmentSolution {
    std::vector<std::size_t> column_for_row;
    double total_cost;
};

std::optional<AssignmentSolution> SolveMinimumCostAssignment(
    const std::vector<std::vector<double>>& costs);
```

非候选边使用 `std::numeric_limits<double>::infinity()`；输入必须是非空或 0x0 方阵、有限非负
成本或正无穷。

- [ ] **Step 1: 写求解器失败测试**

```cpp
BREPMFR_TEST(AssignmentSolverFindsGlobalNotGreedyMinimum)
{
    const std::vector<std::vector<double>> costs{
        {1.0, 2.0, 100.0},
        {1.1, 100.0, 100.0},
        {100.0, 1.0, 1.0}};
    const auto solution = SolveMinimumCostAssignment(costs);
    Expect(solution.has_value(), "perfect assignment missing");
    ExpectEqual(solution->column_for_row,
                std::vector<std::size_t>{1, 0, 2}, "greedy result returned");
    ExpectNear(solution->total_cost, 4.1, 1e-12, "cost mismatch");
}
```

增加置换方阵、0x0、非方阵、NaN、负成本、整行无有限值和不存在完美匹配测试。

- [ ] **Step 2: 构建确认 RED**

运行原生 RuntimeTests 重建。

Expected: 编译失败，因为求解器接口不存在。

- [ ] **Step 3: 实现 O(n^3) Hungarian 求解**

使用 potentials `u/v`、matching `p` 和 predecessor `way`；每轮只放松有限边。结束后验证每行
恰好一个列、列唯一且每条选中边有限，再计算 `total_cost`。0x0 返回空 assignment 和 0.0。

```cpp
for (std::size_t row = 1; row <= n; ++row) {
    p[0] = row;
    std::vector<double> minv(n + 1, infinity);
    std::vector<bool> used(n + 1, false);
    // update potentials until an unmatched column is reached
    // backtrack through way[] to augment exactly one row
}
```

- [ ] **Step 4: 运行测试并提交**

```powershell
$nxPlanVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$nxPlanVsRoot = & $nxPlanVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$nxPlanMsbuild = Join-Path $nxPlanVsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $nxPlanMsbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git add integrations/nx2212/include/assignment_solver.hpp integrations/nx2212/src/assignment_solver.cpp integrations/nx2212/tests/assignment_solver_test.cpp integrations/nx2212/*.vcxproj
git commit -m "feat: solve global face assignments"
```

Expected: 原生测试退出 0。

---

### Task 7: Geometry and Topology Face Matcher

**Files:**
- Create: `integrations/nx2212/include/face_matcher.hpp`
- Create: `integrations/nx2212/src/face_matcher.cpp`
- Create: `integrations/nx2212/tests/face_matcher_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
struct FaceMatchOptions {
    double relative_scalar_tolerance = 1e-4;
    double normalized_position_tolerance = 1e-5;
    double axis_angle_tolerance = 1e-6;
    double sample_distance_tolerance = 2e-3;
    double ambiguity_cost_margin = 1e-8;
};

struct FaceMatch {
    int occ_face_index;
    FaceKey nx_face_tag;
    double cost;
    double max_normalized_error;
};

struct FaceMappingResult {
    bool success = false;
    std::vector<FaceMatch> matches;
    double total_cost = 0.0;
    double max_error = 0.0;
    std::string diagnostic;
};

FaceMappingResult MatchFaces(
    const FaceGraph& occ_graph,
    const FaceGraph& nx_graph,
    const FaceMatchOptions& options = {});
```

- [ ] **Step 1: 写数组置换和全局映射失败测试**

创建四个签名不同但拓扑相同的面图，把 NX face vector 和 tags 打乱，断言 OCC 0..3 仍映射到
预期 tags，而不是 vector 位置。再加入平移/尺度已经规范化后的 `1e-7` 数值扰动，结果不变。

```cpp
const auto result = MatchFaces(occ, PermuteNx(nx, {2, 0, 3, 1}));
Expect(result.success, result.diagnostic);
ExpectEqual(TagFor(result, 0), FaceKey{101}, "face 0 tag mismatch");
```

- [ ] **Step 2: 构建确认 RED**

运行原生 RuntimeTests 重建。

Expected: 编译失败，因为 matcher 尚不存在。

- [ ] **Step 3: 实现候选门控和连续代价**

候选必须满足面型、edge_count、loop_count、neighbor_count 和解析字段存在性。连续分量先各自
检查容差，再用容差归一化求和：

```cpp
const double area_error = RelativeError(occ.area, nx.area);
const double centroid_error = Distance(occ.centroid, nx.centroid);
const double sample_error = SymmetricHausdorff(
    occ.sample_points, nx.sample_points);
if (area_error > options.relative_scalar_tolerance ||
    centroid_error > options.normalized_position_tolerance ||
    sample_error > options.sample_distance_tolerance)
    return std::nullopt;
return area_error / options.relative_scalar_tolerance +
       centroid_error / options.normalized_position_tolerance +
       sample_error / options.sample_distance_tolerance;
```

面积、周长、bbox、解析位置/半径/角度和方向都进入代价；轴比较使用
`acos(clamp(abs(dot(a,b)),0,1))`。每个 `FaceMatch.max_normalized_error` 是该候选全部“误差/对应
容差”分量的最大值，成功结果的 `max_error` 是所有 FaceMatch 的最大值，供 Listing Window
显示和黄金验收记录。

- [ ] **Step 4: 接入全局 assignment 和邻接验证**

构造 NxN 成本矩阵，调用 `SolveMinimumCostAssignment`。把 OCC 邻接端点通过结果转换为 NX
tag 对并规范化排序，要求与 NX graph 邻接完全相等；否则清空 matches 并返回失败诊断。

- [ ] **Step 5: 写歧义和失败封闭测试并观察 RED**

覆盖缺面、多面、错误类型、空候选、断裂邻接、超容差、重复几何唯一拓扑解、重复几何多解。
最后一个必须失败且 `matches.empty()`：

```cpp
Expect(!ambiguous.success, "ambiguous mapping was accepted");
Expect(ambiguous.matches.empty(), "ambiguous mapping leaked partial matches");
ExpectContains(ambiguous.diagnostic, "ambiguous", "reason missing");
```

- [ ] **Step 6: 实现拓扑有效的次优解检查**

用 Murty 分支枚举 `best.total_cost + ambiguity_cost_margin` 以内的全部 assignment：优先队列
节点保存固定的 row→column 前缀、禁止边集合和该约束下的 Hungarian 最优解；弹出一个解后，
依次固定它的前缀并禁止下一条选中边，生成互不遗漏的子问题。队首成本超过阈值即停止。

```cpp
while (!queue.empty() &&
       queue.top().solution.total_cost <= best.total_cost + options.ambiguity_cost_margin) {
    const auto alternative = queue.top();
    queue.pop();
    if (alternative.assignment != best.assignment &&
        AdjacencyMatches(occ_graph, nx_graph, alternative.assignment))
        return Failure("ambiguous topology-valid face assignment");
    PushMurtyChildren(alternative, queue);
}
```

只有邻接验证也通过的不同解才算真实歧义；几何代价接近但拓扑无效的解不触发误报。所有
Murty 子问题仍由 `SolveMinimumCostAssignment` 求解，不引入第三方优化库。

- [ ] **Step 7: 运行测试并提交**

```powershell
$nxPlanVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$nxPlanVsRoot = & $nxPlanVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$nxPlanMsbuild = Join-Path $nxPlanVsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $nxPlanMsbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git add integrations/nx2212/include/face_matcher.hpp integrations/nx2212/src/face_matcher.cpp integrations/nx2212/tests/face_matcher_test.cpp integrations/nx2212/*.vcxproj
git commit -m "feat: match OCC and NX face graphs"
```

Expected: 原生测试退出 0，所有失败用例都没有部分 matches。

---

### Task 8: Deterministic Color Plan and Mapping Display

**Files:**
- Create: `integrations/nx2212/include/color_plan.hpp`
- Create: `integrations/nx2212/src/color_plan.cpp`
- Create: `integrations/nx2212/tests/color_plan_test.cpp`
- Modify: `integrations/nx2212/include/result_formatter.hpp`
- Modify: `integrations/nx2212/src/result_formatter.cpp`
- Modify: `integrations/nx2212/tests/result_formatter_test.cpp`
- Modify: both NX2212 `.vcxproj` files

**Interfaces:**

```cpp
struct FaceColorAssignment {
    FaceKey nx_face_tag;
    int instance_id;
    int color_index;
};

struct InstanceColorRecord {
    int instance_id;
    int color_index;
    std::vector<FaceKey> nx_face_tags;
};

struct ColorPlan {
    bool success = false;
    std::vector<FaceColorAssignment> assignments;
    std::vector<InstanceColorRecord> instances;
    std::string diagnostic;
};

ColorPlan BuildColorPlan(
    const SuccessResult& recognition,
    const FaceMappingResult& mapping);

enum class MappingDisplayStatus { Unsupported, Unavailable, Failed, Colored };
struct MappingDisplay {
    MappingDisplayStatus status;
    std::string diagnostic;
    std::size_t mapped_faces;
    std::size_t colored_instances;
    double max_error;
    std::chrono::milliseconds elapsed;
    std::vector<InstanceColorRecord> instances;
};
```

`FormatSuccessDisplay` 增加四参数 overload：

```cpp
DisplayText FormatSuccessDisplay(
    const SuccessResult& result,
    const std::filesystem::path& result_path,
    std::chrono::milliseconds process_elapsed,
    const MappingDisplay& mapping);
```

现有三参数 overload 在 Task 8 保留并转发一个 Unsupported display，使当前 plugin_runtime 在
Task 9 接入映射前仍能编译和显示文字；Task 9 改为调用四参数 overload。

- [ ] **Step 1: 写颜色计划失败测试**

用两个实例和完整映射断言同实例同色、不同实例不同色、stock 面没有 assignment、输出按
instance_id 和 Face Tag 排序。重复调用必须字节级相同。映射失败、实例引用缺失和 217 个实例
必须 `success=false` 且 assignments 为空。

- [ ] **Step 2: 构建确认 RED**

运行原生 RuntimeTests 重建。

Expected: 编译失败，因为 `color_plan.hpp` 不存在。

- [ ] **Step 3: 实现固定 216 色计划**

将实例按 `instance_id` 排序，第 N 个实例使用固定互异调色板第 N 项；首版调色板精确定义为
升序 NX color index `1, 2, ..., 216`，禁止循环。先完整验证全部实例和 tag 唯一性，再一次性
生成 assignments：

```cpp
if (instances.size() > palette.size())
    return Failure("feature instance count exceeds 216 distinct NX colors");
for (std::size_t ordinal = 0; ordinal < instances.size(); ++ordinal) {
    const int color = palette[ordinal];
    for (const int face_index : instances[ordinal].face_indices)
        pending.push_back({TagFor(mapping, face_index),
                           instances[ordinal].instance_id, color});
}
```

- [ ] **Step 4: 写格式化失败测试**

验证四类输出：JSON 1.0 显示“映射不可用：JSON 1.0”；unavailable/failed 显示“识别成功，但
面映射失败”及诊断且保留 OCC 实例行；colored 显示：

```text
Mapped faces: 2/2; max error: 0.125000; mapping: 0.4 s
[0] blind_hole: OCC faces [1], NX Face Tags [501], color 36, confidence 0.9500
```

- [ ] **Step 5: 实现格式并运行提交**

摘要中的识别面数、实例数、BIN 缓存和 Python 用时保持不变；附加映射面数、已着色实例数。
warning 仍使用信息结果路径，不伪装为 Python 失败。

```powershell
$nxPlanVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$nxPlanVsRoot = & $nxPlanVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$nxPlanMsbuild = Join-Path $nxPlanVsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $nxPlanMsbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git add integrations/nx2212/include/color_plan.hpp integrations/nx2212/src/color_plan.cpp integrations/nx2212/tests/color_plan_test.cpp integrations/nx2212/include/result_formatter.hpp integrations/nx2212/src/result_formatter.cpp integrations/nx2212/tests/result_formatter_test.cpp integrations/nx2212/*.vcxproj
git commit -m "feat: plan and format NX instance colors"
```

Expected: 原生测试退出 0。

---

### Task 9: Host-Neutral Current Part Workflow

**Files:**
- Modify: `integrations/nx2212/include/plugin_runtime.hpp`
- Modify: `integrations/nx2212/src/plugin_runtime.cpp`
- Rewrite: `integrations/nx2212/tests/plugin_runtime_test.cpp`

**Interfaces:**

```cpp
struct PartSnapshot {
    std::filesystem::path step_path;
    FaceKey part_tag;
    FaceKey body_tag;
    std::size_t face_count;
};

class PluginHost {
public:
    virtual ~PluginHost() = default;
    virtual std::filesystem::path ModuleDirectory() const = 0;
    virtual std::optional<std::filesystem::path> SelectStepFile() = 0;
    virtual PartSnapshot CaptureCurrentPart() {
        throw std::runtime_error("NX current Part adapter is not connected");
    }
    virtual FaceGraph ReadCurrentFaceGraph(const PartSnapshot&) {
        throw std::runtime_error("NX face signature adapter is not connected");
    }
    virtual void ApplyFaceColorsAtomically(
        const PartSnapshot&,
        const std::vector<FaceColorAssignment>&) {
        throw std::runtime_error("NX coloring adapter is not connected");
    }
    virtual ProcessExecution RunProcess(...) = 0;
    virtual void ShowSuccess(const DisplayText& display) = 0;
    virtual void ShowError(const std::string& message_utf8) noexcept = 0;
};
```

Task 9 先保留但不再调用 `SelectStepFile()`，使现有 `NxPluginHost` 在 NX 适配完成前仍可编译；
三个新方法使用上述明确的安全失败默认实现。Task 10 覆盖当前 Part 和签名，Task 11 覆盖着色
并删除废弃的 selector 接口和 Win32 对话框实现。最终生产路径不存在默认回退。

- [ ] **Step 1: 改写 RecordingHost 并写当前 Part 失败测试**

`RecordingHost` 提供固定 `PartSnapshot`、可选 NX FaceGraph 和调用记录。测试 CaptureCurrentPart
抛出“Work Part unavailable”时，断言 Python、图提取、着色和成功显示均未发生，错误只显示一次。

- [ ] **Step 2: 运行确认 RED**

运行原生 RuntimeTests 重建。

Expected: 编译失败，因为 `PartSnapshot` 和新 workflow 接口尚不存在。

- [ ] **Step 3: 以当前 STEP 路径替换选择器**

workflow 第一项业务操作改为 `const auto part = host.CaptureCurrentPart();`；输出路径、CLI 参数、
result path 一致性全部使用 `part.step_path`。不存在取消分支。

- [ ] **Step 4: 写 1.0、1.1 成功与失败封闭测试**

覆盖：

1. JSON 1.0：不调用 ReadCurrentFaceGraph/ApplyFaceColors，仍显示成功文字；
2. JSON 1.1 ready：调用图提取、MatchFaces、BuildColorPlan、一次原子着色，再显示 colored；
3. JSON 1.1 unavailable/invalid：不读 NX 图、不着色，显示映射警告；
4. matcher 无解/歧义：读图但不着色，保留文字；
5. 颜色计划失败：不着色；
6. ApplyFaceColorsAtomically 抛错：显示映射失败，不能再次调用着色；
7. Python 非零和 JSON 核心错误：沿用现有推理失败路径。

```cpp
ExpectEqual(host.process_options.display_input_path, host.part.step_path,
            "progress path did not use current Work Part");
ExpectEqual(host.color_calls, 1, "complete mapping was not colored once");
ExpectContains(host.success_display.summary_utf8, u8"已着色实例：1",
               "color summary missing");
```

- [ ] **Step 5: 实现编排和映射计时**

只在 payload Ready 时计时调用 NX 图、matcher 和 color plan；计划完整后调用 host 原子着色。
任何映射或着色异常转为 `MappingDisplayStatus::Failed`，不进入 Python 错误结果分支：

```cpp
MappingDisplay mapping = MappingFromPayload(success.mapping);
if (success.mapping.status == MappingPayloadStatus::Ready) {
    const auto started = std::chrono::steady_clock::now();
    const auto nx_graph = host.ReadCurrentFaceGraph(part);
    const auto matched = MatchFaces(success.mapping.graph, nx_graph);
    const auto plan = BuildColorPlan(success, matched);
    if (matched.success && plan.success) {
        host.ApplyFaceColorsAtomically(part, plan.assignments);
        mapping = ColoredDisplay(matched, plan, Elapsed(started));
    } else {
        mapping = FailedDisplay(matched, plan, Elapsed(started));
    }
}
```

- [ ] **Step 6: 运行测试并提交**

```powershell
$nxPlanVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$nxPlanVsRoot = & $nxPlanVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$nxPlanMsbuild = Join-Path $nxPlanVsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $nxPlanMsbuild integrations\nx2212\BrepMFRNx2212RuntimeTests.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
git add integrations/nx2212/include/plugin_runtime.hpp integrations/nx2212/src/plugin_runtime.cpp integrations/nx2212/tests/plugin_runtime_test.cpp
git commit -m "feat: recognize the current NX work part"
```

Expected: 原生测试退出 0；workflow 不再调用 selector，真实 NX adapter 仍按明确异常安全失败。

---

### Task 10: NX2212 Current Part and Geometry Adapter

**Files:**
- Create: `integrations/nx2212/include/nx_part_adapter.hpp`
- Create: `integrations/nx2212/src/nx_part_adapter.cpp`
- Modify: `integrations/nx2212/include/nx_plugin_host.hpp`
- Modify: `integrations/nx2212/src/nx_plugin_host.cpp`
- Modify: `integrations/nx2212/src/plugin_entry.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**

```cpp
PartSnapshot CaptureNxCurrentPart();
FaceGraph ReadNxFaceGraph(const PartSnapshot& expected);
void ValidateNxCurrentPart(const PartSnapshot& expected);
```

`NxPluginHost` 在本任务覆盖 `CaptureCurrentPart` 和 `ReadCurrentFaceGraph`，分别委托给上述
适配函数；Task 11 覆盖颜色应用并移除旧 selector。

- [ ] **Step 1: 写项目契约失败测试**

扩展 `test_nx2212_project_declares_compatible_portable_contract`，要求 DLL 项目包含
`nx_part_adapter.cpp`、`face_signature.cpp`、`face_matcher.cpp`、`assignment_solver.cpp`、
`color_plan.cpp`，链接 `libufun.lib`，并继续禁止硬编码 Siemens 安装路径。

- [ ] **Step 2: 运行 focused pytest 确认 RED**

```powershell
python -m pytest tests/test_nx2212_project.py::test_nx2212_project_declares_compatible_portable_contract -v
```

Expected: FAIL，因为 NX adapter 和 UFUN 链接尚不存在。

- [ ] **Step 3: 实现严格当前 Part 捕获**

使用：

```cpp
auto* session = NXOpen::Session::GetSession();
auto* part = session->Parts()->Work();
const auto step = std::filesystem::u8path(part->FullPath().GetUTF8Text());
for (auto* body : *part->Bodies()) bodies.push_back(body);
```

要求路径存在且后缀忽略 ASCII 大小写后为 `.step/.stp`；bodies 数量精确为 1；该 Body
`IsSolidBody()==true`、`IsSheetBody()==false`、`IsConvergentBody()==false`；`GetFaces()` 非空。
snapshot 保存 `part->Tag()`、`body->Tag()` 和 face_count。任何 NX 异常转换为包含校验阶段的
`std::runtime_error`。

- [ ] **Step 4: 实现再次校验和 NX 邻接图**

`ValidateNxCurrentPart` 重新获取 Work Part，逐项比较规范化路径、part/body tag 和面数。邻接
通过每个 `Edge::GetUnsortedFaces()` 建立规范化 Tag 对，只接受恰有两个面的实体共享边；
`Face::GetUnsortedEdges()` 给 edge_count，`UF_MODL_ask_face_loops` 给 loop_count。
`ReadNxFaceGraph` 的第一步必须调用 `ValidateNxCurrentPart(expected)`，防止推理后切换 Part。

- [ ] **Step 5: 实现 NX 几何签名**

- Body 和每个 Face 的绝对轴对齐精确 bbox 使用 `UF_MODL_ask_extreme` 沿 `±X/±Y/±Z` 查询，
  不使用会受用户 WCS 影响的 `UF_MODL_ask_bounding_box_exact(..., NULL_TAG, ...)`。
- `Session::Measurement()->GetFaceProperties({face}, 0.999, AlternateFaceRadius, true, ...)`
  获取面积、周长和 COG。
- `UF_MODL_ask_face_data` 获取面型、解析位置、轴、半径和圆锥/圆环第二参数。
- `UF_MODL_ask_face_uv_minmax` 生成 5x5 UV 网格，`UF_MODL_ask_uv_points_containment` 过滤
  trimmed face 外点，`UF_MODL_ask_face_props` 取得非空无序 sample points。
- 所有 UF 返回码通过 `UF_get_fail_message` 转换为异常；UF list 使用 RAII 调用
  `UF_MODL_delete_list`。

解析参数必须与 Python 使用同一规范表示。先把 UF 返回的位置和长度应用 Body
`UnitBoxTransform`，再在规范化坐标系内表示解析曲面：平面 `origin` 为平面上离规范化原点
最近的点；圆柱和旋转面 `origin` 为轴线上离规范化原点最近的点；圆锥先在原坐标中根据 UF
返回的轴上 point、该处 radius 和 half-angle 计算唯一 apex，再规范化并保存为 `origin`；球和
圆环保存规范化后的唯一中心。轴全部经过 `CanonicalAxis`，不能直接比较两个内核任意选择的
等价参数原点。

```cpp
const Vec3 cone_apex = Subtract(
    face_data_point,
    Scale(face_data_direction, radius / std::tan(half_angle)));
```

黄金样例对圆锥 apex 与 OCC 输出做数值断言；如果 UF 对负 half-angle 返回相反轴，先按原始
UF direction/angle 求 apex，再只对保存的 axis 做符号规范化。

取得 Body bbox 后用 `MakeUnitBoxTransform` 对全部位置、长度、面积、解析参数和样本进行一次
Body 级规范化，再执行上述平面/轴线 canonical-origin 计算；face key 使用
`static_cast<FaceKey>(face->Tag())`。

- [ ] **Step 6: 在入口初始化 UFUN 并构建验证**

在 `ufusr` 内用 RAII 保证成功 `UF_initialize()` 后最终调用 `UF_terminate()`，同时保留 ABI
catch-all 和立即卸载。DLL 项目增加 `libufun.lib` 和全部纯 C++/NX adapter 源文件；RuntimeTests
项目只编译纯 C++ 文件，不编译 NX adapter。

```powershell
python -m pytest tests/test_nx2212_project.py -q
```

Expected: 项目契约、RuntimeTests 重建执行、DLL 重建和导出检查全部通过。

- [ ] **Step 7: 提交 NX 几何适配**

```powershell
git add integrations/nx2212/include/nx_part_adapter.hpp integrations/nx2212/src/nx_part_adapter.cpp integrations/nx2212/include/nx_plugin_host.hpp integrations/nx2212/src/nx_plugin_host.cpp integrations/nx2212/src/plugin_entry.cpp integrations/nx2212/BrepMFRNx2212.vcxproj tests/test_nx2212_project.py
git commit -m "feat: extract current NX part face signatures"
```

---

### Task 11: Atomic NX Face Coloring

**Files:**
- Create: `integrations/nx2212/include/nx_coloring.hpp`
- Create: `integrations/nx2212/src/nx_coloring.cpp`
- Modify: `integrations/nx2212/include/plugin_runtime.hpp`
- Modify: `integrations/nx2212/tests/plugin_runtime_test.cpp`
- Modify: `integrations/nx2212/include/nx_plugin_host.hpp`
- Modify: `integrations/nx2212/src/nx_plugin_host.cpp`
- Modify: `integrations/nx2212/BrepMFRNx2212.vcxproj`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**

```cpp
void ApplyNxFaceColorsAtomically(
    const PartSnapshot& expected,
    const std::vector<FaceColorAssignment>& assignments);
```

- [ ] **Step 1: 写构建契约失败测试**

要求 DLL 项目包含 `nx_coloring.cpp`，源码文本包含 `DisplayModification`、`SetNewColor`、
`ApplyFaceColorsAtomically`，并保持 RuntimeTests 不链接 NXOpen。

```powershell
python -m pytest tests/test_nx2212_project.py::test_nx2212_project_declares_compatible_portable_contract -v
```

Expected: FAIL，因为颜色适配文件尚不存在。

- [ ] **Step 2: 完整预校验 assignments**

函数开头调用 `ValidateNxCurrentPart(expected)`，重新枚举唯一 Body 的 faces 建立
`tag -> NXOpen::Face*`。在任何显示修改前要求：assignment tags 唯一、全部存在于当前 Body、
color 在 `[1,216]`、instance_id 非负。空 assignments 直接返回。

- [ ] **Step 3: 实现颜色保存、分组应用和失败恢复**

先记录每个目标面的 `face->Color()`，按新颜色分组。建立可见 undo mark，但不保存：

```cpp
const auto mark = session->SetUndoMark(
    NXOpen::Session::MarkVisibilityVisible,
    NXOpen::NXString("BrepMFR feature colors"));
for (const auto& [color, faces] : GroupByColor(assignments, face_by_tag)) {
    std::unique_ptr<NXOpen::DisplayModification> modification(
        session->DisplayManager()->NewDisplayModification());
    modification->SetApplyToAllFaces(false);
    modification->SetNewColor(color);
    modification->Apply(ToDisplayableObjects(faces));
}
```

catch 路径先尝试 `UndoToMark`，随后检查每个目标面的当前颜色，并把仍不等于保存值的面按
原颜色显式分组恢复和刷新；恢复异常附加到原错误诊断，最后删除失败操作的 undo mark。成功
路径保留一个名为 `BrepMFR feature colors` 的可见 undo mark，使用户能在 NX 中撤销本次
着色；无论哪条路径都不得调用 `BasePart::Save`。

- [ ] **Step 4: 接入 NxPluginHost、删除 selector 并构建**

`NxPluginHost::ApplyFaceColorsAtomically` 只委托上述函数。此时把 PluginHost 的三个 NX 方法改成
纯虚接口，删除 `SelectStepFile`；同步删除 RecordingHost 的 selector、NxPluginHost 的
`GetOpenFileNameW` 实现、`commdlg.h` 和 `comdlg32.lib`。`plugin_runtime_test.cpp` 必须继续断言
workflow 从未请求外部 STEP。把颜色源文件加入 DLL 项目，运行：

```powershell
python -m pytest tests/test_nx2212_project.py -q
```

Expected: RuntimeTests、DLL 构建和 exports 全部通过。

- [ ] **Step 5: 提交**

```powershell
git add integrations/nx2212/include/nx_coloring.hpp integrations/nx2212/src/nx_coloring.cpp integrations/nx2212/include/plugin_runtime.hpp integrations/nx2212/tests/plugin_runtime_test.cpp integrations/nx2212/include/nx_plugin_host.hpp integrations/nx2212/src/nx_plugin_host.cpp integrations/nx2212/BrepMFRNx2212.vcxproj tests/test_nx2212_project.py
git commit -m "feat: color matched NX faces atomically"
```

---

### Task 12: Operator Guide, Full Regression, and NX2212 Golden Acceptance

**Files:**
- Modify: `integrations/nx2212/README.md`
- Modify: `README.md`
- Modify: `tests/test_nx2212_project.py`

**Interfaces:**
- 操作入口为“先在 NX 打开 STEP，再 Ctrl+U 加载插件”。
- 输出目录和 `result.json` 仍由现有配置的 `output_root` 管理。
- 自动化验证必须在现场测试前全部通过。

- [ ] **Step 1: 写 README 契约失败测试**

在 `tests/test_nx2212_project.py` 读取集成 README，要求包含 `JSON 1.1`、
`input.face_signatures.json`、`41/41`、“不自动保存”、“映射失败”和“当前 Work Part”；同时要求
旧步骤原文“在 Unicode 文件对话框中选择现有 `.step` 或 `.stp` 文件”不存在。允许文档在
说明新行为时写“不再显示 STEP 文件选择器”。

- [ ] **Step 2: 运行 focused pytest 确认 RED**

```powershell
python -m pytest tests/test_nx2212_project.py -q
```

Expected: README 契约失败。

- [ ] **Step 3: 更新操作与故障排查文档**

说明单实体/未修改限制、当前 STEP FullPath 门槛、Python 与 NX 边界、sidecar 缓存、JSON 1.0
文字兼容、1.1 映射着色、颜色不自动保存、映射失败零着色，以及 Listing Window 中 OCC 面和
NX Face Tag 的区别。仓库 README 链接到集成指南。

- [ ] **Step 4: 用 DLL 本地配置运行 CLI 两次**

先读取忽略的真实配置，避免在计划或提交中硬编码 checkpoint：

```powershell
$release = Resolve-Path 'integrations\nx2212\build\Release'
$config = Get-Content -Raw -Encoding UTF8 (Join-Path $release 'BrepMFRNx2212.config.json') | ConvertFrom-Json
$smoke = 'D:\Code\BrepMFR\results\nx_face_mapping_smoke'
& $config.python_executable $config.inference_script --step 'tests\assets\step_inference\20221121_154647_0.step' --checkpoint $config.checkpoint --output-dir $smoke --device $config.device --affinity-threshold $config.affinity_threshold
& $config.python_executable $config.inference_script --step 'tests\assets\step_inference\20221121_154647_0.step' --checkpoint $config.checkpoint --output-dir $smoke --device $config.device --affinity-threshold $config.affinity_threshold
$result = Get-Content -Raw -Encoding UTF8 (Join-Path $smoke 'result.json') | ConvertFrom-Json
if ($result.schema_version -ne '1.1' -or $result.input.face_count -ne 41 -or -not $result.cache.hit -or $result.face_mapping.status -ne 'ready') { throw 'JSON 1.1 cache smoke failed' }
```

Expected: 两次退出 0，第二次 cache hit，签名 41 面且 sidecar 存在。

- [ ] **Step 5: 运行完整自动化验证**

```powershell
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\amd64\MSBuild.exe'
& $msbuild integrations\nx2212\BrepMFRNx2212.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64 /m
if ($LASTEXITCODE -ne 0) { throw "MSBuild failed: $LASTEXITCODE" }
& integrations\nx2212\build\Tests\BrepMFRNx2212RuntimeTests.exe
if ($LASTEXITCODE -ne 0) { throw "Native tests failed: $LASTEXITCODE" }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed: $LASTEXITCODE" }
git diff --check
git status --short
```

Expected: solution、原生测试和完整 pytest 均通过；diff check 无错误；状态只包含本计划的有意
修改。

- [ ] **Step 6: 在 NX2212 做黄金样例验收**

1. 在 NX2212 直接打开 `tests/assets/step_inference/20221121_154647_0.step`，不做修改；
2. Ctrl+U 加载 `integrations/nx2212/build/Release/BrepMFRNx2212.dll`；
3. 确认不出现 STEP 文件选择器，进度窗口显示当前 STEP；
4. 确认 `FullPath()` 门槛通过；若 NX 返回非 STEP 路径，记录实际值并停止，本版本不自动导出；
5. 确认识别 41 面、9 个实例，映射 `41/41`，映射阶段小于 1 秒；
6. 确认 9 个实例各自同色、实例间颜色不同、stock 面原色不变；
7. 确认 Listing Window 同时保留 OCC face_index、NX Face Tags、颜色、最大误差和计时；
8. 再次加载 DLL，确认缓存命中且映射和颜色一致；
9. 按下述 fixture wrapper 临时使用 JSON 1.0 和损坏的 1.1，确认文字保留且零颜色修改；
10. 确认插件未自动保存，关闭时由 NX 询问用户是否保存显示属性。

fixture 验收前，用 `apply_patch` 在被忽略的 `results/nx_fixture_runner.py` 创建以下本地脚本，不
提交 Git：

```python
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--step", required=True)
parser.add_argument("--output-dir", required=True)
arguments, _ = parser.parse_known_args()
root = Path(__file__).resolve().parents[1]
mode = os.environ["BREPMFR_NX_FIXTURE_MODE"]
fixture_name = "result_success.json" if mode == "1.0" else "result_success_1_1.json"
fixture = root / "integrations" / "nx2212" / "tests" / "fixtures" / fixture_name
step_text = Path(arguments.step).resolve().as_posix()
document = json.loads(
    fixture.read_text(encoding="utf-8").replace("__STEP_PATH__", step_text)
)
if mode == "invalid-1.1":
    document["face_mapping"]["faces"][0]["sample_points"] = []
output = Path(arguments.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
(output / "result.json").write_text(
    json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
```

备份 DLL 旁真实 config，把 `inference_script` 暂时改成该 wrapper 的绝对路径；分别在启动 NX
前设置 `$env:BREPMFR_NX_FIXTURE_MODE='1.0'` 和 `'invalid-1.1'` 并各运行一次。两次都应显示
识别核心文字、零颜色修改和对应映射诊断。随后恢复 config、关闭相关 NX 进程并删除该 ignored
wrapper；不要提交本机配置或 fixture runner。

记录实际首次/二次 Python 用时、映射用时、最大误差、41 个映射 Tag 和现场截图。只有用户
确认 NX 可见行为后，才声称本功能完成。

- [ ] **Step 7: 提交文档和最终项目检查**

```powershell
git add README.md integrations/nx2212/README.md tests/test_nx2212_project.py
git commit -m "docs: explain NX face mapping workflow"
git status --short
```

Expected: 最后提交成功，工作区干净。

---

## Final Review Checklist

- [ ] Python sidecar 与 BIN 使用同一次 AAG 提取，face_index 精确对齐。
- [ ] 缓存元数据版本 3 同时校验 BIN 和签名哈希，混合代际不能命中。
- [ ] 新 CLI 输出 JSON 1.1；C++ 仍能以文字模式读取 JSON 1.0。
- [ ] 映射扩展损坏不会丢失已验证的识别核心结果。
- [ ] 两端规范化精确使用 bbox center 与 `2 / longest extent`。
- [ ] matcher 不依赖面 vector 顺序，使用全局 assignment 和完整邻接验证。
- [ ] 无解、超差、邻接错误和歧义均返回空 matches 与空颜色计划。
- [ ] 当前 Work Part、路径、Body、Face 数量在推理前和着色前均重新验证。
- [ ] 颜色互异、stock 不修改、失败恢复、无自动保存。
- [ ] 文件选择器已从正常入口和测试中移除。
- [ ] Python focused/full pytest、原生 RuntimeTests、DLL Release/x64 构建和 exports 均通过。
- [ ] NX2212 黄金样例实测 41/41、9 实例、映射小于 1 秒、缓存命中结果一致。
