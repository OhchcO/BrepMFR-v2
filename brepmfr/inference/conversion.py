"""Convert one unlabeled STEP file into the inference BIN contract."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from dgl.data.utils import load_graphs, save_graphs

from brepmfr.inference.postprocess import GeometryRequirements
from tools.geometry_features import PointSamplingConfig, extract_geometry_features
from tools.mfrinstseg_to_bin import (
    DEFAULT_EXTRACTOR_FILE,
    DEFAULT_SCHEMA_FILE,
    BinRequirements,
    ConversionOptions,
    attach_geometry_features,
    build_graph,
    geometry_metadata,
    load_extractor,
    load_schema,
    shortest_path_metadata,
    surface_geometry_metadata,
    validate_bin,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_conversion_implementation_paths() -> Tuple[Path, ...]:
    """Return ordered source files whose bytes define STEP conversion."""
    return (
        Path(__file__).resolve(),
        (PROJECT_ROOT / "tools" / "AAGExtractor.py").resolve(),
        (PROJECT_ROOT / "tools" / "schema.json").resolve(),
        (PROJECT_ROOT / "tools" / "mfrinstseg_to_bin.py").resolve(),
        (PROJECT_ROOT / "tools" / "geometry_features.py").resolve(),
    )


def conversion_options_from_requirements(
    requirements: GeometryRequirements,
    seed: int,
) -> ConversionOptions:
    """Map checkpoint geometry requirements to reproducible conversion options."""
    return ConversionOptions(
        seed=seed,
        include_face_pointcloud=requirements.require_face_pointcloud,
        include_parametric_surface=requirements.require_parametric_surface,
        include_cylindrical_pos=requirements.require_cylindrical_pos,
    )


def validate_inference_bin(
    path: Path,
    requirements: GeometryRequirements,
    point_sampling: PointSamplingConfig,
    expected_nodes: Optional[int] = None,
) -> None:
    """Validate a BIN without requiring semantic or instance ground truth."""
    validate_bin(
        Path(path),
        expected_nodes,
        requirements=BinRequirements(
            require_instance=False,
            require_pointcloud=requirements.require_face_pointcloud,
            require_parametric_surface=requirements.require_parametric_surface,
            require_cylindrical=requirements.require_cylindrical_pos,
        ),
        point_sampling=point_sampling,
    )
    graphs, metadata = load_graphs(str(Path(path)))
    labels = graphs[0].ndata["f"]
    if not bool((labels == -1).all()):
        raise ValueError("Inference BIN must use only semantic sentinel -1")
    if "instance" in metadata:
        raise ValueError("Inference BIN must not contain instance ground truth")


def convert_step_to_inference_bin(
    step_path: Path,
    target_path: Path,
    requirements: GeometryRequirements,
    seed: int,
    extractor_file: Path = DEFAULT_EXTRACTOR_FILE,
    schema_file: Path = DEFAULT_SCHEMA_FILE,
) -> int:
    """Convert, validate, and atomically publish one unlabeled inference BIN."""
    step_path = Path(step_path).resolve()
    target_path = Path(target_path).resolve()
    if step_path.suffix.lower() not in (".step", ".stp"):
        raise ValueError("Input must be a .step or .stp file: {0}".format(step_path))
    if not step_path.is_file():
        raise FileNotFoundError("STEP file not found: {0}".format(step_path))
    if target_path.suffix.lower() != ".bin":
        raise ValueError("Inference target must use the .bin suffix: {0}".format(target_path))
    if target_path == step_path:
        raise ValueError("Inference BIN target must differ from the STEP input")

    options = conversion_options_from_requirements(requirements, seed)
    extractor_class, topology_utils = load_extractor(Path(extractor_file))
    schema = load_schema(Path(schema_file))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        extractor = extractor_class(
            step_file=step_path,
            attribute_schema=schema,
        )
        extraction = extractor.process()
        if not isinstance(extraction, dict):
            raise ValueError("AAG extractor must return a dictionary")
        extraction["_extractor"] = extractor
        graph = build_graph(extraction, None, schema, topology_utils)
        metadata = shortest_path_metadata(
            graph,
            graph.ndata["x"].cpu().numpy(),
        )
        if options.include_face_pointcloud or options.include_cylindrical_pos:
            geometry = extract_geometry_features(
                extraction,
                options.point_sampling,
                options.seed,
                step_path.stem,
                include_face_pointcloud=options.include_face_pointcloud,
                include_cylindrical_pos=options.include_cylindrical_pos,
                include_parametric_surface=options.include_parametric_surface,
            )
            attach_geometry_features(graph, geometry, options)
        if options.include_face_pointcloud:
            metadata.update(geometry_metadata(options.point_sampling))
        if options.include_parametric_surface:
            metadata.update(surface_geometry_metadata())
        save_graphs(str(temporary_path), [graph], metadata)
        face_count = int(graph.num_nodes())
        validate_inference_bin(
            temporary_path,
            requirements,
            options.point_sampling,
            expected_nodes=face_count,
        )
        temporary_path.replace(target_path)
        return face_count
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
