import dgl
import pytest
import torch
from dgl.data.utils import save_graphs

from brepmfr.inference import loading
from brepmfr.inference.loading import (
    SingleBinDataset,
    load_inference_batch,
    select_device,
)
from brepmfr.inference.postprocess import GeometryRequirements


def write_inference_contract_bin(path, labels):
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata["x"] = torch.zeros((2, 5, 5, 7), dtype=torch.float32)
    graph.ndata["y"] = torch.ones(2, dtype=torch.float32)
    graph.ndata["z"] = torch.zeros(2, dtype=torch.int32)
    graph.ndata["l"] = torch.ones(2, dtype=torch.int32)
    graph.ndata["a"] = torch.ones(2, dtype=torch.int32)
    graph.ndata["f"] = torch.tensor(labels, dtype=torch.int32)
    graph.edata["x"] = torch.zeros((2, 5, 7), dtype=torch.float32)
    graph.edata["t"] = torch.zeros(2, dtype=torch.int32)
    graph.edata["l"] = torch.ones(2, dtype=torch.float32)
    graph.edata["a"] = torch.zeros(2, dtype=torch.float32)
    graph.edata["c"] = torch.zeros(2, dtype=torch.int32)
    metadata = {
        "edges_path": torch.zeros((2, 2, 16), dtype=torch.long),
        "spatial_pos": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "d2_distance": torch.zeros((2, 2, 64), dtype=torch.float32),
        "angle_distance": torch.zeros((2, 2, 64), dtype=torch.float32),
    }
    save_graphs(str(path), [graph], metadata)
    return path


def test_single_bin_loader_accepts_only_unlabeled_sentinel(tmp_path):
    bin_path = write_inference_contract_bin(
        tmp_path / "input.bin", labels=[-1, -1]
    )

    batch = load_inference_batch(
        bin_path,
        GeometryRequirements(True, False, False, False),
        num_classes=25,
        seed=42,
        device=torch.device("cpu"),
    )

    assert batch["label_feature"].tolist() == [-1, -1]
    assert batch["instance_available"].tolist() == [False]
    assert batch["padding_mask"].shape == (1, 2)
    assert batch["data_ids"] == ("input",)
    assert batch["graph"].device == torch.device("cpu")
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            assert value.device == torch.device("cpu")


def test_single_bin_dataset_is_manifest_free(tmp_path):
    bin_path = write_inference_contract_bin(
        tmp_path / "standalone.bin", labels=[-1, -1]
    )

    dataset = SingleBinDataset(
        bin_path,
        GeometryRequirements(True, False, False, False),
        num_classes=25,
        seed=42,
    )

    assert dataset.file_paths == [bin_path.resolve()]
    assert len(dataset) == 1
    assert dataset[0].data_id == "standalone"


def test_single_bin_loader_rejects_mixed_labels(tmp_path):
    bin_path = write_inference_contract_bin(
        tmp_path / "input.bin", labels=[-1, 1]
    )

    with pytest.raises(ValueError, match="only semantic sentinel -1"):
        load_inference_batch(
            bin_path,
            GeometryRequirements(True, False, False, False),
            num_classes=25,
            seed=42,
            device=torch.device("cpu"),
        )


def test_select_device_handles_auto_cpu_and_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(loading.torch.cuda, "is_available", lambda: False)

    assert select_device("auto") == torch.device("cpu")
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA.*not available"):
        select_device("cuda")
    with pytest.raises(ValueError, match="Unsupported device"):
        select_device("tpu")
