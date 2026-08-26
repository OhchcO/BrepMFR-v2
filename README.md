# BrepMFR

Code for BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation.

![The network architecture of BrepMFR](docs/img/network_architecture.jpg)

## About BrepMFR

BrepMFR, a novel deep learning network designed for machining feature recognition on B-rep models within the CAD/CAM domain. The original B-rep model is converted into a graph representation for network-friendly input, where graph nodes and edges respectively correspond to faces and edges of the original model. Subsequently, we leverage a graph neural network based on the Transformer architecture and graph attention mechanism to encode both local geometric shape and global topological relationships, achieving high-level semantic extraction and prediction of machining feature categories. Furthermore, to enhance the performance of neural networks on real-world CAD models, we adopt a two-step training strategy within a novel transfer learning framework.

## Preparation

### Environment setup

```
git clone https://github.com/zhangshuming0668/BrepMFR.git
cd BrepMFR
conda env create -f environment.yml
conda activate brep_mfr
```

### Data preparation

Our synthetic CAD datasets have been publicly available on [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=931c088fd44f4d3e82891a5180f10d90)

## Training

For machining feature recognition, the network can be trained using:
```
python segmentation.py train --dataset_path /path/to/dataset --max_epochs 1000 --batch_size 64
```

The logs and checkpoints will be stored in a folder called `results/BrepMFR` based on the experiment name and timestamp, and can be monitored with Tensorboard:

```
tensorboard --logdir results/<experiment_name>
```

## Testing

The best checkpoints based on the smallest validation loss are saved in the results folder. The checkpoints can be used to test the model as follows:

```
python segmentation.py test --dataset_path /path/to/dataset --checkpoint ./results/BrepMFR/best.ckpt --batch_size 64
```

## STEP inference

Run an affinity-capable checkpoint directly on one unlabeled STEP model:

```powershell
python tools/infer_step.py `
  --step D:\path\part.step `
  --checkpoint D:\path\model.ckpt `
  --output-dir D:\path\recognition-output `
  --device auto
```

The output directory contains `input.bin`, `input.bin.meta.json`, and the
canonical `result.json`. The BIN is reused only when the STEP bytes,
conversion implementation, sampling seed/configuration, and checkpoint-derived
geometry requirements still match and the cached BIN passes validation. Use
`--force-rebuild` to bypass a valid cache.

The inference path never reads MFRInstSeg label JSON files. In `result.json`,
`face_index` is the zero-based OCC/AAG graph-node order produced from the STEP
file; it is not an NX face-array index or Face Tag. NX/UG visualization can
consume the JSON later, but is not part of this Python inference pipeline.

### Golden STEP regression

The repository includes the small `20221121_154647_0.step` test-split model
and its content-addressed expected result. The ordinary test suite always
checks that the committed STEP bytes match the golden SHA-256. To run the
full numerical comparison against the final checkpoint, provide the three
large external assets explicitly:

```powershell
$env:BREPMFR_GOLDEN_LABEL = 'D:\path\20221121_154647_0.json'
$env:BREPMFR_GOLDEN_PREPARED_BIN = 'D:\path\20221121_154647_0.bin'
$env:BREPMFR_GOLDEN_CHECKPOINT = 'D:\path\best-model.ckpt'
python -m pytest tests/test_inference_golden.py -q
```

Without these variables, only the external numerical comparison is skipped;
the committed STEP integrity check still runs.

### NX 2212 integration

The optional NX2212 C++ plug-in recognizes the current, unmodified, single-body
STEP Work Part, launches the Python CLI as an isolated process, validates JSON
1.1 face signatures, maps OCC/AAG faces to NX Face Tags, and colors recognized
feature instances. Python remains responsible for STEP conversion, BIN caching,
model inference, and postprocessing; the plug-in does not automatically save the
NX Part. See the
[NX2212 build, configuration, and operation guide](integrations/nx2212/README.md).

The Listing Window preserves both OCC/AAG `face_index` values and mapped NX Face
Tags. JSON 1.0 remains text-only; mapping failure keeps the recognition text and
applies no partial coloring.

