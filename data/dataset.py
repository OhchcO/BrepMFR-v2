# -*- coding: utf-8 -*-
import pathlib
from tqdm import tqdm
import random
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from prefetch_generator import BackgroundGenerator

from .collator import collator, collator_st
from .utils import get_random_rotation, rotate_uvgrid
from utils.reproducibility import DEFAULT_SEED, make_torch_generator, seed_worker


FACE_POINT_CAPACITY = 96


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CADSynth(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        random_rotate=False,
        num_class=25,
        require_instance=False,
        require_face_pointcloud=False,
        require_cylindrical_pos=False,
        seed=DEFAULT_SEED,
    ):
        assert split in ("train", "val", "test")
        if random_rotate and require_cylindrical_pos:
            raise ValueError(
                "random_rotate cannot be used with require_cylindrical_pos"
            )
        path = pathlib.Path(root_dir)
        self.split = split
        self.num_class = num_class
        self.random_rotate = random_rotate
        self.require_instance = require_instance
        self.require_face_pointcloud = require_face_pointcloud
        self.require_cylindrical_pos = require_cylindrical_pos
        self.seed = int(seed)
        self.file_paths = []
        self._get_filenames(path, filelist=split+".txt")

    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        with open(str(root_dir / f"{filelist}"), "r") as f:
            file_list = [x.strip() for x in f.readlines()]
        requested = set(file_list)
        paths_by_stem = {}
        for x in tqdm(sorted(root_dir.rglob("*.bin"))):
            if x.stem in requested:
                if x.stem in paths_by_stem:
                    raise ValueError(
                        "Multiple BIN files share sample stem {}".format(x.stem)
                    )
                paths_by_stem[x.stem] = x
        self.file_paths = [
            paths_by_stem[sample_id]
            for sample_id in file_list
            if sample_id in paths_by_stem
        ]
        print("Done loading {} files".format(len(self.file_paths)))


    @staticmethod
    def _required_node_field(graph, key, file_path):
        if key not in graph.ndata:
            raise KeyError(
                "{} is required for sample {}".format(
                    key, pathlib.Path(file_path).resolve()
                )
            )
        return graph.ndata[key]

    @staticmethod
    def _invalid_geometry(file_path, key, detail):
        raise ValueError(
            "Invalid {} for sample {}: {}".format(
                key, pathlib.Path(file_path).resolve(), detail
            )
        )

    def _load_optional_geometry(self, graph, file_path):
        geometry = {
            "face_points": None,
            "face_point_mask": None,
            "face_point_count": None,
            "cylindrical_pos": None,
        }
        node_count = graph.num_nodes()

        if self.require_face_pointcloud:
            points = self._required_node_field(graph, "face_points", file_path)
            mask = self._required_node_field(graph, "face_point_mask", file_path)
            counts = self._required_node_field(graph, "face_point_count", file_path)
            if tuple(points.shape) != (node_count, FACE_POINT_CAPACITY, 6):
                self._invalid_geometry(
                    file_path,
                    "face_points",
                    "expected shape ({}, {}, 6), got {}".format(
                        node_count, FACE_POINT_CAPACITY, tuple(points.shape)
                    ),
                )
            if points.dtype != torch.float32:
                self._invalid_geometry(file_path, "face_points", "expected float32")
            if not bool(torch.isfinite(points).all()):
                self._invalid_geometry(file_path, "face_points", "contains non-finite values")
            if tuple(mask.shape) != (node_count, FACE_POINT_CAPACITY):
                self._invalid_geometry(
                    file_path,
                    "face_point_mask",
                    "expected shape ({}, {}), got {}".format(
                        node_count, FACE_POINT_CAPACITY, tuple(mask.shape)
                    ),
                )
            if mask.dtype != torch.bool:
                self._invalid_geometry(file_path, "face_point_mask", "expected bool")
            if tuple(counts.shape) != (node_count,):
                self._invalid_geometry(
                    file_path,
                    "face_point_count",
                    "expected shape ({},), got {}".format(node_count, tuple(counts.shape)),
                )
            if counts.dtype != torch.int32:
                self._invalid_geometry(file_path, "face_point_count", "expected int32")
            if not bool(((counts >= 1) & (counts <= FACE_POINT_CAPACITY)).all()):
                self._invalid_geometry(
                    file_path,
                    "face_point_count",
                    "values must be in [1, {}]".format(FACE_POINT_CAPACITY),
                )
            expected_mask = (
                torch.arange(FACE_POINT_CAPACITY, device=counts.device)[None, :]
                < counts.to(torch.int64)[:, None]
            )
            if not torch.equal(mask, expected_mask):
                self._invalid_geometry(
                    file_path,
                    "face_point_mask",
                    "is inconsistent with face_point_count",
                )
            if not bool(points[~mask].eq(0).all()):
                self._invalid_geometry(
                    file_path, "face_points", "padded values must be zero"
                )
            normal_lengths = torch.linalg.norm(points[..., 3:6][mask], dim=1)
            if not torch.allclose(
                normal_lengths,
                torch.ones_like(normal_lengths),
                rtol=1e-4,
                atol=1e-4,
            ):
                self._invalid_geometry(
                    file_path, "face_points", "valid normals must be unit length"
                )
            geometry.update(
                face_points=points,
                face_point_mask=mask,
                face_point_count=counts,
            )

        if self.require_cylindrical_pos:
            cylindrical = self._required_node_field(
                graph, "cylindrical_pos", file_path
            )
            if tuple(cylindrical.shape) != (node_count, 4):
                self._invalid_geometry(
                    file_path,
                    "cylindrical_pos",
                    "expected shape ({}, 4), got {}".format(
                        node_count, tuple(cylindrical.shape)
                    ),
                )
            if cylindrical.dtype != torch.float32:
                self._invalid_geometry(
                    file_path, "cylindrical_pos", "expected float32"
                )
            if not bool(torch.isfinite(cylindrical).all()):
                self._invalid_geometry(
                    file_path, "cylindrical_pos", "contains non-finite values"
                )
            geometry["cylindrical_pos"] = cylindrical

        return geometry


    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        geometry = self._load_optional_geometry(graph, file_path)
        if(self.random_rotate):
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
            if self.require_face_pointcloud:
                geometry["face_points"] = rotate_uvgrid(
                    geometry["face_points"], rotation
                )
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)  # edge_data[num_edges, U_grid, pnt_feature]

        pyg_graph.face_points = geometry["face_points"]
        pyg_graph.face_point_mask = geometry["face_point_mask"]
        pyg_graph.face_point_count = geometry["face_point_count"]
        pyg_graph.cylindrical_pos = geometry["cylindrical_pos"]

        pyg_graph.face_type = graph.ndata["z"].type(torch.int)   # face_type[num_nodes]
        pyg_graph.face_area = graph.ndata["y"].type(torch.float) # face_area[num_nodes]
        pyg_graph.face_loop = graph.ndata["l"].type(torch.int)   # face_loop[num_nodes]
        pyg_graph.face_adj = graph.ndata["a"].type(torch.int)    # face_loop[num_nodes]
        pyg_graph.label_feature = graph.ndata["f"].type(torch.int)  # feature_type[num_nodes]

        pyg_graph.edge_type = graph.edata["t"].type(torch.int)   # edge_type[num_edges]
        pyg_graph.edge_len = graph.edata["l"].type(torch.float)  # edge_len[num_edges]
        pyg_graph.edge_ang = graph.edata["a"].type(torch.float)  # edge_ang[num_edges]
        pyg_graph.edge_conv = graph.edata["c"].type(torch.int)   # edge_conv[num_edges]

        dense_adj = graph.adj().to_dense().type(torch.int)
        n_nodes = graph.num_nodes()
        pyg_graph.node_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.edge_path = graphfile[1]["edges_path"]           # edge_input[num_nodes, num_nodes, max_dist]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]        # spatial_pos[num_nodes, num_nodes]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]        # d2_distance[num_nodes, num_nodes, 64]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]  # angle_distance[num_nodes, num_nodes, 64]

        instance = graphfile[1].get("instance")
        n_nodes = graph.num_nodes()
        if instance is None:
            if self.require_instance:
                raise KeyError(
                    "Instance annotation is required but missing in {}".format(file_path)
                )
            pyg_graph.instance = torch.zeros(
                (n_nodes, n_nodes), dtype=torch.float32
            )
            pyg_graph.instance_available = False
        else:
            if tuple(instance.shape) != (n_nodes, n_nodes):
                raise ValueError(
                    "Invalid instance shape {} for {} nodes in {}".format(
                        tuple(instance.shape), n_nodes, file_path
                    )
                )
            if not torch.equal(instance, instance.transpose(0, 1)):
                raise ValueError(
                    "Instance matrix must be symmetric in {}".format(file_path)
                )
            pyg_graph.instance = (instance > 0).type(torch.float32)
            pyg_graph.instance_available = True

        pyg_graph.data_id = pathlib.Path(file_path).stem

        if(torch.max(pyg_graph.label_feature) > 24 or torch.max(pyg_graph.label_feature) < 0):
            print(pyg_graph.data_id)

        return pyg_graph

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)
        return sample
        
    def _collate(self, batch):  #batch=({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
        return collator(
            batch,
            multi_hop_max_dist=16,  # multi_hop_max_dist: max distance of multi-hop edges 大于该值认为这两个节点没有关系，边编码为0
            spatial_pos_max=32,     # spatial_pos_max: max distance of multi-hop edges 大于该值认为这两个节点没有关系，空间编码降为0
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, seed_purpose=None):
        purpose = seed_purpose or "{}_loader".format(self.split)
        loader_args = dict(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=(self.split == "train"),
            pin_memory=True,
            persistent_workers=False,
            generator=make_torch_generator(self.seed, purpose),
            worker_init_fn=seed_worker,
        )
        if num_workers > 0:
            loader_args["prefetch_factor"] = 2
        return DataLoaderX(**loader_args)


class TransferDataset(Dataset):
    def __init__(
            self,
            root_dir_source,
            root_dir_target,
            split="train",
            random_rotate=False,
            num_class=25,
            open_set=0
    ):
        assert split in ("train", "val", "test")
        source_path = pathlib.Path(root_dir_source)
        target_path = pathlib.Path(root_dir_target)
        self.split = split
        self.random_rotate = random_rotate
        self.num_class = num_class
        self.open_set = bool(open_set)

        self.source_file_paths = []
        self.target_file_paths = []
        self._get_filenames(source_path, target_path)


    def _get_filenames(self, source_dir, target_dir):
        if self.split == "train":
            filelist_s = "s_train.txt"
            filelist_t = "t_train.txt"
        elif self.split == "val":
            filelist_s = "s_val.txt"
            filelist_t = "t_val.txt"
        elif self.split == "test":
            filelist_s = "s_test.txt"
            filelist_t = "t_test.txt"

        print(f"Loading source data...")
        with open(str(source_dir / f"{filelist_s}"), "r") as f:
            s_file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(source_dir.rglob("*.bin")):
            if x.stem in s_file_list:
                if (self.open_set):
                    bin_file = load_graphs(str(x))
                    face_labels = bin_file[0][0].ndata["f"].type(torch.int)  # [num_nodes]
                    if (torch.max(face_labels) > self.num_class):
                        continue
                self.source_file_paths.append(x)
        print("Done loading {} files".format(len(self.source_file_paths)))

        print(f"Loading target data...")
        with open(str(target_dir / f"{filelist_t}"), "r") as f:
            t_file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(target_dir.rglob("*.bin")):
            if x.stem in t_file_list:
                if (self.open_set):
                    bin_file = load_graphs(str(x))
                    face_labels = bin_file[0][0].ndata["f"].type(torch.int)  # [num_nodes]
                    if (torch.max(face_labels) > self.num_class):
                        continue
                self.target_file_paths.append(x)
        print("Done loading {} files".format(len(self.target_file_paths)))

        if self.split != "test":
            random.shuffle(self.source_file_paths)
            random.shuffle(self.target_file_paths)


    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        if (self.random_rotate):
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)  # edge_data[num_edges, U_grid, pnt_feature]

        pyg_graph.face_type = graph.ndata["z"].type(torch.int)  # face_type[num_nodes]
        pyg_graph.face_area = graph.ndata["y"].type(torch.float)  # face_area[num_nodes]
        pyg_graph.face_loop = graph.ndata["l"].type(torch.int)  # face_loop[num_nodes]
        pyg_graph.face_adj = graph.ndata["a"].type(torch.int)   # face_loop[num_nodes]
        pyg_graph.label_feature = graph.ndata["f"].type(torch.int)  # feature_type[num_nodes]

        pyg_graph.edge_type = graph.edata["t"].type(torch.int)  # edge_type[num_edges]
        pyg_graph.edge_len = graph.edata["l"].type(torch.float)  # edge_len[num_edges]
        pyg_graph.edge_ang = graph.edata["a"].type(torch.float)  # edge_ang[num_edges]
        pyg_graph.edge_conv = graph.edata["c"].type(torch.int)  # edge_conv[num_edges]

        dense_adj = graph.adj().to_dense().type(torch.int)
        n_nodes = graph.num_nodes()
        pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.edge_path = graphfile[1]["edges_path"]  # edge_input[num_nodes, num_nodes, max_dist]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]  # spatial_pos[num_nodes, num_nodes]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]  # d2_distance[num_nodes, num_nodes, 64]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]  # angle_distance[num_nodes, num_nodes, 64]

        pyg_graph.data_id = pathlib.Path(file_path).stem

        return pyg_graph


    def __len__(self):
        if self.split == "train":
            return max(len(self.source_file_paths), len(self.target_file_paths))
        else:
            return len(self.target_file_paths)

    def __getitem__(self, idx):
        idx_s = idx
        idx_t = idx
        if idx_s >= len(self.source_file_paths):
            idx_s = random.randint(0, len(self.source_file_paths)-1)
        if idx_t >= len(self.target_file_paths):
            idx_t = random.randint(0, len(self.target_file_paths)-1)

        fn_s = self.source_file_paths[idx_s]
        fn_t = self.target_file_paths[idx_t]

        sample_s = self.load_one_graph(fn_s)
        sample_t = self.load_one_graph(fn_t)
        sample = {"source_data": sample_s, "target_data": sample_t}
        return sample

    def _collate(self, batch):  # batch=({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
        return collator_st(
            batch,
            multi_hop_max_dist=16,  # multi_hop_max_dist: max distance of multi-hop edges 大于该值认为这两个节点没有关系，边编码为0
            spatial_pos_max=32,  # spatial_pos_max: max distance of multi-hop edges 大于该值认为这两个节点没有关系，空间编码降为0
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoaderX(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )
