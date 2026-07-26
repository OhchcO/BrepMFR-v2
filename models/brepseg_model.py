# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pathlib
import os

from .modules.brep_encoder import BrepEncoder
from .modules.instance_bias import EXPERIMENT_MODES
from .modules.utils.macro import *


def feature_result_path(output_path, sample_id):
    """Build a result path from one validated full-stem string sample ID."""
    if (
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id in (".", "..")
        or "/" in sample_id
        or "\\" in sample_id
    ):
        raise ValueError("sample ID is not safe for a result filename")
    return pathlib.Path(output_path) / ("feature_" + sample_id + ".txt")

class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=dropout)
        self.linear4 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)
        x = F.softmax(x, dim=-1)
        return x


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def balanced_affinity_loss(logits, target, padding_mask, available=None):
    """Balanced pairwise BCE over valid upper-triangular face pairs."""
    graph_losses = []
    batch_size, max_nodes, _ = logits.shape
    upper_triangle = torch.triu(
        torch.ones(
            (max_nodes, max_nodes),
            device=logits.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    for graph_index in range(batch_size):
        if available is not None and not bool(available[graph_index].item()):
            continue
        valid_nodes = ~padding_mask[graph_index]
        valid_pairs = (
            valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(0) & upper_triangle
        )
        graph_logits = logits[graph_index][valid_pairs]
        graph_target = target[graph_index][valid_pairs].to(logits.dtype)
        if graph_logits.numel() == 0:
            continue

        positive = graph_target > 0.5
        negative = ~positive
        terms = []
        if bool(positive.any().item()):
            terms.append(F.softplus(-graph_logits[positive]).mean())
        if bool(negative.any().item()):
            terms.append(F.softplus(graph_logits[negative]).mean())
        graph_losses.append(torch.stack(terms).mean())

    if not graph_losses:
        return logits.sum() * 0.0
    return torch.stack(graph_losses).mean()


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = self.dense_weight(stacked)
        weights = F.softmax(weights, dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class BrepSeg(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes
        self.experiment_mode = getattr(args, "experiment_mode", "baseline")
        if self.experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(
                "Unsupported experiment mode: {}".format(self.experiment_mode)
            )
        self.affinity_loss_weight = getattr(args, "affinity_loss_weight", 0.1)
        self.affinity_warmup_epochs = getattr(
            args, "affinity_warmup_epochs", 10
        )

        self.brep_encoder = BrepEncoder(
            # < for graphormer
            num_degree=128,  # number of in degree types in the graph
            num_spatial=64,  # number of spatial types in the graph
            num_edge_dis=64,  # number of edge dis types in the graph
            edge_type="multi_hop",  # edge type in the graph "multi_hop"
            multi_hop_max_dist=16,  # max distance of multi-hop edges
            # >
            num_encoder_layers=args.n_layers_encode,  # num encoder layers
            embedding_dim=args.dim_node,  # encoder embedding dimension
            ffn_embedding_dim=args.d_model,  # encoder embedding dimension for FFN
            num_attention_heads=args.n_heads,  # num encoder attention heads
            dropout=args.dropout,  # dropout probability
            attention_dropout=args.attention_dropout,  # dropout probability for"attention weights"
            activation_dropout=args.act_dropout,  # dropout probability after"activation in FFN"
            layerdrop=0.1,
            encoder_normalize_before=True,  # apply layernorm before each encoder block
            pre_layernorm=True,
            # apply layernorm before self-attention and ffn. Without this, post layernorm will used
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
            experiment_mode=self.experiment_mode,
            affinity_dim=getattr(args, "affinity_dim", 64),
            affinity_temperature=getattr(args, "affinity_temperature", 0.2),
            instance_bias_scale=getattr(args, "instance_bias_scale", 1.0),
        )

        self.attention = Attention(args.dim_node)

        self.classifier = NonLinearClassifier(args.dim_node, args.num_classes, args.dropout)

        self.pred = []
        self.label = []

    def forward(self, batch):
        """Return per-face semantic probabilities for inference."""
        node_emb, graph_emb = self.brep_encoder(
            batch,
            last_state_only=True,
        )
        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]
        padding_mask = batch["padding_mask"]
        node_pos = torch.where(padding_mask == False)
        node_z = node_emb[node_pos]
        num_nodes_per_graph = torch.sum((~padding_mask).long(), dim=-1)
        graph_z = graph_emb.repeat_interleave(
            num_nodes_per_graph, dim=0
        ).to(graph_emb.device)
        z = self.attention([node_z, graph_z])
        return self.classifier(z)


    def training_step(self, batch, batch_idx):
        self.brep_encoder.train()
        self.attention.train()
        self.classifier.train()
        torch.cuda.empty_cache()

        # brep encoder----------------------------------------------------------------------------------
        node_emb, graph_emb, instance_logits = self.brep_encoder(
            batch,
            last_state_only=True,
            return_instance_logits=True,
        )

        # node classifier--------------------------------------------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node dim=0
        node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
        padding_mask = batch["padding_mask"]     # [batch_size, max_node_num]
        node_pos = torch.where(padding_mask == False)  # [(batch_size, node_index)]
        node_z = node_emb[node_pos]  # [total_nodes, dim_z]
        padding_mask_ = ~padding_mask
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)  # [batch_size]
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        z = self.attention([node_z, graph_z])
        node_seg = self.classifier(z) # [total_nodes, num_classes]

        # loss-------------------------------------------------------------------------------------------
        labels = batch["label_feature"].long()
        labels_onehot = F.one_hot(labels, self.num_classes)
        semantic_loss = CrossEntropyLoss(labels_onehot, node_seg)
        loss = semantic_loss
        self.log("train_semantic_loss", semantic_loss, on_step=False, on_epoch=True)
        if self.experiment_mode == "dynamic_affinity":
            if instance_logits is None:
                raise RuntimeError(
                    "dynamic_affinity mode did not produce affinity logits"
                )
            if not bool(batch["instance_available"].all().item()):
                raise ValueError(
                    "dynamic_affinity training requires instance annotations"
                )
            affinity_loss = balanced_affinity_loss(
                instance_logits,
                batch["instance"],
                batch["padding_mask"],
                batch["instance_available"],
            )
            if self.affinity_warmup_epochs > 0:
                warmup = min(
                    1.0,
                    float(self.current_epoch + 1)
                    / float(self.affinity_warmup_epochs),
                )
            else:
                warmup = 1.0
            affinity_weight = self.affinity_loss_weight * warmup
            loss = semantic_loss + affinity_weight * affinity_loss
            self.log(
                "train_affinity_loss",
                affinity_loss,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train_affinity_weight",
                affinity_weight,
                on_step=False,
                on_epoch=True,
            )
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.attention.eval()
        self.classifier.eval()
        torch.cuda.empty_cache()

        node_emb, graph_emb, _ = self.brep_encoder(
            batch,
            last_state_only=True,
            return_instance_logits=True,
        )  # logits [total_nodes, num_classes]

        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node dim=0
        node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
        padding_mask = batch["padding_mask"]     # [batch_size, max_node_num]
        node_pos = torch.where(padding_mask == False)  # [(batch_size, node_index)]
        node_z = node_emb[node_pos]  # [total_nodes, dim]
        padding_mask_ = ~padding_mask
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)  # [batch_size]
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        z = self.attention([node_z, graph_z])
        node_seg = self.classifier(z)  # [total_nodes, num_classes]

        labels = batch["label_feature"].long()  # labels [total_nodes]
        labels_np = labels.long().detach().cpu().numpy()
        labels_onehot = F.one_hot(labels, self.num_classes)
        loss = CrossEntropyLoss(labels_onehot, node_seg)
        self.log("eval_loss", loss, on_step=False, on_epoch=True)

        preds = torch.argmax(node_seg, dim=-1)  # pres [total_nodes]
        preds_np = preds.long().detach().cpu().numpy()
        for i in range(len(preds_np)): self.pred.append(preds_np[i])
        for i in range(len(labels_np)): self.label.append(labels_np[i])

        return loss

    def validation_epoch_end(self, val_step_outputs):
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []
        per_face_comp = (preds_np == labels_np).astype(np.float32)
        self.log("per_face_accuracy", np.mean(per_face_comp))

        class_ious = []
        class_f1s = []
        for class_index in range(self.num_classes):
            predicted = preds_np == class_index
            expected = labels_np == class_index
            true_positive = np.logical_and(predicted, expected).sum()
            false_positive = np.logical_and(predicted, ~expected).sum()
            false_negative = np.logical_and(~predicted, expected).sum()
            union = true_positive + false_positive + false_negative
            if union > 0:
                class_iou = true_positive / float(union)
                class_ious.append(class_iou)
                self.log(
                    "iou_class_{:02d}".format(class_index),
                    float(class_iou),
                )
            f1_denominator = 2 * true_positive + false_positive + false_negative
            if f1_denominator > 0:
                class_f1s.append(2 * true_positive / float(f1_denominator))

        if class_ious:
            self.log("mean_iou", float(np.mean(class_ious)))
        if class_f1s:
            self.log("macro_f1", float(np.mean(class_f1s)))

    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.attention.eval()
        self.classifier.eval()

        # brep encoder----------------------------------------------------------------------------------
        node_emb, graph_emb, _ = self.brep_encoder(
            batch,
            last_state_only=True,
            return_instance_logits=True,
        )  # logits [total_nodes, num_classes]

        # node classifier-------------------------------------------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node dim=0
        node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
        padding_mask = batch["padding_mask"]  # [batch_size, max_node_num]
        node_pos = torch.where(padding_mask == False)  # [(batch_size, node_index)]
        node_z = node_emb[node_pos]  # [total_nodes, dim]
        padding_mask_ = ~padding_mask
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)  # [batch_size]
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        z = self.attention([node_z, graph_z])
        node_seg = self.classifier(z)  # [total_nodes, num_classes]

        preds = torch.argmax(node_seg, dim=-1)  # pres [total_nodes]
        labels = batch["label_feature"].long()  # labels [total_nodes]
        known_pos = torch.where(labels < self.num_classes)
        labels_ = labels[known_pos]
        preds_ = preds[known_pos]
        labels_np = labels_.long().detach().cpu().numpy()
        preds_np = preds_.long().detach().cpu().numpy()

        for i in range(len(preds_np)): self.pred.append(preds_np[i])
        for i in range(len(labels_np)): self.label.append(labels_np[i])

        # 将结果转为txt文件----------------------------------------------------------------------------
        n_graph, max_n_node = batch["padding_mask"].size()[:2]
        node_pos = torch.where(batch["padding_mask"] == False)
        face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        face_feature[node_pos] = preds[:]
        out_face_feature = face_feature.long().detach().cpu().numpy()  # [n_graph, max_n_node]
        for i in range(n_graph):
            # 计算每个graph的实际n_node
            end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int))
            # masked出实际face feature
            pred_feature = out_face_feature[i][:end_index + 1]  # (n_node)

            output_path = pathlib.Path("/home/zhang/datasets_segmentation/2_val")
            file_path = feature_result_path(output_path, batch["data_ids"][i])
            feature_file = open(file_path, mode="a")
            for j in range(end_index):
                feature_file.write(str(pred_feature[j]))
                feature_file.write("\n")
            feature_file.close()

    def test_epoch_end(self, outputs):
        print("num_classes: %s" % self.num_classes)
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []

        per_face_comp = (preds_np == labels_np).astype(np.int)
        self.log("per_face_accuracy", np.mean(per_face_comp))
        print("per_face_accuracy: %s" % np.mean(per_face_comp))

        # pre-class acc-----------------------------------------------------------------------------
        per_class_acc = []
        for i in range (0, self.num_classes):
            class_pos = np.where(labels_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = preds_np[class_pos]
                class_i_label = labels_np[class_pos]
                per_face_comp = (class_i_preds == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i+1, np.mean(per_face_comp)))
        self.log("per_class_accuracy", np.mean(per_class_acc))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))

        # IoU---------------------------------------------------------------------------------------
        per_class_iou = []
        for i in range (0, self.num_classes):
            label_pos = np.where(labels_np == i)
            pred_pos = np.where(preds_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = preds_np[label_pos]
                class_i_label = labels_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int)
                Union = (class_i_preds != class_i_label).astype(np.int)
                class_i_preds_ = preds_np[pred_pos]
                class_i_label_ = labels_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int)
                per_class_iou.append(np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_)))
        self.log("IoU", np.mean(per_class_iou))
        print("IoU: %s" % np.mean(per_class_iou))

        # confusion_matrix---------------------------------------------------------------------------
        # output_path = pathlib.Path("/home/zhang/datasets_segmentation/confusion_matrix.txt")
        # result_file = open(output_path, mode="a")
        # for i in range(0, self.num_classes):
        #     class_pos = np.where(labels_np == i)
        #     if len(class_pos[0]) > 0:
        #         class_i_preds = preds_np[class_pos]
        #         for j in range(0, self.num_classes):
        #             per_face_comp = (class_i_preds == j).astype(np.int)
        #             acc_class_i = np.mean(per_face_comp)
        #             result_file.write(str(acc_class_i))
        #             if(j < self.num_classes-1):
        #                 result_file.write(" ")
        #         result_file.write("\n")
        # result_file.close()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.002, betas=(0.99, 0.999))

        # 学习策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               threshold=0.0001, threshold_mode='rel',
                                                               min_lr=0.000001, cooldown=2, verbose=False)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "eval_loss"}
                }

    # 逐渐增大学习率
    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu,
                       using_native_amp,
                       using_lbfgs,
                       ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < 5000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.002
