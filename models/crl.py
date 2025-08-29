import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils import get_backbones
from models.components import BinarizeLayer, UnionLayer, LRLayer, Connection


class CRL(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()

        self.opt_cfgs = cfgs.opt
        self.l2_weight = cfgs.model.l2_weight

        self.dim_list = [
            cfgs.data.n_concepts,
            cfgs.model.l1,
            cfgs.model.l2,
            cfgs.data.n_classes,
        ]
        self.use_not = cfgs.model.use_not
        self.use_skip = cfgs.model.use_skip

        self.layer_list = nn.ModuleList([])
        self.t = nn.Parameter(torch.log(torch.tensor([cfgs.model.temperature])))

        self.pre_concept_model, n_features = get_backbones(cfgs.model.backbone)
        self.concept_predictor = torch.nn.Linear(n_features, cfgs.data.n_concepts)

        for idx, dim in enumerate(self.dim_list):

            skip_from_layer = None
            if self.use_skip and idx >= 3:
                skip_from_layer = self.layer_list[-2]
                prev_layer_dim += skip_from_layer.output_dim

            if idx == 0:
                layer = BinarizeLayer(dim, self.use_not)
                layer_name = f"binary{idx}"
            elif idx == len(self.dim_list) - 1:
                layer = LRLayer(prev_layer_dim, dim)
                layer_name = f"lr{idx}"
            else:
                # The first logical layer does not use NOT if the binarization layer has already used NOT
                layer_use_not = True if idx != 1 else False
                layer = UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)
                layer_name = f"union{idx}"

            layer.conn = Connection(
                prev_layer=self.layer_list[-1] if len(self.layer_list) > 0 else None,
                is_skip_to_layer=False,
                skip_from_layer=skip_from_layer,
            )
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True

            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

        self.loss_concept = torch.nn.BCELoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.pre_concept_model(x)
        concepts = self.concept_predictor(features)
        x = concepts

        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x

        return x, concepts

    def bi_forward(self, x, count=False):
        features = self.pre_concept_model(x)
        concepts = self.concept_predictor(features)
        x = concepts

        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer.binarized_forward(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
            if count and layer.layer_type != "linear":
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]
        return x

    def l2_penalty(self):
        l2_penalty = 0.0
        for layer in self.layer_list[1:]:
            l2_penalty += layer.l2_norm()
        return l2_penalty

    def training_step(self, batch, _):
        x, y, c = batch

        y_bar, c_bar = self.forward(x)
        # trainable softmax temperature
        y_bar = y_bar / torch.exp(self.t)
        c_bar = torch.sigmoid(c_bar)

        concept_loss = self.loss_concept(c_bar, c)
        l2_loss = self.l2_weight * self.l2_penalty()
        rrl_loss = self.loss(y_bar, y)

        loss = concept_loss + l2_loss + rrl_loss

        self.log("concept_loss", concept_loss)
        self.log("l2_loss", l2_loss)
        self.log("rrl_loss", rrl_loss)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, _, _ = batch
        output, concepts = self.forward(x)
        concepts = torch.sigmoid(concepts)
        return {"y": output, "c": concepts}

    def test_step(self, batch, _):
        x, _, _ = batch
        output, concepts = self.forward(x)
        concepts = torch.sigmoid(concepts)
        return {"y": output, "c": concepts}

    def predict_step(self, batch, _):
        x, _, _ = batch
        output = self.bi_forward(x, count=True)
        return output

    def configure_optimizers(self):
        param_groups = [
            {
                "params": [
                    param
                    for name, param in self.named_parameters()
                    if "concept" in name
                ],
                "lr": self.opt_cfgs.lr / 100,
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    param
                    for name, param in self.named_parameters()
                    if "concept" not in name
                ],
                "weight_decay": 0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.opt_cfgs.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.opt_cfgs.max_epochs
        )
        return [optimizer], [scheduler]
