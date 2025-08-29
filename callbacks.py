import torch
from pathlib import Path
from metrics import compute_metric
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint


class ClipWeights(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, _, _, _ = trainer, outputs, batch, batch_idx
        for layer in pl_module.layer_list[:-1]:
            layer.clip()


class ComputeMetric(Callback):
    def __init__(self):
        self.y_c_dict = {"y_pred": [], "y_label": [], "c_pred": [], "c_label": []}

    @staticmethod
    def compute_log(pl_module, y_c_dict, stage):
        y_pred = torch.stack(y_c_dict["y_pred"])
        y_label = torch.stack(y_c_dict["y_label"])
        c_pred = torch.stack(y_c_dict["c_pred"])
        c_label = torch.stack(y_c_dict["c_label"])

        (c_acc, c_f1), (y_acc, y_f1) = compute_metric(c_pred, y_pred, c_label, y_label)
        pl_module.log(f"{stage}/c_acc", c_acc)
        pl_module.log(f"{stage}/c_f1", c_f1)
        pl_module.log(f"{stage}/y_acc", y_acc)
        pl_module.log(f"{stage}/y_f1", y_f1)

    def record_y_c(self, outputs, batch):
        self.y_c_dict["y_pred"].extend(outputs["y"])
        self.y_c_dict["y_label"].extend(batch[1])
        self.y_c_dict["c_pred"].extend(outputs["c"])
        self.y_c_dict["c_label"].extend(batch[2])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, _, _ = trainer, pl_module, batch_idx
        self.record_y_c(outputs, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, _, _ = trainer, pl_module, batch_idx
        self.record_y_c(outputs, batch)

    def on_validation_epoch_end(self, _, pl_module):
        self.compute_log(pl_module, self.y_c_dict, "val")
        self.y_c_dict = {"y_pred": [], "y_label": [], "c_pred": [], "c_label": []}

    def on_test_epoch_end(self, _, pl_module):
        self.compute_log(pl_module, self.y_c_dict, "test")
        self.y_c_dict = {"y_pred": [], "y_label": [], "c_pred": [], "c_label": []}


class ExtractRule(Callback):
    def __init__(self, exp_name):
        save_dir = Path("./rules")
        save_dir.mkdir(exist_ok=True)
        self.save_path = save_dir / f"{exp_name}.txt"

    def on_predict_start(self, _, pl_module):
        for layer in pl_module.layer_list[:-1]:
            layer.node_activation_cnt = torch.zeros(
                layer.output_dim, dtype=torch.double, device=pl_module.device
            )
            layer.forward_tot = 0

    def on_predict_end(self, trainer, pl_module):
        feature_name = trainer.datamodule.concept_list
        label_name = trainer.datamodule.classes
        # for Binarize Layer, layer_list[0].rule_name == bound_name
        pl_module.layer_list[0].get_rule_name(feature_name)

        # for Union Layer
        for i in range(1, len(pl_module.layer_list) - 1):
            layer = pl_module.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = (
                None
                if layer.conn.skip_from_layer is None
                else layer.conn.skip_from_layer.rule_name
            )
            wrap_prev_rule = False if i == 1 else True  # do not warp the bound_name
            layer.get_rule_description(
                (skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule
            )

        # for LR Layer
        layer = pl_module.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)

        with open(self.save_path, "w") as file:
            print("RID", end="\t", file=file)
            for i, ln in enumerate(label_name):
                print("{}(b={:.4f})".format(ln, layer.bl[i]), end="\t", file=file)
            print("Support\tRule", file=file)
            for rid, w in layer.rule2weights:
                print(rid[1], end="\t", file=file)
                for li in range(len(label_name)):
                    print("{:.4f}".format(w[li]), end="\t", file=file)
                now_layer = pl_module.layer_list[-1 + rid[0]]
                print(
                    "{:.4f}".format(
                        (
                            now_layer.node_activation_cnt[layer.rid2dim[rid]]
                            / now_layer.forward_tot
                        ).item()
                    ),
                    end="\t",
                    file=file,
                )
                print(now_layer.rule_name[rid[1]], end="\n", file=file)
            print("#" * 60, file=file)


def get_callbacks(cfgs):
    callbacks = []
    callbacks.append(ComputeMetric())
    callbacks.append(ClipWeights())
    callbacks.append(ExtractRule(cfgs.exp_name))

    if cfgs.log.logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if cfgs.ckpt.saving:
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{cfgs.ckpt.save_dir}/{cfgs.exp_name}",
                save_weights_only=True,
                save_last=True,
            ),
        )

    return callbacks
