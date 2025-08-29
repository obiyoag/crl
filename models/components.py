import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass
from collections import defaultdict

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
INIT_L = 0.0


@dataclass
class Connection:
    prev_layer: Any = None
    is_skip_to_layer: bool = False
    skip_from_layer: Any = None


class GradGraft(torch.autograd.Function):
    """Implement the Gradient Grafting."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class Binarizer(torch.autograd.Function):

    @staticmethod
    def forward(_, concepts):
        hard_concepts = (concepts.detach() > 0.0).float()
        return hard_concepts

    @staticmethod
    def backward(_, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    def __init__(self, n_concepts, use_not):
        super(BinarizeLayer, self).__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.input_dim = n_concepts
        self.output_dim = 2 * n_concepts if use_not else n_concepts
        self.layer_type = "binarization"
        self.dim2id = {i: i for i in range(self.output_dim)}

    def forward(self, x):
        x = Binarizer.apply(x)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        pass

    def get_rule_name(self, concept_names):
        self.rule_name = []
        for i in range(self.n_concepts):
            self.rule_name.append(concept_names[i])
        if self.use_not:
            for i in range(self.n_concepts):
                self.rule_name.append("~" + concept_names[i])


class Product(torch.autograd.Function):
    """Tensor product function."""

    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, input_dim, output_dim):
        super(LRLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_type = "linear"
        self.rid2dim = None
        self.rule2weights = None

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.fc1(x)

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)

    def l1_norm(self):
        return torch.norm(self.fc1.weight, p=1)

    def l2_norm(self):
        return torch.sum(self.fc1.weight**2)

    def get_rule2weights(self, prev_layer, skip_connect_layer):
        prev_layer = self.conn.prev_layer
        skip_connect_layer = self.conn.skip_from_layer

        always_act_pos = prev_layer.node_activation_cnt == prev_layer.forward_tot
        merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        if skip_connect_layer is not None:
            shifted_dim2id = {
                (k + prev_layer.output_dim): (-2, v)
                for k, v in skip_connect_layer.dim2id.items()
            }
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [
                    always_act_pos,
                    (
                        skip_connect_layer.node_activation_cnt
                        == skip_connect_layer.forward_tot
                    ),
                ]
            )

        Wl, bl = list(self.fc1.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        self.rid2dim = rid2dim
        self.rule2weights = sorted(
            marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True
        )


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, input_dim, output_dim, use_not=False):
        super(ConjunctionLayer, self).__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.layer_type = "conjunction"

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.output_dim))
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarizer.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, input_dim, output_dim, use_not=False):
        super(DisjunctionLayer, self).__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.layer_type = "disjunction"

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.output_dim))
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - Product.apply(1 - x.unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarizer.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class UnionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, use_not=False):
        super(UnionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.use_not = use_not
        self.layer_type = "union"
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        self.con_layer = ConjunctionLayer(self.input_dim, output_dim, use_not=use_not)
        self.dis_layer = DisjunctionLayer(self.input_dim, output_dim, use_not=use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat(
            [self.con_layer.binarized_forward(x), self.dis_layer.binarized_forward(x)],
            dim=1,
        )

    def edge_count(self):
        con_Wb = Binarizer.apply(self.con_layer.W - THRESHOLD)
        dis_Wb = Binarizer.apply(self.dis_layer.W - THRESHOLD)
        return torch.sum(con_Wb) + torch.sum(dis_Wb)

    def l1_norm(self):
        return torch.sum(self.con_layer.W) + torch.sum(self.dis_layer.W)

    def l2_norm(self):
        return torch.sum(self.con_layer.W**2) + torch.sum(self.dis_layer.W**2)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = (
            self.node_activation_cnt
        )

        # get dim2id and rule lists of the conjunction layer and the disjunction layer
        # dim2id: dimension --> (k, rule id)
        con_dim2id, con_rule_list = extract_rules(
            prev_layer, skip_connect_layer, self.con_layer
        )
        dis_dim2id, dis_rule_list = extract_rules(
            prev_layer, skip_connect_layer, self.dis_layer, self.con_layer.W.shape[1]
        )

        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})

        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

    def get_rule_description(self, input_rule_name, wrap=False):
        """
        input_rule_name: (skip_connect_rule_name, prev_rule_name)
        """
        self.rule_name = []
        for rl, op in zip(self.rule_list, ("&", "|")):
            for rule in rl:
                name = ""
                for i, ri in enumerate(rule):
                    op_str = " {} ".format(op) if i != 0 else ""
                    layer_shift = ri[0]
                    not_str = ""
                    if ri[0] > 0:  # ri[0] == 1 or ri[0] == 2
                        layer_shift *= -1
                        not_str = "~"
                    var_str = ("({})" if (wrap or not_str == "~") else "{}").format(
                        input_rule_name[2 + layer_shift][ri[1]]
                    )
                    name += op_str + not_str + var_str
                self.rule_name.append(name)


def extract_rules(prev_layer, skip_connect_layer, layer, pos_shift=0):
    # dim2id = {dimension: rule_id} :
    dim2id = defaultdict(lambda: -1)
    rules = {}
    tmp = 0
    rule_list = []

    # Wb.shape = (output_dim, input_dim)
    Wb = (layer.W.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # merged_dim2id is the dim2id of the input (the prev_layer and skip_connect_layer)
    merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
    if skip_connect_layer is not None:
        shifted_dim2id = {
            (k + prev_layer.output_dim): (-2, v)
            for k, v in skip_connect_layer.dim2id.items()
        }
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})

    for ri, row in enumerate(Wb):
        # delete dead nodes
        no_activated = layer.node_activation_cnt[ri + pos_shift] == 0
        all_activated = layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot
        if no_activated or all_activated:
            dim2id[ri + pos_shift] = -1
            continue

        # rule[i] = (k, rule_id):
        #     k == -1: connects to a rule in prev_layer,
        #     k ==  1: connects to a rule in prev_layer (NOT),
        #     k == -2: connects to a rule in skip_connect_layer,
        #     k ==  2: connects to a rule in skip_connect_layer (NOT).
        rule = {}
        for i, w in enumerate(row):
            # deal with "use NOT", use_not_mul = -1 if it used NOT in that input dimension
            use_not_mul = 1
            if layer.use_not:
                if i >= layer.input_dim // 2:
                    use_not_mul = -1
                i = i % (layer.input_dim // 2)

            if w > 0 and merged_dim2id[i][1] != -1:
                rid = merged_dim2id[i]
                rule[(rid[0] * use_not_mul, rid[1])] = 1

        # give each unique rule an id, and save this id in dim2id
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list
