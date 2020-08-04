import torch
from torch.nn import functional as F
import numpy as np
from .utils import make_onehot_torch
from scipy.ndimage.morphology import distance_transform_edt


class SoftDiceLossPyTorch(torch.nn.Module):
    def __init__(
        self,
        square_nom=False,
        square_denom=False,
        weight=None,
        smooth=1.0,
        reduction="elementwise_mean",
        non_lin=None,
    ):

        super().__init__()
        self.square_nom = square_nom
        self.square_denom = square_denom

        self.smooth = smooth
        if weight is not None:
            self.weight = np.array(weight)
        else:
            self.weight = None

        self.reduction = reduction
        self.non_lin = non_lin

    def forward(self, inp, target):

        n_classes = inp.shape[1]
        with torch.no_grad():
            target_onehot = make_onehot_torch(target, n_classes=n_classes)
        dims = tuple(range(2, inp.dim()))

        if self.non_lin is not None:
            inp = self.non_lin(inp)

        if self.square_nom:
            nom = torch.sum((inp * target_onehot.float()) ** 2, dim=dims)
        else:
            nom = torch.sum(inp * target_onehot.float(), dim=dims)
        nom = 2 * nom + self.smooth

        if self.square_denom:
            i_sum = torch.sum(inp ** 2, dim=dims)
            t_sum = torch.sum(target_onehot ** 2, dim=dims)
        else:
            i_sum = torch.sum(inp, dim=dims)
            t_sum = torch.sum(target_onehot, dim=dims)

        denom = i_sum + t_sum.float() + self.smooth

        frac = nom / denom

        if self.weight is not None:
            weight = torch.from_numpy(self.weight).to(
                dtype=inp.dtype, device=inp.device
            )
            frac = weight * frac

        frac = -torch.mean(frac, dim=1)

        if self.reduction == "elementwise_mean":
            return torch.mean(frac)
        if self.reduction == "none":
            return frac
        if self.reduction == "sum":
            return torch.sum(frac)
        raise AttributeError("Reduction parameter unknown.")


def get_dist_map(true_hot, in_object=None, factor=-1):
    distance_map = torch.zeros_like(true_hot)

    for i in range(true_hot.shape[0]):
        for j in range(true_hot.shape[1]):
            distance_map[i, j] = torch.tensor(
                distance_transform_edt((true_hot[i, j] != 1).cpu().numpy())
            )
            # distance_map[i, j][true_hot[i, j] == 1] = in_object[j]
            distance_map[i, j][true_hot[i, j] == 1] = factor * distance_map[i, j].max()

    return distance_map


class DistanceLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction="mean", **kwargs):
        super().__init__()

        if weight is None:
            self.weight = weight
        else:
            self.register_buffer("weight", torch.tensor(weight))
        self.reduction = reduction

    def forward(self, outputs, targets):
        weight = self.weight
        if weight is None or len(weight) != outputs.shape[1]:
            weight = torch.ones(
                outputs.shape[1], device=outputs.device, dtype=outputs.dtype
            )

        weight_matrix = torch.stack(
            [torch.ones_like(targets, dtype=weight.dtype) * x for x in weight], 1
        )

        num_classes = outputs.shape[1]
        true_hot = torch.unsqueeze(targets, 1)
        true_hot = torch.stack([targets == x for x in range(num_classes)], 1).to(
            weight.dtype
        )

        loss = outputs.softmax(dim=1) * get_dist_map(true_hot, factor=-1)
        loss = loss * weight_matrix / weight.sum()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BFocalLossPyTorch(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, p, t):

        bce_loss = F.binary_cross_entropy(p, t, reduction="none")

        if self.alpha is not None:
            # create weights for alpha
            alpha_weight = torch.ones(t.shape, device=p.device) * self.alpha
            alpha_weight = torch.where(torch.eq(t, 1.0), alpha_weight, 1 - alpha_weight)
        else:
            alpha_weight = torch.Tensor([1]).to(p.device)

        # create weights for focal loss
        focal_weight = 1 - torch.where(torch.eq(t, 1.0), p, 1 - p)
        focal_weight.pow_(self.gamma)
        focal_weight.to(p.device)

        # compute loss
        focal_loss = focal_weight * alpha_weight * bce_loss

        if self.reduction == "elementwise_mean":
            return torch.mean(focal_loss)
        if self.reduction == "none":
            return focal_loss
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        raise AttributeError("Reduction parameter unknown.")


class BFocalLossWithLogitsPyTorch(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, p, t):

        bce_loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")

        p = torch.sigmoid(p)

        if self.alpha is not None:
            # create weights for alpha
            alpha_weight = torch.ones_like(t) * self.alpha
            alpha_weight = torch.where(torch.eq(t, 1.0), alpha_weight, 1 - alpha_weight)
        else:
            alpha_weight = torch.Tensor([1]).to(p.device)

        # create weights for focal loss
        focal_weight = 1 - torch.where(torch.eq(t, 1.0), p, 1 - p)
        focal_weight.pow_(self.gamma)
        focal_weight.to(p.device)

        # compute loss
        focal_loss = focal_weight * alpha_weight * bce_loss

        if self.reduction == "elementwise_mean":
            return torch.mean(focal_loss)
        if self.reduction == "none":
            return focal_loss
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        raise AttributeError("Reduction parameter unknown.")


class FocalLossPyTorch(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):
        super().__init__()
        self.gamma = gamma
        self.nnloss_fn = torch.nn.NLLLoss(weight=alpha, reduction="none")
        self.reduction = reduction

    def forward(self, inp, target):
        n_classes = inp.shape[1]
        inp_log = torch.log(inp)
        nn_loss = self.nnloss_fn(inp_log, target)

        target_onehot = make_onehot_torch(target, n_classes=n_classes)
        focal_weights = ((1 - inp) * target_onehot.to(torch.float)).sum(
            dim=1
        ) ** self.gamma
        focal_loss = focal_weights * nn_loss
        if self.reduction == "elementwise_mean":
            return torch.mean(focal_loss)
        if self.reduction == "none":
            return focal_loss
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        raise AttributeError("Reduction parameter unknown.")


class FocalLossWithLogits(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):
        super().__init__()
        self.gamma = gamma
        self.ce_fn = torch.nn.CrossEntropyLoss(weight=alpha, reduction="none")
        self.reduction = reduction

    def forward(self, inp, target):
        n_classes = inp.shape[1]
        ce_loss = self.ce_fn(inp, target)
        inp = F.softmax(inp, dim=1)

        target_onehot = make_onehot_torch(target, n_classes=n_classes)
        focal_weights = ((1 - inp) * target_onehot.to(torch.float)).sum(
            dim=1
        ) ** self.gamma
        focal_loss = focal_weights * ce_loss
        if self.reduction == "elementwise_mean":
            return torch.mean(focal_loss)
        if self.reduction == "none":
            return focal_loss
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        raise AttributeError("Reduction parameter unknown.")


class LossMulti(torch.nn.Module):
    def __init__(
        self, jaccard_weight=0, class_weights=None, num_classes=1, device=None
    ):

        super().__init__()
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None
        self.nll_loss = torch.nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).to(outputs.dtype)
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                    torch.log((intersection + eps) / (union - intersection + eps))
                    * self.jaccard_weight
                )
        return loss


class LossPsiNet(torch.nn.Module):
    def __init__(self, weights: dict = None):
        super().__init__()

        if weights is None:
            weights = {"seg": 1, "contour": 1, "distance": 1}
        self.criterion_seg = LossMulti(num_classes=3)
        self.criterion_contour = LossMulti(num_classes=3)
        self.criterion_distance = torch.nn.MSELoss()
        self.weights = weights

    def __call__(
        self,
        outputs_seg,
        outputs_contour,
        outputs_distance,
        targets_seg,
        targets_contour,
        targets_distance,
    ):

        criterion = (
            self.weights["seg"] * self.criterion_seg(outputs_seg, targets_seg)
            + self.weights["contour"]
            * self.criterion_contour(outputs_contour, targets_contour)
            + self.weights["distance"]
            * self.criterion_distance(outputs_distance, targets_distance)
        )

        return criterion
