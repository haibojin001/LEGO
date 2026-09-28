import torch
from torch import nn

from ultralytics.utils.metrics import bbox_iou, smooth_bce
from utils.torch_utils import de_parallel


class BCEBlurWithLogitsLoss(nn.Module):
    """Binary cross-entropy loss with reduced penalty for probable missing labels."""

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        probability = torch.sigmoid(pred)
        delta = probability - true
        attenuation = 1.0 - torch.exp((delta - 1.0) / (self.alpha + 1e-4))
        return (loss * attenuation).mean()


class FocalLoss(nn.Module):
    """Focal-loss modifier for BCEWithLogitsLoss instances."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        probability = torch.sigmoid(pred)
        probability_of_target = true * probability + (1.0 - true) * (1.0 - probability)
        class_weight = true * self.alpha + (1.0 - true) * (1.0 - self.alpha)
        focus_weight = (1.0 - probability_of_target) ** self.gamma
        loss *= class_weight * focus_weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class QFocalLoss(nn.Module):
    """Quality focal-loss modifier for BCEWithLogitsLoss instances."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        probability = torch.sigmoid(pred)
        class_weight = true * self.alpha + (1.0 - true) * (1.0 - self.alpha)
        focus_weight = torch.abs(true - probability) ** self.gamma
        loss *= class_weight * focus_weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ComputeLoss:
    """Calculate YOLOv5 box, objectness, and classification losses."""

    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        hyperparameters = model.hyp

        classification_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyperparameters["cls_pw"]], device=device)
        )
        objectness_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyperparameters["obj_pw"]], device=device)
        )

        self.cp, self.cn = smooth_bce(eps=hyperparameters.get("label_smoothing", 0.0))

        gamma = hyperparameters["fl_gamma"]
        if gamma > 0:
            classification_criterion = FocalLoss(classification_criterion, gamma)
            objectness_criterion = FocalLoss(objectness_criterion, gamma)

        detect = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(detect.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(detect.stride).index(16) if autobalance else 0

        self.BCEcls = classification_criterion
        self.BCEobj = objectness_criterion
        self.gr = 1.0
        self.hyp = hyperparameters
        self.autobalance = autobalance
        self.na = detect.na
        self.nc = detect.nc
        self.nl = detect.nl
        self.anchors = detect.anchors
        self.device = device

    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for layer_index, prediction in enumerate(p):
            batch_index, anchor_index, grid_y, grid_x = indices[layer_index]
            object_targets = torch.zeros(
                prediction.shape[:4],
                dtype=prediction.dtype,
                device=self.device,
            )

            count = batch_index.shape[0]
            if count:
                xy, wh, _, classes = prediction[batch_index, anchor_index, grid_y, grid_x].split(
                    (2, 2, 1, self.nc), 1
                )

                xy = xy.sigmoid() * 2.0 - 0.5
                wh = (wh.sigmoid() * 2.0) ** 2 * anchors[layer_index]
                predicted_box = torch.cat((xy, wh), 1)

                iou = bbox_iou(predicted_box, tbox[layer_index], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(object_targets.dtype)
                if self.sort_obj_iou:
                    order = iou.argsort()
                    batch_index = batch_index[order]
                    anchor_index = anchor_index[order]
                    grid_y = grid_y[order]
                    grid_x = grid_x[order]
                    iou = iou[order]

                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou

                object_targets[batch_index, anchor_index, grid_y, grid_x] = iou

                if self.nc > 1:
                    class_targets = torch.full_like(classes, self.cn, device=self.device)
                    class_targets[range(count), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(classes, class_targets)

            objectness_loss = self.BCEobj(prediction[..., 4], object_targets)
            lobj += objectness_loss * self.balance[layer_index]

            if self.autobalance:
                self.balance[layer_index] = (
                    self.balance[layer_index] * 0.9999 + 0.0001 / objectness_loss.detach().item()
                )

        if self.autobalance:
            self.balance = [value / self.balance[self.ssi] for value in self.balance]

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]

        batch_size = object_targets.shape[0]
        return (lbox + lobj + lcls) * batch_size, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        number_of_anchors = self.na
        number_of_targets = targets.shape[0]

        target_classes = []
        target_boxes = []
        target_indices = []
        target_anchors = []

        gain = torch.ones(7, device=self.device)
        anchor_ids = (
            torch.arange(number_of_anchors, device=self.device)
            .float()
            .view(number_of_anchors, 1)
            .repeat(1, number_of_targets)
        )
        targets = torch.cat((targets.repeat(number_of_anchors, 1, 1), anchor_ids[..., None]), 2)

        bias = 0.5
        offsets_base = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ],
            device=self.device,
        ).float() * bias

        for layer_index in range(self.nl):
            anchors = self.anchors[layer_index]
            gain[2:6] = torch.tensor(p[layer_index].shape, device=self.device)[[3, 2, 3, 2]]
            scaled_targets = targets * gain

            if number_of_targets:
                ratios = scaled_targets[..., 4:6] / anchors[:, None]
                matched = torch.max(ratios, 1.0 / ratios).max(2)[0] < self.hyp["anchor_t"]
                scaled_targets = scaled_targets[matched]
            else:
                scaled_targets = targets[0]

            grid_xy = scaled_targets[:, 2:4]
            inverse_grid_xy = gain[[2, 3]] - grid_xy

            left, top = ((grid_xy % 1.0 < bias) & (grid_xy > 1.0)).T
            right, bottom = ((inverse_grid_xy % 1.0 < bias) & (inverse_grid_xy > 1.0)).T

            selection = torch.stack((torch.ones_like(left), left, top, right, bottom))
            scaled_targets = scaled_targets.repeat((5, 1, 1))[selection]
            offsets = (torch.zeros_like(grid_xy)[None] + offsets_base[:, None])[selection]

            batch, classes = scaled_targets[:, :2].long().T
            grid_xy = scaled_targets[:, 2:4]
            grid_wh = scaled_targets[:, 4:6]
            anchors_selected = scaled_targets[:, 6].long()

            grid_indices = (grid_xy - offsets).long()
            grid_x, grid_y = grid_indices.T

            target_indices.append(
                (
                    batch,
                    anchors_selected,
                    grid_y.clamp_(0, gain[3] - 1),
                    grid_x.clamp_(0, gain[2] - 1),
                )
            )
            target_boxes.append(torch.cat((grid_xy - grid_indices, grid_wh), 1))
            target_anchors.append(anchors[anchors_selected])
            target_classes.append(classes)

        return target_classes, target_boxes, target_indices, target_anchors