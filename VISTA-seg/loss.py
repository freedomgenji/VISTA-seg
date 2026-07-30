import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def task_loss(pred, label):
    dice_loss_f = DiceLoss()
    wce_f = WeightedCrossEntropyLoss()
    wce_loss = wce_f(pred, label)
    dice_loss = dice_loss_f(pred, label)
    return wce_loss, dice_loss


class BCELoss(nn.Module):
    def __init__(self, index=0):
        super(BCELoss, self).__init__()
        self.label_index = index
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        assert input.shape == target.shape
        tot_loss = 0
        for i in range(3):
            pred = input[:, i]
            gt = target[:, i]
            tot_loss += self.loss(pred, gt)
        return tot_loss


class DiceLoss(nn.Module):
    """Computes Dice loss for normalized probability maps."""

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)

    def forward(self, input, target):
        per_channel_dice = self.dice(input, target, weight=self.weight)
        return 1. - torch.mean(per_channel_dice)


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, weight=None, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

    def forward(self, input, target):
        per_channel_dice = self.dice(input, target, weight=self.weight)
        return 1. - torch.mean(per_channel_dice)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """Computes Dice coefficient for a multi-channel 3D prediction."""
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a tensor so the channel axis is first."""
    c = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(c, -1)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(target)
        target = torch.argmax(target, 1)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        flattened = flatten(input)
        nominator = flattened.sum(-1)
        denominator = flattened.sum()
        class_weights = Variable(1 - nominator / denominator, requires_grad=False)
        return class_weights
